"""
signal_engine/scoring_engine.py
────────────────────────────────────────────────────────────────────────────────
Multi-factor Signal Scoring Engine.

Combines inputs from:
  ┌─────────────────────┬────────┬──────────────────────────────────────────┐
  │ Factor              │ Weight │ Source                                   │
  ├─────────────────────┼────────┼──────────────────────────────────────────┤
  │ gex_signal          │  0.30  │ options_engine (GEX regime + distance)   │
  │ structure_quality   │  0.20  │ pattern_engine (harmonic, VCP, chart)    │
  │ momentum_strength   │  0.20  │ scanner_engine / price data              │
  │ cycle_alignment     │  0.15  │ cycle_engine / wave_phase                │
  │ volatility_state    │  0.15  │ market_regime / volatility_engine        │
  └─────────────────────┴────────┴──────────────────────────────────────────┘
  catalyst_score removed — fires <5% of scans, no meaningful average contribution.

Output per symbol:
  {
    "symbol":          str,
    "score":           float,    # 0.0 – 1.0 normalized
    "signal_type":     str,      # "long" | "short" | "neutral"
    "confidence":      str,      # "high" | "medium" | "low"
    "factors":         dict,     # raw factor scores for transparency
    "gex_context":     dict,     # GEX regime, gamma flip, dealer bias
    "explanation":     list[str],# human-readable reasoning
  }

GEX Signal Logic:
  +1.0  → Negative GEX + price breaking above gamma flip    (squeeze/trend)
  +0.8  → Negative GEX + strong momentum (trend acceleration)
  +0.7  → Price within 1% of gamma flip (imminent flip)
  +0.5  → Neutral / transitioning GEX
  +0.3  → Positive GEX (mean-reversion market — dampened moves)
  -0.2  → Positive GEX + overbought (expect pin / reversal)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Weight table ──────────────────────────────────────────────────────────────
# Philosophy: GEX is the most forward-looking signal (options market tells the
# future more than price history). Structure quality raised because harmonic
# patterns pass 5 independent gates — cheapening them at 10% was wrong.
# Cycle alignment trimmed: it's unvalidated vs GEX which has real edge.
# Catalyst removed from formula — fires <5% of the time, meaningless average weight.
WEIGHTS = {
    "cycle_alignment":   0.15,   # ↓ trimmed — unvalidated vs GEX
    "gex_signal":        0.30,   # ↑ options Greeks are the smartest signal
    "momentum_strength": 0.20,   # ← unchanged
    "volatility_state":  0.15,   # ← unchanged
    "structure_quality": 0.20,   # ↑↑ harmonic/VCP patterns deserve real weight
    # catalyst_score removed — fires <5% of scans, dead weight in the average
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# ── Confidence thresholds ─────────────────────────────────────────────────────
# Raised HIGH bar: GEX(30%) + structure(20%) + momentum(20%) all need to align.
_CONF_HIGH   = 0.75   # ↑ from 0.72 — fewer, better HIGH signals
_CONF_MEDIUM = 0.55   # ↑ from 0.52
_CONF_LOW    = 0.40   # ↑ from 0.35 — discard more aggressively


@dataclass
class SignalScore:
    symbol:       str
    score:        float          # 0.0 – 1.0
    signal_type:  str            # long | short | neutral
    confidence:   str            # high | medium | low | discard
    factors:      dict = field(default_factory=dict)
    gex_context:  dict = field(default_factory=dict)
    explanation:  list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol":       self.symbol,
            "score":        round(self.score, 4),
            "signal_type":  self.signal_type,
            "confidence":   self.confidence,
            "factors":      {k: round(v, 4) for k, v in self.factors.items()},
            "gex_context":  self.gex_context,
            "explanation":  self.explanation,
        }


# ── GEX Signal Component ──────────────────────────────────────────────────────

def _compute_gex_signal(
    gex_profile: dict | None,
    momentum_pct: float = 0.0,
    direction: str = "long",
) -> tuple[float, dict, list[str]]:
    """
    Translate GEX regime into a 0-1 signal score.

    Args:
        gex_profile: dict from GEXEngine / gamma_exposure or None
        momentum_pct: intraday % change (positive = up)
        direction: intended trade direction

    Returns:
        (gex_score, gex_context, notes)

    Scenarios:
      1. GAMMA SQUEEZE: negative GEX + price breaking above gamma flip
         → explosive upside, score = 0.95
      2. MEAN REVERSION: positive GEX + price near call/put wall
         → expect pin, fade the move, score depends on direction
      3. BREAKDOWN: negative GEX + price below put wall
         → accelerating downside, short score = 0.90
    """
    if gex_profile is None:
        return 0.5, {"regime": "unknown"}, ["GEX data unavailable — using neutral"]

    regime       = gex_profile.get("gamma_regime", "NEUTRAL")
    zero_gamma   = float(gex_profile.get("zero_gamma", 0))
    spot         = float(gex_profile.get("spot", 0))
    call_wall    = float(gex_profile.get("call_wall", 0))
    put_wall     = float(gex_profile.get("put_wall", 0))
    dealer_bias  = int(gex_profile.get("dealer_bias", 0))
    gex_flip_dist = float(gex_profile.get("gex_flip_dist", 99))

    notes:  list[str] = []
    ctx = {
        "regime":      regime,
        "gamma_flip":  zero_gamma,
        "call_wall":   call_wall,
        "put_wall":    put_wall,
        "dealer_bias": dealer_bias,
        "price":       spot,
        "flip_dist_pct": round(gex_flip_dist, 2),
    }

    # ── Scenario 1: GAMMA SQUEEZE ─────────────────────────────────────────
    # Negative GEX + upward breakout + price crossed above gamma flip
    if regime == "NEGATIVE" and momentum_pct > 2.0 and spot > zero_gamma > 0:
        score = 0.95
        notes.append("🚀 GAMMA SQUEEZE: negative GEX + price above gamma flip + strong momentum")
        notes.append(f"   Dealers must buy as price rises → self-reinforcing move")
        ctx["scenario"] = "gamma_squeeze"
        return score, ctx, notes

    # ── Scenario 2: MEAN REVERSION (positive GEX) ────────────────────────
    if regime == "POSITIVE":
        if call_wall > 0 and abs(spot - call_wall) / spot < 0.01:
            score = 0.30 if direction == "long" else 0.70
            notes.append(f"📌 MEAN REVERSION: positive GEX + price at call wall ${call_wall}")
            notes.append(f"   Dealers long gamma → sell into rallies (pinning effect)")
            ctx["scenario"] = "mean_reversion_call_wall"
        elif put_wall > 0 and abs(spot - put_wall) / spot < 0.01:
            score = 0.70 if direction == "long" else 0.30
            notes.append(f"📌 MEAN REVERSION: positive GEX + price at put wall ${put_wall}")
            ctx["scenario"] = "mean_reversion_put_wall"
        else:
            score = 0.35
            notes.append("Positive GEX market — moves dampened by dealer hedging")
            ctx["scenario"] = "positive_gex_stable"
        return score, ctx, notes

    # ── Scenario 3: BREAKDOWN (negative GEX + below put wall) ────────────
    if regime == "NEGATIVE" and put_wall > 0 and spot < put_wall:
        score = 0.15 if direction == "long" else 0.90
        notes.append(f"💥 BREAKDOWN: negative GEX + price below put wall ${put_wall}")
        notes.append("   Dealers must sell as price falls → acceleration downside")
        ctx["scenario"] = "breakdown"
        return score, ctx, notes

    # ── Gamma flip proximity ──────────────────────────────────────────────
    if gex_flip_dist < 1.0:
        score = 0.70
        notes.append(f"⚡ FLIP ZONE: {gex_flip_dist:.1f}% from gamma flip ${zero_gamma:.2f}")
        notes.append("   Market transitioning — directional move imminent")
        ctx["scenario"] = "near_gamma_flip"
        return score, ctx, notes

    # ── Default: negative GEX + trending ─────────────────────────────────
    if regime == "NEGATIVE":
        score = 0.75 if abs(momentum_pct) > 1.0 else 0.60
        notes.append("Negative GEX regime — trend moves amplified")
        ctx["scenario"] = "negative_gex_trending"
    else:
        score = 0.50
        notes.append("Neutral/unknown GEX regime")
        ctx["scenario"] = "neutral"

    return score, ctx, notes


# ── Factor computation helpers ────────────────────────────────────────────────

def _momentum_score(change_pct: float, vol_ratio: float) -> float:
    """Normalize momentum to 0-1. Combines price change + volume confirmation."""
    pct_score = float(np.clip(abs(change_pct) / 10.0, 0, 1))  # 10% move = max score
    vol_score = float(np.clip((vol_ratio - 1.0) / 4.0, 0, 1))  # 5× avg vol = max score
    return round((pct_score * 0.6 + vol_score * 0.4), 4)


def _volatility_score(vix: float | None, atr_ratio: float | None) -> float:
    """
    Score volatility state for signal quality.
    Moderate volatility (VIX 15-25) is ideal — score 0.7-0.9.
    Too low (VIX<12) or too high (VIX>40) reduces score.
    """
    if vix is None:
        return 0.5
    if 15 <= vix <= 25:
        return 0.85
    if 12 <= vix < 15 or 25 < vix <= 35:
        return 0.65
    if vix < 12:
        return 0.40   # too calm — moves don't extend
    return 0.30       # too volatile — unpredictable


def _cycle_score(cycle_data: dict | None) -> float:
    """Extract cycle alignment score from cycle_engine output."""
    if cycle_data is None:
        return 0.5
    raw = float(cycle_data.get("alignment_score", cycle_data.get("score", 50)) or 50)
    return float(np.clip(raw / 100.0, 0, 1))


def _structure_score(pattern_data: dict | None) -> float:
    """Extract structure quality from pattern_engine output."""
    if pattern_data is None:
        return 0.5
    conf = float(pattern_data.get("confidence", 50) or 50)
    return float(np.clip(conf / 100.0, 0, 1))


# ── Main scorer ───────────────────────────────────────────────────────────────

class ScoringEngine:
    """
    Multi-factor signal scoring engine.

    All inputs are optional — missing data defaults to neutral (0.5).
    Only cycle_alignment and gex_signal require non-trivial data.
    """

    def __init__(self, weights: dict | None = None):
        self.weights = weights or WEIGHTS

    def score(
        self,
        symbol:       str,
        *,
        # Scanner inputs
        price:        float       = 0.0,
        change_pct:   float       = 0.0,
        vol_ratio:    float       = 1.0,
        direction:    str         = "long",  # intended direction from scanner
        # Options inputs
        gex_profile:  dict | None = None,
        # Market context inputs
        vix:          float | None = None,
        atr_ratio:    float | None = None,
        # Engine inputs
        cycle_data:   dict | None = None,
        pattern_data: dict | None = None,
        catalyst_data: dict | None = None,
    ) -> SignalScore:
        """
        Score a trade candidate from all available data sources.

        Returns SignalScore with normalized score 0-1, signal type, confidence.
        """
        explanation: list[str] = []

        # ── 1. Cycle alignment ────────────────────────────────────────────
        cycle_raw = _cycle_score(cycle_data)
        explanation.append(f"Cycle alignment: {cycle_raw:.0%}")

        # ── 2. GEX signal ─────────────────────────────────────────────────
        gex_raw, gex_ctx, gex_notes = _compute_gex_signal(
            gex_profile, change_pct, direction
        )
        explanation.extend(gex_notes)

        # ── 3. Momentum strength ──────────────────────────────────────────
        mom_raw = _momentum_score(change_pct, vol_ratio)
        if mom_raw > 0.7:
            explanation.append(f"Strong momentum: {change_pct:+.2f}% on {vol_ratio:.1f}× volume")
        elif mom_raw > 0.4:
            explanation.append(f"Moderate momentum: {change_pct:+.2f}%")
        else:
            explanation.append(f"Weak momentum: {change_pct:+.2f}%")

        # ── 4. Volatility state ───────────────────────────────────────────
        vol_raw = _volatility_score(vix, atr_ratio)
        if vix is not None:
            explanation.append(f"VIX {vix:.1f} → volatility state score {vol_raw:.0%}")

        # ── 5. Structure quality ──────────────────────────────────────────
        struct_raw = _structure_score(pattern_data)

        # ── 6. Catalyst ───────────────────────────────────────────────────
        cat_raw = float((catalyst_data or {}).get("catalyst_score", 0.05))
        if cat_raw > 0.5:
            cat_type = (catalyst_data or {}).get("catalyst_type", "")
            explanation.append(f"Catalyst detected: {cat_type} (score {cat_raw:.0%})")

        # ── Weighted composite ────────────────────────────────────────────
        factors = {
            "cycle_alignment":   cycle_raw,
            "gex_signal":        gex_raw,
            "momentum_strength": mom_raw,
            "volatility_state":  vol_raw,
            "structure_quality": struct_raw,
            "catalyst_score":    cat_raw,
        }
        score = float(sum(
            self.weights[k] * factors[k] for k in self.weights
        ))
        score = float(np.clip(score, 0.0, 1.0))

        # ── Signal type ───────────────────────────────────────────────────
        gex_scenario = gex_ctx.get("scenario", "")
        if gex_scenario in ("breakdown",) or (direction == "short" and score > 0.5):
            signal_type = "short"
        elif score < 0.35:
            signal_type = "neutral"
        else:
            signal_type = direction  # follow scanner direction

        # ── Confidence ────────────────────────────────────────────────────
        if score >= _CONF_HIGH:
            confidence = "high"
        elif score >= _CONF_MEDIUM:
            confidence = "medium"
        elif score >= _CONF_LOW:
            confidence = "low"
        else:
            confidence = "discard"

        return SignalScore(
            symbol      = symbol.upper(),
            score       = score,
            signal_type = signal_type,
            confidence  = confidence,
            factors     = factors,
            gex_context = gex_ctx,
            explanation = explanation,
        )

    def score_batch(self, candidates: list[dict]) -> list[SignalScore]:
        """
        Score multiple candidates. Each dict must have 'symbol' key.
        Other keys map to score() parameters.
        Returns sorted by score descending, discards removed.
        """
        results: list[SignalScore] = []
        for c in candidates:
            sym = c.get("symbol", "")
            if not sym:
                continue
            sig = self.score(
                symbol       = sym,
                price        = c.get("price", 0),
                change_pct   = c.get("change_percent", 0),
                vol_ratio    = c.get("volume_ratio", 1),
                direction    = c.get("direction", "long"),
                gex_profile  = c.get("gex_profile"),
                vix          = c.get("vix"),
                cycle_data   = c.get("cycle_data"),
                pattern_data = c.get("pattern_data"),
                catalyst_data = c.get("catalyst_data"),
            )
            if sig.confidence != "discard":
                results.append(sig)
        results.sort(key=lambda x: x.score, reverse=True)
        return results


# ── Example scenarios ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = ScoringEngine()

    print("=== SCENARIO 1: GAMMA SQUEEZE ===")
    s1 = engine.score(
        "SPY",
        price=505,
        change_pct=2.5,
        vol_ratio=3.2,
        direction="long",
        gex_profile={
            "gamma_regime": "NEGATIVE",
            "zero_gamma": 498,
            "call_wall": 510,
            "put_wall": 490,
            "spot": 505,
            "gex_flip_dist": 1.4,
            "dealer_bias": 1,
        },
        vix=22.0,
    )
    print(f"Score: {s1.score:.2f} | Type: {s1.signal_type} | Confidence: {s1.confidence}")
    for line in s1.explanation:
        print(f"  {line}")

    print("\n=== SCENARIO 2: MEAN REVERSION ===")
    s2 = engine.score(
        "SPY",
        price=500,
        change_pct=0.8,
        vol_ratio=1.1,
        direction="short",
        gex_profile={
            "gamma_regime": "POSITIVE",
            "zero_gamma": 485,
            "call_wall": 500,
            "put_wall": 488,
            "spot": 500,
            "gex_flip_dist": 3.0,
            "dealer_bias": -1,
        },
        vix=14.0,
    )
    print(f"Score: {s2.score:.2f} | Type: {s2.signal_type} | Confidence: {s2.confidence}")

    print("\n=== SCENARIO 3: BREAKDOWN ===")
    s3 = engine.score(
        "QQQ",
        price=445,
        change_pct=-3.2,
        vol_ratio=4.5,
        direction="short",
        gex_profile={
            "gamma_regime": "NEGATIVE",
            "zero_gamma": 455,
            "call_wall": 460,
            "put_wall": 448,
            "spot": 445,
            "gex_flip_dist": 2.2,
            "dealer_bias": -1,
        },
        vix=28.0,
    )
    print(f"Score: {s3.score:.2f} | Type: {s3.signal_type} | Confidence: {s3.confidence}")
