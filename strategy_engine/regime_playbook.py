"""
RoboAlgo — Regime Playbook
Single source of truth for all regime-dependent trading behavior.

The playbook maps each market regime to a complete trading ruleset.
Every downstream engine (signal, portfolio, execution) must query this
module before making decisions — never hardcode regime behavior elsewhere.

Market cycle:
  Compression → Breakout → Expansion → Trend → Exhaustion → Compression

Usage
-----
    from strategy_engine.regime_playbook import get_active_strategy, PLAYBOOK

    # Simple lookup — returns the canonical three-field dict
    result = get_active_strategy("EXPANSION")
    # → {"strategy": "breakout", "position_multiplier": 1.25, "risk_mode": "aggressive"}

    # Full rule with all guard rails
    rule = PLAYBOOK["EXPANSION"]
    rule.risk_per_trade        # → 0.02
    rule.max_positions         # → 5
    rule.volume_requirement    # → 1.5  (volume must be ≥ 1.5× average)
    rule.quality_score_min     # → 62   (minimum SetupQualityScore)
    rule.entry_description     # → human-readable entry logic
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlaybookRule:
    """
    Complete per-regime trading ruleset.
    All fields are immutable — modify PLAYBOOK dict to change behaviour.
    """
    regime: str

    # ── Identity ─────────────────────────────────────────────────────────────
    strategy_type: str      # human-readable label, e.g. "Breakout Momentum"
    strategy_key:  str      # machine-readable key: breakout | pullback | watchlist | defensive

    # ── Risk parameters ───────────────────────────────────────────────────────
    position_multiplier: float   # applied to base position size (0.0 = no trades)
    risk_per_trade:      float   # fraction of account equity risked per trade
    max_positions:       int     # maximum open positions allowed simultaneously
    risk_mode:           str     # "aggressive" | "normal" | "reduced" | "defensive"

    # ── Entry guards ──────────────────────────────────────────────────────────
    confluence_min:    float   # minimum ConfluenceScore to allow entry
    quality_score_min: float   # minimum SetupQualityScore to allow entry
    volume_requirement: float  # minimum volume_ratio multiple (e.g. 1.5×)

    # ── Setup types allowed ───────────────────────────────────────────────────
    allowed_setups: list[str]  # empty list = no entries allowed

    # ── Human-readable rules ─────────────────────────────────────────────────
    entry_description:  str
    stop_description:   str
    target_description: str
    size_description:   str

    # ── UI ───────────────────────────────────────────────────────────────────
    color: str   # primary regime colour (hex)


# ── The Playbook ──────────────────────────────────────────────────────────────

PLAYBOOK: dict[str, PlaybookRule] = {

    "COMPRESSION": PlaybookRule(
        regime              = "COMPRESSION",
        strategy_type       = "Watchlist Builder",
        strategy_key        = "watchlist",
        # Risk
        position_multiplier = 0.0,
        risk_per_trade      = 0.01,    # 1% if a hedge/range trade is taken
        max_positions       = 3,
        risk_mode           = "reduced",
        # Guards
        confluence_min      = 999.0,   # effectively blocked for directional trades
        quality_score_min   = 0.0,
        volume_requirement  = 0.0,
        # Setups
        allowed_setups      = [],      # no directional entries
        # Descriptions
        entry_description   = (
            "Compression detected — energy building. "
            "Build watchlist, identify setups, activate hedged or range strategies. "
            "No directional entries until expansion is confirmed."
        ),
        stop_description    = "Tighter stops required — narrow range environment.",
        target_description  = "Range targets only. Do not chase breakouts inside compression.",
        size_description    = "NO DIRECTIONAL TRADES — watchlist phase. "
                              "Hedge or range trades use reduced size (1% risk).",
        color               = "#60a5fa",
    ),

    "EXPANSION": PlaybookRule(
        regime              = "EXPANSION",
        strategy_type       = "Breakout Momentum",
        strategy_key        = "breakout",
        # Risk
        position_multiplier = 1.25,
        risk_per_trade      = 0.02,   # 2% — full risk
        max_positions       = 5,
        risk_mode           = "aggressive",
        # Guards
        confluence_min      = 60.0,
        quality_score_min   = 62.0,   # Grade B or better
        volume_requirement  = 1.5,    # ≥ 1.5× average volume REQUIRED
        # Setups
        allowed_setups      = [
            "compression_breakout",
            "liquidity_sweep",
            "wyckoff_spring",
            "pattern_reversal",
            "trend_pullback",
        ],
        # Descriptions
        entry_description   = (
            "Breakout confirmed — volume participation and momentum accelerating. "
            "Enter aggressively on breakout close. Allow wider stops. "
            "Volume must be ≥ 1.5× average. Scale into winners."
        ),
        stop_description    = "Wider stops allowed — below breakout level or 1.5× ATR.",
        target_description  = "Extended targets: Tier 1 at +1 ATR, Tier 2 at +2.5 ATR, "
                              "Tier 3 hold as house money.",
        size_description    = "Increased sizing (1.25×) — expansion regime. "
                              "2% risk per trade. Max 5 positions.",
        color               = "#f97316",
    ),

    "TREND": PlaybookRule(
        regime              = "TREND",
        strategy_type       = "Trend Pullback",
        strategy_key        = "pullback",
        # Risk
        position_multiplier = 1.0,
        risk_per_trade      = 0.015,  # 1.5%
        max_positions       = 5,
        risk_mode           = "normal",
        # Guards
        confluence_min      = 65.0,
        quality_score_min   = 55.0,   # Grade C or better
        volume_requirement  = 1.0,    # standard volume
        # Setups
        allowed_setups      = [
            "trend_pullback",
            "compression_breakout",
            "liquidity_sweep",
        ],
        # Descriptions
        entry_description   = (
            "Stable directional trend. Enter on pullbacks to MA20 or MA50. "
            "Require trend alignment across timeframes. "
            "Avoid entries against the primary trend direction."
        ),
        stop_description    = "Standard stops — below recent swing low or MA50.",
        target_description  = "Standard targets: Tier 1 at +1.5 ATR, Tier 2 at trend extension.",
        size_description    = "Standard sizing (1.0×) — trend regime. 1.5% risk per trade.",
        color               = "#22c55e",
    ),

    "CHAOS": PlaybookRule(
        regime              = "CHAOS",
        strategy_type       = "Defensive Mode",
        strategy_key        = "defensive",
        # Risk
        position_multiplier = 0.5,   # -50% size
        risk_per_trade      = 0.005, # 0.5%
        max_positions       = 2,
        risk_mode           = "defensive",
        # Guards — very high bar
        confluence_min      = 85.0,
        quality_score_min   = 75.0,   # Grade A only
        volume_requirement  = 0.0,
        # Setups
        allowed_setups      = [
            "compression_breakout",
            "pattern_reversal",
        ],
        # Descriptions
        entry_description   = (
            "Chaos regime — volatility spike, wide ranges, high correlation. "
            "Defensive mode only. Only highest-conviction setups (≥85 confluence, "
            "Grade A quality score). Reduce all exposure immediately."
        ),
        stop_description    = "Tight stops mandatory — capital preservation is priority.",
        target_description  = "Conservative targets only — accept smaller wins.",
        size_description    = "Reduced sizing (0.5×) — chaos regime. "
                              "0.5% risk per trade. Max 2 positions.",
        color               = "#ef4444",
    ),
}


# ── Public API ────────────────────────────────────────────────────────────────

def get_active_strategy(regime: str) -> dict:
    """
    Return the canonical three-field strategy descriptor for *regime*.

    Args:
        regime: One of "COMPRESSION", "EXPANSION", "TREND", "CHAOS"

    Returns:
        {
            "strategy":            str,    # "breakout" | "pullback" | "watchlist" | "defensive"
            "position_multiplier": float,  # applied to base position size
            "risk_mode":           str,    # "aggressive" | "normal" | "reduced" | "defensive"
        }

    Example:
        >>> get_active_strategy("EXPANSION")
        {"strategy": "breakout", "position_multiplier": 1.25, "risk_mode": "aggressive"}
    """
    rule = PLAYBOOK.get(regime.upper(), PLAYBOOK["COMPRESSION"])
    return {
        "strategy":            rule.strategy_key,
        "position_multiplier": rule.position_multiplier,
        "risk_mode":           rule.risk_mode,
    }


def get_rule(regime: str) -> PlaybookRule:
    """Return the full PlaybookRule for *regime* (defaults to COMPRESSION)."""
    return PLAYBOOK.get(regime.upper(), PLAYBOOK["COMPRESSION"])


def signal_allowed(
    regime:          str,
    setup_type:      str,
    confluence:      float,
    quality_score:   float,
    volume_ratio:    float = 1.0,
) -> tuple[bool, str]:
    """
    Gate function — determine whether a signal is allowed under the current regime.

    Returns:
        (allowed: bool, reason: str)

    Example:
        allowed, reason = signal_allowed("COMPRESSION", "compression_breakout", 72, 65)
        # → (False, "COMPRESSION state — no directional entries allowed")
    """
    rule = get_rule(regime)

    # ── 1. No entries in COMPRESSION ──────────────────────────────────────────
    if rule.position_multiplier == 0.0:
        return False, f"{regime} state — no directional entries allowed. Building watchlist."

    # ── 2. Setup type must be permitted ──────────────────────────────────────
    if rule.allowed_setups and setup_type not in rule.allowed_setups:
        allowed_str = ", ".join(rule.allowed_setups)
        return False, (
            f"Setup '{setup_type}' not permitted in {regime} regime. "
            f"Allowed: {allowed_str}"
        )

    # ── 3. Confluence threshold ───────────────────────────────────────────────
    if confluence < rule.confluence_min:
        return False, (
            f"Confluence {confluence:.1f} < {rule.confluence_min:.0f} required "
            f"for {regime} regime ({rule.strategy_type})"
        )

    # ── 4. Setup quality score ────────────────────────────────────────────────
    if quality_score < rule.quality_score_min:
        return False, (
            f"SetupQualityScore {quality_score:.0f} < {rule.quality_score_min:.0f} "
            f"required for {regime} regime"
        )

    # ── 5. Volume requirement (EXPANSION only: ≥ 1.5×) ───────────────────────
    if rule.volume_requirement > 0 and volume_ratio < rule.volume_requirement:
        return False, (
            f"Volume {volume_ratio:.2f}× < {rule.volume_requirement:.1f}× required "
            f"for {regime} regime breakout confirmation"
        )

    return True, f"Signal approved — {rule.strategy_type} strategy active"


def build_decision_trace(
    symbol:        str,
    regime:        str,
    setup_type:    str,
    confluence:    float,
    quality_score: float,
    volume_ratio:  float,
    entry_price:   float | None = None,
    stop_price:    float | None = None,
    target_price:  float | None = None,
    extra_factors: dict | None  = None,
) -> str:
    """
    Build a human-readable decision trace string for a signal.
    Returned string is stored in Signal.decision_trace.

    Format matches the spec:
        Symbol: SOXL
        Regime: EXPANSION  |  Strategy Mode: Breakout Momentum
        Compression Score: 82  |  Volume Participation: 1.8×
        Setup Quality Score: 86
        Signal Reason: …
    """
    rule = get_rule(regime)
    allowed, reason = signal_allowed(regime, setup_type, confluence, quality_score, volume_ratio)

    lines = [
        f"Symbol: {symbol}",
        f"Regime: {regime}  |  Strategy Mode: {rule.strategy_type}",
        f"Confluence Score: {confluence:.0f}  |  Volume Participation: {volume_ratio:.2f}×",
        f"Setup Quality Score: {quality_score:.0f}",
        f"Setup Type: {setup_type}",
    ]

    if entry_price is not None:
        lines.append(f"Entry: ${entry_price:.2f}")
    if stop_price is not None:
        lines.append(f"Stop: ${stop_price:.2f}")
    if target_price is not None:
        lines.append(f"Target: ${target_price:.2f}")

    if extra_factors:
        for k, v in extra_factors.items():
            lines.append(f"{k}: {v}")

    lines.append(f"Signal Reason: {reason}")
    lines.append(f"Entry Logic: {rule.entry_description}")

    return "\n".join(lines)


def get_reliability_gate(setup_type: str) -> dict:
    """
    Gate function — retrieve the current signal reliability status for *setup_type*
    and return the position size multiplier that should be applied.

    This is the single integration point for the signal engine and portfolio engine.
    Call this AFTER signal_allowed() passes.

    Returns:
        {
            "setup_type":          str,
            "reliability_score":   float | None,   # None if no history
            "status":              str,             # "healthy" | "warning" | "disabled" | "no_data"
            "position_multiplier": float,           # 1.0 | 0.5 | 0.0
            "allowed":             bool,            # False when status == "disabled"
            "reason":              str,
        }

    Rules:
        reliability_score ≥ 70  → normal trading  (multiplier 1.0)
        reliability_score 50–69 → reduce size 50% (multiplier 0.5)
        reliability_score < 50  → disable strategy (multiplier 0.0, allowed=False)
        no history              → allow at full size (don't penalise new setups)

    Example:
        gate = get_reliability_gate("compression_breakout")
        if not gate["allowed"]:
            return  # strategy suspended
        final_size = base_size * gate["position_multiplier"]
    """
    try:
        from analytics_engine.signal_reliability import SignalReliabilityEngine
        engine = SignalReliabilityEngine()
        result = engine.get_reliability_status(setup_type)
    except Exception as e:
        logger.warning("get_reliability_gate: engine call failed for %s: %s", setup_type, e)
        return {
            "setup_type":          setup_type,
            "reliability_score":   None,
            "status":              "no_data",
            "position_multiplier": 1.0,
            "allowed":             True,
            "reason":              f"Reliability engine unavailable: {e}",
        }

    score  = result.get("reliability_score")
    status = result.get("status", "no_data")
    mult   = result.get("position_multiplier", 1.0)

    if status == "disabled":
        reason  = (
            f"Strategy '{setup_type}' suspended — "
            f"reliability score {score:.0f} < 50. "
            "Edge has deteriorated. Waiting for recovery."
        )
        allowed = False
    elif status == "warning":
        reason  = (
            f"Strategy '{setup_type}' in warning — "
            f"reliability score {score:.0f}. "
            "Position size reduced to 50%."
        )
        allowed = True
    elif status == "healthy":
        reason  = (
            f"Strategy '{setup_type}' healthy — "
            f"reliability score {score:.0f}. Full position size."
        )
        allowed = True
    else:
        reason  = f"No reliability history for '{setup_type}'. Proceeding at full size."
        allowed = True

    return {
        "setup_type":          setup_type,
        "reliability_score":   score,
        "status":              status,
        "position_multiplier": mult,
        "allowed":             allowed,
        "reason":              reason,
    }


def get_playbook_summary() -> list[dict]:
    """Return a JSON-serializable summary of all playbook rules (for API)."""
    return [
        {
            "regime":              rule.regime,
            "strategy_type":       rule.strategy_type,
            "strategy_key":        rule.strategy_key,
            "position_multiplier": rule.position_multiplier,
            "risk_per_trade":      rule.risk_per_trade,
            "max_positions":       rule.max_positions,
            "risk_mode":           rule.risk_mode,
            "confluence_min":      rule.confluence_min,
            "quality_score_min":   rule.quality_score_min,
            "volume_requirement":  rule.volume_requirement,
            "allowed_setups":      rule.allowed_setups,
            "entry_description":   rule.entry_description,
            "stop_description":    rule.stop_description,
            "size_description":    rule.size_description,
            "color":               rule.color,
        }
        for rule in PLAYBOOK.values()
    ]
