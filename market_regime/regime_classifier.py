"""Regime Classifier — final decision layer for market regime labelling.

Consumes pre-computed signals from trend_detector, range_detector, and
volatility_state and applies a priority-ordered decision tree to produce
a single regime label with a confidence score.

Priority order
--------------
1. VOLATILITY_EXPANSION — ATR spike overrides structural bias.
2. TREND_UP             — confirmed bullish swing structure + slope.
3. TREND_DOWN           — confirmed bearish swing structure + slope.
4. ACCUMULATION         — range with bullish-biased liquidity sweeps.
5. DISTRIBUTION         — range with bearish-biased liquidity sweeps.
6. RANGE                — default sideways when no stronger signal fires.

Each regime also carries a ``volatility_state`` tag (EXPANDING / NORMAL /
CONTRACTING) for downstream consumers.

Regime labels
-------------
    TREND_UP             Strong bullish trending structure.
    TREND_DOWN           Strong bearish trending structure.
    RANGE                Sideways consolidation, no clear bias.
    VOLATILITY_EXPANSION ATR significantly above norm (breakout risk).
    ACCUMULATION         Compression with buy-side sweep activity.
    DISTRIBUTION         Compression with sell-side sweep activity.
"""

from __future__ import annotations

# ── Regime label constants ─────────────────────────────────────────────────────

REGIME_TREND_UP    = "TREND_UP"
REGIME_TREND_DOWN  = "TREND_DOWN"
REGIME_RANGE       = "RANGE"
REGIME_VOL_EXP     = "VOLATILITY_EXPANSION"
REGIME_ACCUM       = "ACCUMULATION"
REGIME_DISTRIB     = "DISTRIBUTION"

ALL_REGIMES = [
    REGIME_TREND_UP,
    REGIME_TREND_DOWN,
    REGIME_RANGE,
    REGIME_VOL_EXP,
    REGIME_ACCUM,
    REGIME_DISTRIB,
]


def classify_regime(
    trend: dict,
    range_info: dict,
    vol_state: dict,
    swept_zones: list[dict] | None = None,
) -> dict:
    """Apply the priority decision tree and return a labelled regime dict.

    Args:
        trend:       Output of :func:`trend_detector.detect_trend`.
        range_info:  Output of :func:`range_detector.detect_range`.
        vol_state:   Output of :func:`volatility_state.classify_volatility_state`.
        swept_zones: Optional list of recently swept zone event dicts from
                     liquidity_map.sweep_detection.  Used to distinguish
                     ACCUMULATION from DISTRIBUTION within a range.

    Returns:
        dict::

            {
                "regime":           str,    # one of ALL_REGIMES
                "confidence":       float,  # 0–1
                "trend":            str,    # "UP" | "DOWN" | "NEUTRAL"
                "trend_strength":   float,
                "volatility_state": str,    # "EXPANDING" | "NORMAL" | "CONTRACTING"
                "is_range":         bool,
                "range_high":       float,
                "range_low":        float,
            }
    """
    is_expansion  = vol_state.get("is_expansion",     False)
    is_contraction= vol_state.get("is_contraction",   False)
    vstate        = vol_state.get("volatility_state", "NORMAL")
    atr_ratio     = vol_state.get("atr_ratio", 1.0)

    trend_label   = trend.get("trend",          "NEUTRAL")
    trend_str     = trend.get("trend_strength", 0.0)

    is_range      = range_info.get("is_range",         False)
    range_conf    = range_info.get("range_confidence", 0.0)
    range_high    = range_info.get("range_high",       0.0)
    range_low     = range_info.get("range_low",        0.0)

    # ── Priority 1: Volatility Expansion ──────────────────────────────────────
    if is_expansion:
        # Confidence scales with how extreme the ATR spike is
        exp_conf = round(min((atr_ratio - 1.5) / 1.5 + 0.5, 1.0), 4)
        return _make_result(
            regime=REGIME_VOL_EXP,
            confidence=max(exp_conf, 0.5),
            trend=trend_label, trend_strength=trend_str,
            volatility_state=vstate,
            is_range=is_range,
            range_high=range_high, range_low=range_low,
        )

    # ── Priority 2 & 3: Trend ─────────────────────────────────────────────────
    if trend_label == "UP" and trend_str >= 0.4:
        return _make_result(
            regime=REGIME_TREND_UP,
            confidence=round(trend_str, 4),
            trend=trend_label, trend_strength=trend_str,
            volatility_state=vstate,
            is_range=False,
            range_high=range_high, range_low=range_low,
        )

    if trend_label == "DOWN" and trend_str >= 0.4:
        return _make_result(
            regime=REGIME_TREND_DOWN,
            confidence=round(trend_str, 4),
            trend=trend_label, trend_strength=trend_str,
            volatility_state=vstate,
            is_range=False,
            range_high=range_high, range_low=range_low,
        )

    # ── Priority 4 & 5: Accumulation / Distribution (within a range) ──────────
    if is_range and range_conf >= 0.3:
        bias = _sweep_bias(swept_zones or [])

        if bias == "bullish" or (bias == "neutral" and is_contraction):
            return _make_result(
                regime=REGIME_ACCUM,
                confidence=round((range_conf + 0.5) / 2, 4),
                trend=trend_label, trend_strength=trend_str,
                volatility_state=vstate,
                is_range=True,
                range_high=range_high, range_low=range_low,
            )

        if bias == "bearish":
            return _make_result(
                regime=REGIME_DISTRIB,
                confidence=round((range_conf + 0.5) / 2, 4),
                trend=trend_label, trend_strength=trend_str,
                volatility_state=vstate,
                is_range=True,
                range_high=range_high, range_low=range_low,
            )

        return _make_result(
            regime=REGIME_RANGE,
            confidence=round(range_conf, 4),
            trend=trend_label, trend_strength=trend_str,
            volatility_state=vstate,
            is_range=True,
            range_high=range_high, range_low=range_low,
        )

    # ── Priority 6: Fallback Range / weak trend ───────────────────────────────
    if is_range:
        return _make_result(
            regime=REGIME_RANGE,
            confidence=round(range_conf, 4),
            trend=trend_label, trend_strength=trend_str,
            volatility_state=vstate,
            is_range=True,
            range_high=range_high, range_low=range_low,
        )

    # Weak trend — pick direction but low confidence
    if trend_label == "UP":
        return _make_result(
            regime=REGIME_TREND_UP,
            confidence=round(trend_str * 0.6, 4),
            trend=trend_label, trend_strength=trend_str,
            volatility_state=vstate,
            is_range=False,
            range_high=range_high, range_low=range_low,
        )
    if trend_label == "DOWN":
        return _make_result(
            regime=REGIME_TREND_DOWN,
            confidence=round(trend_str * 0.6, 4),
            trend=trend_label, trend_strength=trend_str,
            volatility_state=vstate,
            is_range=False,
            range_high=range_high, range_low=range_low,
        )

    return _make_result(
        regime=REGIME_RANGE,
        confidence=0.3,
        trend=trend_label, trend_strength=trend_str,
        volatility_state=vstate,
        is_range=is_range,
        range_high=range_high, range_low=range_low,
    )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _sweep_bias(swept_zones: list[dict]) -> str:
    """Determine net sweep direction from recent zone events.

    Returns:
        ``"bullish"``  — more bottom sweeps (buying pressure).
        ``"bearish"``  — more top sweeps (selling pressure).
        ``"neutral"``  — balanced or no sweeps.
    """
    bottom = sum(1 for e in swept_zones if e.get("sweep_type") == "bottom_sweep")
    top    = sum(1 for e in swept_zones if e.get("sweep_type") == "top_sweep")

    if bottom > top:
        return "bullish"
    if top > bottom:
        return "bearish"
    return "neutral"


def _make_result(
    regime: str,
    confidence: float,
    trend: str,
    trend_strength: float,
    volatility_state: str,
    is_range: bool,
    range_high: float,
    range_low: float,
) -> dict:
    return {
        "regime":           regime,
        "confidence":       round(max(0.0, min(confidence, 1.0)), 4),
        "trend":            trend,
        "trend_strength":   round(trend_strength, 4),
        "volatility_state": volatility_state,
        "is_range":         is_range,
        "range_high":       round(range_high, 4),
        "range_low":        round(range_low,  4),
    }
