"""Liquidity Zone Builder — aggregates all zone types into a unified map.

Zone type catalogue
-------------------
equal_high_cluster   — repeated equal highs (resting sell stops above)
equal_low_cluster    — repeated equal lows  (resting buy stops below)
previous_day_high    — prior session's high (key intraday reference)
previous_day_low     — prior session's low  (key intraday reference)
swing_high_liquidity — ATR-adaptive swing highs from structure_engine
swing_low_liquidity  — ATR-adaptive swing lows  from structure_engine
range_high           — top of the most recent N-bar consolidation range
range_low            — bottom of the most recent N-bar consolidation range

Each zone dict has a stable schema so callers can iterate without branching::

    {
        "type":     str,    # one of the 8 zone type strings above
        "price":    float,  # single representative price level
        "strength": float,  # 0–1 liquidity strength score
        "side":     str,    # "high" | "low"
        "meta":     dict,   # type-specific extra fields (may be empty)
    }
"""

from __future__ import annotations

import pandas as pd


# ── Zone type identifiers ──────────────────────────────────────────────────────

ZONE_TYPES: dict[str, str] = {
    "equal_high_cluster":   "equal_high_cluster",
    "equal_low_cluster":    "equal_low_cluster",
    "previous_day_high":    "previous_day_high",
    "previous_day_low":     "previous_day_low",
    "swing_high_liquidity": "swing_high_liquidity",
    "swing_low_liquidity":  "swing_low_liquidity",
    "range_high":           "range_high",
    "range_low":            "range_low",
}


# ── Public builder ─────────────────────────────────────────────────────────────

def build_zones(
    df: pd.DataFrame,
    swing_highs: list[dict],
    swing_lows: list[dict],
    equal_high_clusters: list[dict],
    equal_low_clusters: list[dict],
    range_lookback: int = 20,
) -> list[dict]:
    """Aggregate all liquidity zone types into a single sorted zone list.

    Args:
        df:                   OHLCV DataFrame.  Index may be datetime or int.
        swing_highs:          Swing high dicts from structure_engine.swing_detector.
                              Each must contain ``"price"`` and ``"index"`` keys.
        swing_lows:           Swing low dicts with the same schema.
        equal_high_clusters:  Output of :func:`equal_levels.detect_equal_highs`.
        equal_low_clusters:   Output of :func:`equal_levels.detect_equal_lows`.
        range_lookback:       Bar window used to compute the consolidation range.

    Returns:
        List of zone dicts sorted by ``price`` descending (highest zone first),
        each conforming to the module-level schema.
    """
    zones: list[dict] = []

    # ── 1. Equal high / low clusters ──────────────────────────────────────────
    for cluster in equal_high_clusters:
        zones.append(_make_zone(
            zone_type="equal_high_cluster",
            price=cluster["price"],
            strength=cluster.get("strength", 0.5),
            side="high",
            meta={"touches": cluster.get("touches", 0)},
        ))

    for cluster in equal_low_clusters:
        zones.append(_make_zone(
            zone_type="equal_low_cluster",
            price=cluster["price"],
            strength=cluster.get("strength", 0.5),
            side="low",
            meta={"touches": cluster.get("touches", 0)},
        ))

    # ── 2. Previous day high / low ────────────────────────────────────────────
    pdh, pdl = _get_previous_day_levels(df)
    if pdh is not None:
        zones.append(_make_zone(
            zone_type="previous_day_high",
            price=pdh,
            strength=0.75,   # institutional reference: fixed high strength
            side="high",
        ))
    if pdl is not None:
        zones.append(_make_zone(
            zone_type="previous_day_low",
            price=pdl,
            strength=0.75,
            side="low",
        ))

    # ── 3. Swing high / low liquidity ─────────────────────────────────────────
    n = len(df)
    for sh in swing_highs:
        bar_idx = sh.get("index", -1)
        if bar_idx < 0:
            continue
        # Recency: more recent bars score higher (0→oldest, 1→newest)
        recency = bar_idx / max(n - 1, 1)
        strength = round(0.5 + 0.5 * recency, 4)   # range 0.5–1.0
        zones.append(_make_zone(
            zone_type="swing_high_liquidity",
            price=sh["price"],
            strength=strength,
            side="high",
            meta={"bar_index": bar_idx},
        ))

    for sl in swing_lows:
        bar_idx = sl.get("index", -1)
        if bar_idx < 0:
            continue
        recency = bar_idx / max(n - 1, 1)
        strength = round(0.5 + 0.5 * recency, 4)
        zones.append(_make_zone(
            zone_type="swing_low_liquidity",
            price=sl["price"],
            strength=strength,
            side="low",
            meta={"bar_index": bar_idx},
        ))

    # ── 4. Range high / low ───────────────────────────────────────────────────
    rh, rl = _get_range_levels(df, lookback=range_lookback)
    if rh is not None:
        zones.append(_make_zone(
            zone_type="range_high",
            price=rh,
            strength=0.6,
            side="high",
            meta={"lookback": range_lookback},
        ))
    if rl is not None:
        zones.append(_make_zone(
            zone_type="range_low",
            price=rl,
            strength=0.6,
            side="low",
            meta={"lookback": range_lookback},
        ))

    # Sort highest price first so callers can iterate top-down
    zones.sort(key=lambda z: z["price"], reverse=True)
    return zones


# ── Internal helpers ───────────────────────────────────────────────────────────

def _make_zone(
    zone_type: str,
    price: float,
    strength: float,
    side: str,
    meta: dict | None = None,
) -> dict:
    """Construct a canonical zone dict."""
    return {
        "type":     zone_type,
        "price":    round(float(price), 4),
        "strength": round(float(strength), 4),
        "side":     side,
        "meta":     meta or {},
    }


def _get_previous_day_levels(df: pd.DataFrame) -> tuple[float | None, float | None]:
    """Extract the prior session's high and low from the DataFrame.

    Works for both datetime-indexed and integer-indexed DataFrames.
    For intraday data, groups by calendar date and returns the previous
    day's extremes.  Falls back to a rolling 390-bar proxy when dates
    are unavailable.
    """
    if df.empty:
        return None, None

    # Try datetime index → group by date
    if hasattr(df.index, "date"):
        try:
            dates = df.index.date
            unique_dates = sorted(set(dates))
            if len(unique_dates) < 2:
                return None, None
            prev_date = unique_dates[-2]
            mask = (dates == prev_date)
            prev_session = df[mask]
            if prev_session.empty:
                return None, None
            pdh = round(float(prev_session["high"].max()), 4)
            pdl = round(float(prev_session["low"].min()),  4)
            return pdh, pdl
        except Exception:
            pass

    # Fallback for integer-indexed data: use previous 390-bar proxy (~1 session)
    session_bars = 390
    if len(df) < session_bars + 1:
        return None, None

    prev_window = df.iloc[-(session_bars * 2):-session_bars]
    pdh = round(float(prev_window["high"].max()), 4)
    pdl = round(float(prev_window["low"].min()),  4)
    return pdh, pdl


def _get_range_levels(
    df: pd.DataFrame,
    lookback: int = 20,
) -> tuple[float | None, float | None]:
    """Compute the high and low of the most recent ``lookback`` bars."""
    if len(df) < lookback:
        return None, None

    window = df.tail(lookback)
    rh = round(float(window["high"].max()), 4)
    rl = round(float(window["low"].min()),  4)
    return rh, rl
