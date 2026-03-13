"""Measured Move (AB=CD) pattern detector.

Exposes:
    detect(symbol: str, df: pd.DataFrame) -> list[dict]

Each dict follows the RoboAlgo pattern-engine service contract:
    pattern_name, pattern_category, status, direction, confidence,
    breakout_level, invalidation_level, target, points

Patterns detected
-----------------
- Bullish Measured Move  (A swing-high → B swing-low → C recovery → D projected)
- Bearish Measured Move  (A swing-low  → B swing-high → C pullback → D projected)
- Bullish Measured Move Extension  (D → E = D + AB leg)
- Bearish Measured Move Extension  (D → E = D - AB leg)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_CATEGORY = "measured_move"

# Retracement window accepted for BC leg (38-70 % of AB distance)
_RETRACE_MIN = 0.38
_RETRACE_MAX = 0.70

# Swing detection half-window (bars each side)
_SWING_WINDOW = 5

# How many bars of lookback to scan for setups
_LOOKBACK = 100

# Price proximity buffer to classify status (0.5 %)
_PROXIMITY = 0.005


def _make(
    pattern_name: str,
    status: str,
    direction: str,
    confidence: float,
    breakout_level: float | None = None,
    invalidation_level: float | None = None,
    target: float | None = None,
    points: list | None = None,
) -> dict[str, Any]:
    return {
        "pattern_name": pattern_name,
        "pattern_category": _CATEGORY,
        "status": status,
        "direction": direction,
        "confidence": float(np.clip(round(confidence, 2), 0.0, 100.0)),
        "breakout_level": breakout_level,
        "invalidation_level": invalidation_level,
        "target": target,
        "points": points or [],
    }


def _find_swing_highs(series: pd.Series, window: int = _SWING_WINDOW) -> list[int]:
    """Local maxima with strict inequality within `window` bars on each side."""
    vals = series.values
    n = len(vals)
    indices = []
    for i in range(window, n - window):
        if all(vals[i] > vals[i - j] for j in range(1, window + 1)) and \
           all(vals[i] > vals[i + j] for j in range(1, window + 1)):
            indices.append(i)
    return indices


def _find_swing_lows(series: pd.Series, window: int = _SWING_WINDOW) -> list[int]:
    """Local minima with strict inequality within `window` bars on each side."""
    vals = series.values
    n = len(vals)
    indices = []
    for i in range(window, n - window):
        if all(vals[i] < vals[i - j] for j in range(1, window + 1)) and \
           all(vals[i] < vals[i + j] for j in range(1, window + 1)):
            indices.append(i)
    return indices


def _retrace_confidence(retrace_pct: float, base_low: float = 62.0, base_high: float = 82.0) -> float:
    """Return confidence score highest at 50 % retracement, tapering toward edges."""
    ideal = 0.50
    deviation = abs(retrace_pct - ideal)
    allowed_half = max(ideal - _RETRACE_MIN, _RETRACE_MAX - ideal)
    quality = max(0.0, 1.0 - deviation / allowed_half)
    return float(np.clip(base_low + quality * (base_high - base_low), base_low, base_high))


def _status_bullish(
    current_price: float,
    b_price: float,
    c_price: float,
    d_price: float,
) -> str:
    """Classify status for a bullish measured move."""
    if current_price >= d_price:
        return "COMPLETED"
    if current_price >= c_price * (1 - _PROXIMITY):
        if current_price >= c_price:
            return "BREAKOUT"
        return "READY"
    if current_price >= b_price:
        return "FORMING"
    return "NOT_PRESENT"


def _status_bearish(
    current_price: float,
    b_price: float,
    c_price: float,
    d_price: float,
) -> str:
    """Classify status for a bearish measured move."""
    if current_price <= d_price:
        return "COMPLETED"
    if current_price <= c_price * (1 + _PROXIMITY):
        if current_price <= c_price:
            return "BREAKOUT"
        return "READY"
    if current_price <= b_price:
        return "FORMING"
    return "NOT_PRESENT"


def _detect_bullish_measured_move(
    df: pd.DataFrame,
    offset: int,
    window_high: pd.Series,
    window_low: pd.Series,
    window_close: pd.Series,
) -> list[dict[str, Any]]:
    """Bullish: A (swing high) → B (swing low) → C (partial recovery) → D (target)."""
    results: list[dict[str, Any]] = []

    swing_highs = _find_swing_highs(window_high)
    swing_lows = _find_swing_lows(window_low)

    if len(swing_highs) < 1 or len(swing_lows) < 1:
        return results

    current_price = float(window_close.iloc[-1])

    # Iterate over candidate A (swing high) / B (swing low) pairs
    for a_idx in swing_highs:
        a_price = float(window_high.iloc[a_idx])

        # B must be a swing low that comes AFTER A
        valid_b = [b for b in swing_lows if b > a_idx]
        if not valid_b:
            continue

        for b_idx in valid_b:
            b_price = float(window_low.iloc[b_idx])
            ab_dist = a_price - b_price
            if ab_dist <= 0:
                continue

            # C must be a swing high after B that retraces 38-70 % of AB
            valid_c = [h for h in swing_highs if h > b_idx]
            if not valid_c:
                continue

            for c_idx in valid_c:
                c_price = float(window_high.iloc[c_idx])
                bc_dist = c_price - b_price
                retrace_pct = bc_dist / ab_dist

                if not (_RETRACE_MIN <= retrace_pct <= _RETRACE_MAX):
                    continue

                # Project D = C + AB distance (equal-leg target)
                d_price = round(c_price + ab_dist, 4)

                status = _status_bullish(current_price, b_price, c_price, d_price)
                if status == "NOT_PRESENT":
                    continue

                confidence = _retrace_confidence(retrace_pct)

                results.append(_make(
                    pattern_name="Bullish Measured Move",
                    status=status,
                    direction="bullish",
                    confidence=confidence,
                    breakout_level=round(c_price, 4),
                    invalidation_level=round(b_price * (1 - 0.005), 4),
                    target=d_price,
                    points=[
                        [offset + a_idx, a_price],
                        [offset + b_idx, b_price],
                        [offset + c_idx, c_price],
                    ],
                ))

                # Extension: if price already reached D, project E = D + AB
                if status in ("COMPLETED",) or current_price >= d_price * 0.98:
                    e_price = round(d_price + ab_dist, 4)
                    ext_confidence = _retrace_confidence(retrace_pct, base_low=58.0, base_high=70.0)
                    ext_status = "READY" if current_price >= d_price else "FORMING"
                    results.append(_make(
                        pattern_name="Bullish Measured Move Extension",
                        status=ext_status,
                        direction="bullish",
                        confidence=ext_confidence,
                        breakout_level=round(d_price, 4),
                        invalidation_level=round(c_price, 4),
                        target=e_price,
                        points=[
                            [offset + a_idx, a_price],
                            [offset + b_idx, b_price],
                            [offset + c_idx, c_price],
                        ],
                    ))

    return results


def _detect_bearish_measured_move(
    df: pd.DataFrame,
    offset: int,
    window_high: pd.Series,
    window_low: pd.Series,
    window_close: pd.Series,
) -> list[dict[str, Any]]:
    """Bearish: A (swing low) → B (swing high) → C (partial pullback) → D (target)."""
    results: list[dict[str, Any]] = []

    swing_lows = _find_swing_lows(window_low)
    swing_highs = _find_swing_highs(window_high)

    if len(swing_lows) < 1 or len(swing_highs) < 1:
        return results

    current_price = float(window_close.iloc[-1])

    for a_idx in swing_lows:
        a_price = float(window_low.iloc[a_idx])

        # B must be a swing high after A
        valid_b = [b for b in swing_highs if b > a_idx]
        if not valid_b:
            continue

        for b_idx in valid_b:
            b_price = float(window_high.iloc[b_idx])
            ab_dist = b_price - a_price
            if ab_dist <= 0:
                continue

            # C must be a swing low after B that pulls back 38-70 % of AB
            valid_c = [l for l in swing_lows if l > b_idx]
            if not valid_c:
                continue

            for c_idx in valid_c:
                c_price = float(window_low.iloc[c_idx])
                bc_dist = b_price - c_price
                retrace_pct = bc_dist / ab_dist

                if not (_RETRACE_MIN <= retrace_pct <= _RETRACE_MAX):
                    continue

                # Project D = C - AB distance
                d_price = round(c_price - ab_dist, 4)

                status = _status_bearish(current_price, b_price, c_price, d_price)
                if status == "NOT_PRESENT":
                    continue

                confidence = _retrace_confidence(retrace_pct)

                results.append(_make(
                    pattern_name="Bearish Measured Move",
                    status=status,
                    direction="bearish",
                    confidence=confidence,
                    breakout_level=round(c_price, 4),
                    invalidation_level=round(b_price * (1 + 0.005), 4),
                    target=d_price,
                    points=[
                        [offset + a_idx, a_price],
                        [offset + b_idx, b_price],
                        [offset + c_idx, c_price],
                    ],
                ))

                # Extension: D → E = D - AB
                if status in ("COMPLETED",) or current_price <= d_price * 1.02:
                    e_price = round(d_price - ab_dist, 4)
                    ext_confidence = _retrace_confidence(retrace_pct, base_low=58.0, base_high=70.0)
                    ext_status = "READY" if current_price <= d_price else "FORMING"
                    results.append(_make(
                        pattern_name="Bearish Measured Move Extension",
                        status=ext_status,
                        direction="bearish",
                        confidence=ext_confidence,
                        breakout_level=round(d_price, 4),
                        invalidation_level=round(c_price, 4),
                        target=e_price,
                        points=[
                            [offset + a_idx, a_price],
                            [offset + b_idx, b_price],
                            [offset + c_idx, c_price],
                        ],
                    ))

    return results


def _deduplicate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove near-duplicate patterns keeping the highest-confidence instance.

    Two results are considered duplicates when they share the same pattern_name
    and their breakout_level values are within 0.5 % of each other.
    """
    if not results:
        return results

    kept: list[dict[str, Any]] = []
    for candidate in sorted(results, key=lambda x: x.get("confidence", 0.0), reverse=True):
        name = candidate["pattern_name"]
        bl = candidate.get("breakout_level") or 0.0
        is_dup = False
        for existing in kept:
            if existing["pattern_name"] != name:
                continue
            ebl = existing.get("breakout_level") or 0.0
            if ebl > 0 and abs(bl - ebl) / ebl < 0.005:
                is_dup = True
                break
        if not is_dup:
            kept.append(candidate)
    return kept


def detect(symbol: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect Measured Move (AB=CD) patterns in the given OHLCV DataFrame.

    Parameters
    ----------
    symbol : str
        Ticker symbol (used for logging only).
    df : pd.DataFrame
        Must contain columns: date, open, high, low, close, volume.
        Rows should be sorted oldest-first.

    Returns
    -------
    list[dict]
        Zero or more pattern dicts following the RoboAlgo service contract.
        Returns [] if df has fewer than 40 rows or on any unhandled error.
    """
    try:
        if df is None or len(df) < 40:
            return []

        df = df.sort_values("date").reset_index(drop=True).copy()

        # Restrict scan to last LOOKBACK bars
        lookback = min(_LOOKBACK, len(df))
        window = df.tail(lookback).reset_index(drop=True)
        offset = len(df) - lookback  # global bar index of window[0]

        window_high = window["high"].astype(float)
        window_low = window["low"].astype(float)
        window_close = window["close"].astype(float)

        results: list[dict[str, Any]] = []
        results.extend(_detect_bullish_measured_move(df, offset, window_high, window_low, window_close))
        results.extend(_detect_bearish_measured_move(df, offset, window_high, window_low, window_close))

        # Remove NOT_PRESENT, deduplicate, sort by confidence
        results = [r for r in results if r.get("status") != "NOT_PRESENT"]
        results = _deduplicate(results)
        results.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        return results

    except Exception:
        return []
