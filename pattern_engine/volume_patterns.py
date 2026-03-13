"""Volume-based pattern detector.

Exposes:
    detect(symbol: str, df: pd.DataFrame) -> list[dict]

Each dict follows the RoboAlgo pattern-engine service contract:
    pattern_name, pattern_category, status, direction, confidence,
    breakout_level, invalidation_level, target, points
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_CATEGORY = "volume"


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


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    return float((high - low).tail(period).mean())


def _detect_climactic_volume(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Selling Climax / Buying Climax patterns."""
    results: list[dict[str, Any]] = []
    if len(df) < 25:
        return results

    volume = df["volume"].astype(float)
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)

    avg_vol = volume.rolling(20).mean()

    # Scan last 10 bars
    n = len(df)
    for i in range(max(0, n - 10), n):
        if pd.isna(avg_vol.iloc[i]) or avg_vol.iloc[i] <= 0:
            continue
        vol_ratio = float(volume.iloc[i]) / float(avg_vol.iloc[i])
        if vol_ratio < 2.5:
            continue

        bar_open = float(open_.iloc[i])
        bar_close = float(close.iloc[i])
        if bar_open <= 0:
            continue
        price_change = (bar_close - bar_open) / bar_open

        bars_ago = n - 1 - i
        status = "READY" if bars_ago <= 5 else "FORMING"

        # Confidence scales from 65 at vol_ratio=2.5 to 82 at vol_ratio≈5
        confidence = float(np.clip(65 + (vol_ratio - 2.5) / 2.5 * 17, 65, 82))

        if price_change <= -0.03:
            # Selling Climax → bullish reversal signal
            results.append(_make(
                pattern_name="Selling Climax",
                status=status,
                direction="bullish",
                confidence=confidence,
                breakout_level=round(bar_close * 1.01, 4),
                invalidation_level=round(float(df["low"].astype(float).iloc[i]), 4),
                target=round(bar_close + (float(df["high"].astype(float).iloc[i]) - float(df["low"].astype(float).iloc[i])) * 2.0, 4),
                points=[[i, bar_close]],
            ))
        elif price_change >= 0.03:
            # Buying Climax → bearish distribution signal
            results.append(_make(
                pattern_name="Buying Climax",
                status=status,
                direction="bearish",
                confidence=confidence,
                breakout_level=round(bar_close * 0.99, 4),
                invalidation_level=round(float(df["high"].astype(float).iloc[i]), 4),
                target=round(bar_close - (float(df["high"].astype(float).iloc[i]) - float(df["low"].astype(float).iloc[i])) * 2.0, 4),
                points=[[i, bar_close]],
            ))

    return results


def _detect_volume_dry_up(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Volume Dry Up (VDU / Spring) — coiling before a move."""
    if len(df) < 25:
        return []

    volume = df["volume"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    atr = _compute_atr(df)

    avg_vol = volume.rolling(20).mean()
    n = len(df)

    # Find the most recent run of 3+ consecutive low-volume, narrow-range bars
    # that ended within the last 10 bars
    end_search = n
    start_search = max(0, n - 20)

    best_start: int | None = None
    best_end: int | None = None
    run_start: int | None = None

    for i in range(start_search, end_search):
        avg = float(avg_vol.iloc[i]) if not pd.isna(avg_vol.iloc[i]) else None
        if avg is None or avg <= 0:
            run_start = None
            continue
        is_low_vol = float(volume.iloc[i]) < 0.4 * avg
        bar_range = float(high.iloc[i]) - float(low.iloc[i])
        is_narrow = bar_range < atr if atr > 0 else True

        if is_low_vol and is_narrow:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_len = i - run_start
                if run_len >= 3:
                    best_start = run_start
                    best_end = i - 1
            run_start = None

    # Check if a run is still open at the end of the scan window
    if run_start is not None:
        run_len = end_search - run_start
        if run_len >= 3:
            best_start = run_start
            best_end = end_search - 1

    if best_start is None or best_end is None:
        return []

    # VDU is FORMING if the dry-up period is still ongoing (last bar is still low vol)
    # or just ended (last bar is the first normal-volume bar after the dry-up)
    last_avg = float(avg_vol.iloc[n - 1]) if not pd.isna(avg_vol.iloc[n - 1]) else 0.0
    last_vol = float(volume.iloc[n - 1])
    just_ended = best_end < n - 1 and n - 1 - best_end <= 2
    still_in = best_end == n - 1

    if not (just_ended or still_in):
        return []

    status = "FORMING" if just_ended else "FORMING"

    run_len = best_end - best_start + 1
    # Confidence 60-75 based on run length (longer = more reliable coiling)
    confidence = float(np.clip(60 + (run_len - 3) * 3, 60, 75))

    mid_close = float(df["close"].astype(float).iloc[best_end])
    points = [[i, float(df["close"].astype(float).iloc[i])] for i in range(best_start, best_end + 1)]

    return [_make(
        pattern_name="Volume Dry Up",
        status=status,
        direction="neutral",
        confidence=confidence,
        breakout_level=round(float(df["high"].astype(float).iloc[best_start:best_end + 1].max()), 4),
        invalidation_level=round(float(df["low"].astype(float).iloc[best_start:best_end + 1].min()), 4),
        target=None,
        points=points,
    )]


def _compute_obv(df: pd.DataFrame) -> pd.Series:
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    direction = np.sign(close.diff().fillna(0.0))
    obv = (direction * volume).cumsum()
    return obv


def _find_swing_lows(series: pd.Series, window: int = 3) -> list[int]:
    """Return indices of swing lows (local minima with given half-window)."""
    indices = []
    vals = series.values
    n = len(vals)
    for i in range(window, n - window):
        if all(vals[i] <= vals[i - j] for j in range(1, window + 1)) and \
           all(vals[i] <= vals[i + j] for j in range(1, window + 1)):
            indices.append(i)
    return indices


def _find_swing_highs(series: pd.Series, window: int = 3) -> list[int]:
    """Return indices of swing highs (local maxima with given half-window)."""
    indices = []
    vals = series.values
    n = len(vals)
    for i in range(window, n - window):
        if all(vals[i] >= vals[i - j] for j in range(1, window + 1)) and \
           all(vals[i] >= vals[i + j] for j in range(1, window + 1)):
            indices.append(i)
    return indices


def _detect_obv_divergence(df: pd.DataFrame) -> list[dict[str, Any]]:
    """OBV Divergence — bullish when price LL but OBV HL; bearish when price HH but OBV LH."""
    if len(df) < 30:
        return []

    results: list[dict[str, Any]] = []
    lookback = min(30, len(df))
    window_df = df.tail(lookback).reset_index(drop=True)
    offset = len(df) - lookback  # global bar offset for point indices

    close = window_df["close"].astype(float)
    obv = _compute_obv(window_df)

    # --- Bullish divergence: price lower low, OBV higher low ---
    price_lows = _find_swing_lows(close, window=2)
    obv_lows = _find_swing_lows(obv, window=2)

    if len(price_lows) >= 2 and len(obv_lows) >= 2:
        # Compare last two swing lows
        p1_idx, p2_idx = price_lows[-2], price_lows[-1]
        # Find closest OBV swing lows to the price swing lows
        o1 = min(obv_lows, key=lambda x: abs(x - p1_idx))
        o2 = min(obv_lows, key=lambda x: abs(x - p2_idx))

        price_ll = float(close.iloc[p2_idx]) < float(close.iloc[p1_idx])
        obv_hl = float(obv.iloc[o2]) > float(obv.iloc[o1])

        if price_ll and obv_hl:
            obv_spread = abs(float(obv.iloc[o2]) - float(obv.iloc[o1]))
            obv_mean = abs(float(obv.iloc[o1]))
            strength = float(np.clip(obv_spread / max(obv_mean, 1.0), 0, 1))
            confidence = float(np.clip(68 + strength * 12, 68, 80))
            results.append(_make(
                pattern_name="Bullish OBV Divergence",
                status="READY",
                direction="bullish",
                confidence=confidence,
                breakout_level=round(float(close.iloc[-1]) * 1.01, 4),
                invalidation_level=round(float(close.iloc[p2_idx]), 4),
                target=None,
                points=[
                    [offset + p1_idx, float(close.iloc[p1_idx])],
                    [offset + p2_idx, float(close.iloc[p2_idx])],
                ],
            ))

    # --- Bearish divergence: price higher high, OBV lower high ---
    price_highs = _find_swing_highs(close, window=2)
    obv_highs = _find_swing_highs(obv, window=2)

    if len(price_highs) >= 2 and len(obv_highs) >= 2:
        p1_idx, p2_idx = price_highs[-2], price_highs[-1]
        o1 = min(obv_highs, key=lambda x: abs(x - p1_idx))
        o2 = min(obv_highs, key=lambda x: abs(x - p2_idx))

        price_hh = float(close.iloc[p2_idx]) > float(close.iloc[p1_idx])
        obv_lh = float(obv.iloc[o2]) < float(obv.iloc[o1])

        if price_hh and obv_lh:
            obv_spread = abs(float(obv.iloc[o1]) - float(obv.iloc[o2]))
            obv_mean = abs(float(obv.iloc[o1]))
            strength = float(np.clip(obv_spread / max(obv_mean, 1.0), 0, 1))
            confidence = float(np.clip(68 + strength * 12, 68, 80))
            results.append(_make(
                pattern_name="Bearish OBV Divergence",
                status="READY",
                direction="bearish",
                confidence=confidence,
                breakout_level=round(float(close.iloc[-1]) * 0.99, 4),
                invalidation_level=round(float(close.iloc[p2_idx]), 4),
                target=None,
                points=[
                    [offset + p1_idx, float(close.iloc[p1_idx])],
                    [offset + p2_idx, float(close.iloc[p2_idx])],
                ],
            ))

    return results


def _detect_volume_accumulation(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Institutional Accumulation / Distribution via volume-weighted day counts."""
    if len(df) < 25:
        return []

    tail = df.tail(20).reset_index(drop=True)
    offset = len(df) - 20

    volume = tail["volume"].astype(float)
    close = tail["close"].astype(float)
    open_ = tail["open"].astype(float)
    avg_vol = float(volume.mean())
    if avg_vol <= 0:
        return []

    acc_days = []
    dist_days = []

    for i in range(len(tail)):
        vol = float(volume.iloc[i])
        if vol < 1.2 * avg_vol:
            continue
        c = float(close.iloc[i])
        o = float(open_.iloc[i])
        if c > o:
            acc_days.append(i)
        elif c < o:
            dist_days.append(i)

    results: list[dict[str, Any]] = []
    n_acc = len(acc_days)
    n_dist = len(dist_days)
    last_close = float(close.iloc[-1])
    atr = _compute_atr(df)

    if n_acc > n_dist + 3:
        excess = n_acc - n_dist
        confidence = float(np.clip(65 + excess * 2.5, 65, 82))
        results.append(_make(
            pattern_name="Institutional Accumulation",
            status="FORMING",
            direction="bullish",
            confidence=confidence,
            breakout_level=round(float(tail["high"].astype(float).max()), 4),
            invalidation_level=round(float(tail["low"].astype(float).min()), 4),
            target=round(float(tail["high"].astype(float).max()) + atr * 3, 4),
            points=[[offset + i, float(close.iloc[i])] for i in acc_days],
        ))
    elif n_dist > n_acc + 3:
        excess = n_dist - n_acc
        confidence = float(np.clip(65 + excess * 2.5, 65, 82))
        results.append(_make(
            pattern_name="Institutional Distribution",
            status="FORMING",
            direction="bearish",
            confidence=confidence,
            breakout_level=round(float(tail["low"].astype(float).min()), 4),
            invalidation_level=round(float(tail["high"].astype(float).max()), 4),
            target=round(float(tail["low"].astype(float).min()) - atr * 3, 4),
            points=[[offset + i, float(close.iloc[i])] for i in dist_days],
        ))

    return results


def _detect_hvn_rejection(df: pd.DataFrame) -> list[dict[str, Any]]:
    """High Volume Node (HVN) Rejection — price touching a high-volume price cluster."""
    if len(df) < 30:
        return []

    tail = df.tail(50).reset_index(drop=True)
    offset = len(df) - 50

    high = tail["high"].astype(float)
    low = tail["low"].astype(float)
    close = tail["close"].astype(float)
    open_ = tail["open"].astype(float)
    volume = tail["volume"].astype(float)

    price_min = float(low.min())
    price_max = float(high.max())
    price_range = price_max - price_min
    if price_range <= 0:
        return []

    # Build a volume profile: 20 price buckets
    n_buckets = 20
    bucket_size = price_range / n_buckets
    bucket_vol = np.zeros(n_buckets)

    for i in range(len(tail)):
        mid = (float(high.iloc[i]) + float(low.iloc[i])) / 2.0
        bucket = int((mid - price_min) / bucket_size)
        bucket = max(0, min(n_buckets - 1, bucket))
        bucket_vol[bucket] += float(volume.iloc[i])

    # Top 3 highest-volume buckets → HVN price levels
    top_bucket_indices = np.argsort(bucket_vol)[-3:][::-1]
    hvn_levels = [price_min + (b + 0.5) * bucket_size for b in top_bucket_indices]

    current_close = float(close.iloc[-1])
    current_open = float(open_.iloc[-1])
    last_idx_global = len(df) - 1
    atr = _compute_atr(df)

    results: list[dict[str, Any]] = []

    for node_price in hvn_levels:
        proximity = abs(current_close - node_price) / max(node_price, 1e-9)
        if proximity > 0.005:
            continue

        bucket_rank = float(np.searchsorted(np.sort(bucket_vol), bucket_vol[np.searchsorted(
            [price_min + (b + 0.5) * bucket_size for b in range(n_buckets)],
            node_price
        )]))
        vol_percentile = float(np.clip(bucket_rank / n_buckets, 0, 1))
        confidence = float(np.clip(70 + vol_percentile * 10, 70, 80))

        # Determine direction from current bar's close vs open
        if current_close < current_open:
            # Rejecting downward from resistance node
            results.append(_make(
                pattern_name="HVN Resistance Rejection",
                status="READY",
                direction="bearish",
                confidence=confidence,
                breakout_level=round(node_price * 0.99, 4),
                invalidation_level=round(node_price * 1.005, 4),
                target=round(node_price - atr * 3, 4),
                points=[[last_idx_global, current_close]],
            ))
        elif current_close > current_open:
            # Bouncing upward off support node
            results.append(_make(
                pattern_name="HVN Support Bounce",
                status="READY",
                direction="bullish",
                confidence=confidence,
                breakout_level=round(node_price * 1.01, 4),
                invalidation_level=round(node_price * 0.995, 4),
                target=round(node_price + atr * 3, 4),
                points=[[last_idx_global, current_close]],
            ))

    return results


def detect(symbol: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect volume-based patterns in the given OHLCV DataFrame.

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
        Returns [] if df has fewer than 25 rows or on any unhandled error.
    """
    try:
        if df is None or len(df) < 25:
            return []

        df = df.sort_values("date").reset_index(drop=True).copy()

        results: list[dict[str, Any]] = []
        results.extend(_detect_climactic_volume(df))
        results.extend(_detect_volume_dry_up(df))
        results.extend(_detect_obv_divergence(df))
        results.extend(_detect_volume_accumulation(df))
        results.extend(_detect_hvn_rejection(df))

        # Filter out NOT_PRESENT patterns and sort by confidence descending
        results = [r for r in results if r.get("status") != "NOT_PRESENT"]
        results.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        return results

    except Exception:
        return []
