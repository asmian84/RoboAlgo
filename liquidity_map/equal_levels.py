"""Equal Levels Detector — finds clusters of equal highs and equal lows.

Equal highs or lows occur when price tests the same level multiple times,
creating a cluster of resting stop-loss orders.  The detection tolerance is
configurable (default 0.2%) to handle minor wick variations.

Cluster strength is a weighted blend of:
    60% — touch count  (more retests = stronger cluster)
    40% — recency      (more recent = not yet swept = stronger)
"""

from __future__ import annotations

import pandas as pd


def detect_equal_highs(
    df: pd.DataFrame,
    tolerance: float = 0.002,
    min_touches: int = 2,
    lookback: int = 50,
) -> list[dict]:
    """Detect clusters of equal highs within a configurable tolerance band.

    Args:
        df:          OHLCV DataFrame — must contain a ``high`` column.
        tolerance:   Maximum relative difference to consider two highs equal
                     (e.g. 0.002 = 0.2%).  Default 0.2%.
        min_touches: Minimum number of bars at the same level to form a cluster.
        lookback:    Number of recent bars to scan.

    Returns:
        List of cluster dicts::

            {
                "price":       float,      # mean price of the cluster
                "touches":     int,        # number of bars in the cluster
                "bar_indices": list[int],  # bar positions within the window
                "strength":    float,      # 0–1 liquidity strength score
            }
    """
    window = df.tail(lookback).reset_index(drop=True)
    highs  = window["high"].values
    n      = len(highs)

    clusters: list[dict] = []
    visited: set[int]    = set()

    for i in range(n):
        if i in visited:
            continue

        ref   = highs[i]
        group = [i]

        for j in range(i + 1, n):
            if j in visited:
                continue
            # Relative tolerance check
            if abs(highs[j] - ref) / max(ref, 1e-9) < tolerance:
                group.append(j)
                visited.add(j)

        if len(group) >= min_touches:
            visited.add(i)
            mean_price    = sum(highs[k] for k in group) / len(group)
            recency_score = max(group) / max(n - 1, 1)        # 0→oldest, 1→newest
            touch_score   = min(len(group) / 5, 1.0)          # caps at 5 touches = 1.0
            strength      = round(0.6 * touch_score + 0.4 * recency_score, 4)

            clusters.append({
                "price":       round(float(mean_price), 4),
                "touches":     len(group),
                "bar_indices": group,
                "strength":    strength,
            })

    return clusters


def detect_equal_lows(
    df: pd.DataFrame,
    tolerance: float = 0.002,
    min_touches: int = 2,
    lookback: int = 50,
) -> list[dict]:
    """Detect clusters of equal lows within a configurable tolerance band.

    Args:
        df:          OHLCV DataFrame — must contain a ``low`` column.
        tolerance:   Maximum relative difference to consider two lows equal.
        min_touches: Minimum number of bars at the same level to form a cluster.
        lookback:    Number of recent bars to scan.

    Returns:
        List of cluster dicts (same schema as :func:`detect_equal_highs`)::

            {
                "price":       float,
                "touches":     int,
                "bar_indices": list[int],
                "strength":    float,
            }
    """
    window = df.tail(lookback).reset_index(drop=True)
    lows   = window["low"].values
    n      = len(lows)

    clusters: list[dict] = []
    visited: set[int]    = set()

    for i in range(n):
        if i in visited:
            continue

        ref   = lows[i]
        group = [i]

        for j in range(i + 1, n):
            if j in visited:
                continue
            if abs(lows[j] - ref) / max(ref, 1e-9) < tolerance:
                group.append(j)
                visited.add(j)

        if len(group) >= min_touches:
            visited.add(i)
            mean_price    = sum(lows[k] for k in group) / len(group)
            recency_score = max(group) / max(n - 1, 1)
            touch_score   = min(len(group) / 5, 1.0)
            strength      = round(0.6 * touch_score + 0.4 * recency_score, 4)

            clusters.append({
                "price":       round(float(mean_price), 4),
                "touches":     len(group),
                "bar_indices": group,
                "strength":    strength,
            })

    return clusters
