"""Price-time symmetry detector.

Detects when price_move ≈ time_move (in normalized units), a key Gann concept
where markets tend to reverse when the price change in points equals the
time change in bars/days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from structure_engine.swing_detector import detect_swings, compute_adaptive_minimum_move


def compute_price_time_symmetry(df: pd.DataFrame) -> dict:
    """Compute price-time symmetry score from swing data.

    Checks recent swing moves for cases where:
    price_move (in ATR-normalized units) ≈ time_move (in bars)

    Returns dict with symmetry_score (0-100) and symmetry_ratio.
    """
    result = {"symmetry_score": 0.0, "symmetry_ratio": 0.0, "symmetric_swings": []}

    if df is None or len(df) < 60:
        return result

    df = df.sort_values("date").reset_index(drop=True).copy()
    adaptive_mm = compute_adaptive_minimum_move(df)
    swings = detect_swings(df, minimum_move=adaptive_mm)
    pivots = sorted(
        swings["swing_highs"] + swings["swing_lows"],
        key=lambda p: p["index"],
    )

    if len(pivots) < 3:
        return result

    # Compute ATR for normalization
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))
    if atr <= 0:
        return result

    # Check symmetry for each consecutive swing pair
    symmetry_ratios: list[float] = []
    symmetric_swings: list[dict] = []

    for i in range(len(pivots) - 1):
        p0 = pivots[i]
        p1 = pivots[i + 1]
        price_move = abs(p1["price"] - p0["price"]) / atr  # normalized by ATR
        time_move = abs(p1["index"] - p0["index"])           # in bars

        if time_move == 0 or price_move == 0:
            continue

        ratio = price_move / time_move
        symmetry_ratios.append(ratio)

        # Near-symmetric when ratio is close to 1.0
        if 0.7 <= ratio <= 1.4:
            symmetric_swings.append({
                "from_idx": p0["index"],
                "to_idx": p1["index"],
                "price_move_atr": round(price_move, 2),
                "time_move_bars": int(time_move),
                "ratio": round(ratio, 3),
            })

    if not symmetry_ratios:
        return result

    avg_ratio = float(np.mean(symmetry_ratios))
    n_symmetric = len(symmetric_swings)
    n_total = len(symmetry_ratios)

    # Score: higher when more swings are symmetric and avg ratio near 1.0
    ratio_score = max(0, 1.0 - abs(avg_ratio - 1.0)) * 50
    pct_symmetric = (n_symmetric / max(n_total, 1)) * 50
    score = float(np.clip(ratio_score + pct_symmetric, 0, 100))

    result["symmetry_score"] = round(score, 2)
    result["symmetry_ratio"] = round(avg_ratio, 4)
    result["symmetric_swings"] = symmetric_swings[-5:]  # last 5
    # Alias for confluence scorer compatibility
    result["symmetry_zones"] = result["symmetric_swings"]

    return result
