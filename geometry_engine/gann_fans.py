"""Gann Fan generator — projects fan lines from significant pivot points.

A Gann Fan is a set of 9 angular lines radiating from a major high or low,
used to identify support/resistance based on price-time geometry.
Each line represents a different rate of price change per unit of time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from geometry_engine.gann_angles import GANN_ANGLE_DEFS, compute_gann_angles
from structure_engine.swing_detector import detect_swings, compute_adaptive_minimum_move


def generate_gann_fans(
    df: pd.DataFrame,
    n_fans: int = 2,
    projection_bars: int = 60,
) -> dict:
    """Generate Gann fans from the most significant recent pivots.

    Args:
        df: OHLCV DataFrame with columns [date, open, high, low, close, volume].
        n_fans: Number of fans to generate (from most significant pivots).
        projection_bars: How many bars forward to project each fan line.

    Returns:
        {
            "fans": [
                {
                    "anchor_date": str,
                    "anchor_price": float,
                    "anchor_type": "high" | "low",
                    "lines": [
                        {"angle": "1x1", "prices": [float, ...], "direction": "up"|"down"}
                    ]
                }
            ],
            "overlay_lines": [[[date1, price1], [date2, price2]], ...],
        }
    """
    if df is None or len(df) < 30:
        return {"fans": [], "overlay_lines": []}

    df = df.sort_values("date").reset_index(drop=True).copy()
    close = df["close"].astype(float).values

    # Detect swings
    adaptive_mm = compute_adaptive_minimum_move(df)
    swings = detect_swings(df, minimum_move=adaptive_mm)

    # Find most significant pivots (largest swing amplitude)
    all_pivots = []
    for h in swings["swing_highs"]:
        all_pivots.append({**h, "type": "high"})
    for lo in swings["swing_lows"]:
        all_pivots.append({**lo, "type": "low"})

    if not all_pivots:
        return {"fans": [], "overlay_lines": []}

    # Sort by significance (price distance from neighbors)
    all_pivots.sort(key=lambda p: p["index"])
    scored = []
    for i, p in enumerate(all_pivots):
        left = all_pivots[i - 1]["price"] if i > 0 else p["price"]
        right = all_pivots[i + 1]["price"] if i < len(all_pivots) - 1 else p["price"]
        amplitude = max(abs(p["price"] - left), abs(p["price"] - right))
        scored.append((amplitude, p))

    scored.sort(key=lambda x: -x[0])
    top_pivots = [p for _, p in scored[:n_fans]]

    # Compute ATR for scaling
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    tr = high - low
    atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))
    if atr <= 0:
        atr = float(np.mean(close)) * 0.02

    # Unit price per bar for 1x1 angle (1 ATR per bar is the "natural" scale)
    unit_price = atr

    fans = []
    overlay_lines = []
    dates = df["date"].astype(str).tolist()
    n_bars = len(df)

    for pivot in top_pivots:
        anchor_idx = pivot["index"]
        anchor_price = pivot["price"]
        anchor_type = pivot["type"]
        anchor_date = dates[anchor_idx] if anchor_idx < len(dates) else dates[-1]

        # Fan direction: from a high → lines go down; from a low → lines go up
        direction = "down" if anchor_type == "high" else "up"
        sign = -1.0 if direction == "down" else 1.0

        fan_lines = []
        for angle_name, multiplier in GANN_ANGLE_DEFS:
            slope = sign * multiplier * unit_price  # price change per bar

            prices = []
            for bar_offset in range(projection_bars + 1):
                proj_price = anchor_price + slope * bar_offset
                prices.append(round(proj_price, 4))

            fan_lines.append({
                "angle": angle_name,
                "prices": prices,
                "direction": direction,
            })

            # Create overlay line segment (anchor → projection end)
            end_idx = anchor_idx + projection_bars
            end_date = dates[min(end_idx, n_bars - 1)]
            end_price = anchor_price + slope * projection_bars

            overlay_lines.append([
                [anchor_date, round(anchor_price, 4)],
                [end_date, round(end_price, 4)],
            ])

        fans.append({
            "anchor_date": anchor_date,
            "anchor_price": round(anchor_price, 4),
            "anchor_type": anchor_type,
            "lines": fan_lines,
        })

    return {
        "fans": fans,
        "overlay_lines": overlay_lines,
    }


def fan_adherence_score(
    df: pd.DataFrame,
    fan: dict,
    tolerance_pct: float = 0.02,
) -> float:
    """Measure how closely price adheres to fan lines (0-100).

    Higher score = price is frequently near a fan line = strong geometry.
    """
    if not fan or "lines" not in fan:
        return 0.0

    close = df["close"].astype(float).values
    dates = df["date"].astype(str).tolist()
    n = len(close)

    # Find anchor index
    anchor_date = fan.get("anchor_date", "")
    try:
        anchor_idx = dates.index(anchor_date)
    except ValueError:
        return 0.0

    touches = 0
    tested = 0

    for bar_idx in range(anchor_idx + 1, n):
        offset = bar_idx - anchor_idx
        price = close[bar_idx]
        tested += 1

        for line in fan["lines"]:
            if offset < len(line["prices"]):
                fan_price = line["prices"][offset]
                if abs(price - fan_price) / (price + 1e-8) <= tolerance_pct:
                    touches += 1
                    break  # one touch per bar is enough

    if tested == 0:
        return 0.0

    return round(min(touches / tested * 100, 100), 2)
