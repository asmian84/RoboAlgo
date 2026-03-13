"""Liquidity sweep pattern detector using swing levels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pattern_engine.common import (
    base_result,
    composite_probability,
    liquidity_alignment_score,
    market_regime_score,
    momentum_score,
    volume_confirmation,
)
from structure_engine.swing_detector import detect_swings, compute_adaptive_minimum_move


def detect(symbol: str, price_data: pd.DataFrame) -> dict[str, Any]:
    result = base_result("Liquidity Sweep")
    if price_data is None or len(price_data) < 50:
        return result

    df = price_data.sort_values("date").reset_index(drop=True).copy()
    adaptive_mm = compute_adaptive_minimum_move(df)
    swings = detect_swings(df, minimum_move=adaptive_mm)
    highs = swings["swing_highs"]
    lows = swings["swing_lows"]
    if not highs or not lows or len(df) < 6:
        return result

    recent = df.tail(4).reset_index(drop=True)
    break_bar = recent.iloc[0]
    follow = recent.iloc[1:]
    ref_high = max(h["price"] for h in highs[:-1]) if len(highs) > 1 else highs[-1]["price"]
    ref_low = min(l["price"] for l in lows[:-1]) if len(lows) > 1 else lows[-1]["price"]

    sweep_type = None
    level = None
    if float(break_bar["high"]) > ref_high:
        sweep_type = "bearish"
        level = float(ref_high)
        reverted = (follow["close"].astype(float) < level).any()
        invalidation = float(break_bar["high"])
        target = float(level - (invalidation - level) * 1.5)
    elif float(break_bar["low"]) < ref_low:
        sweep_type = "bullish"
        level = float(ref_low)
        reverted = (follow["close"].astype(float) > level).any()
        invalidation = float(break_bar["low"])
        target = float(level + (level - invalidation) * 1.5)
    else:
        return result

    if not reverted or level is None:
        return result

    vol = df["volume"].astype(float).tail(30)
    baseline = float(vol.iloc[:-4].mean()) if len(vol) > 6 else float(vol.mean())
    vol_spike = float(break_bar.get("volume", 0.0) or 0.0) > baseline * 1.3 if baseline > 0 else False

    structure_quality = 72.0 if vol_spike else 58.0
    probability = composite_probability(
        structure_quality=structure_quality,
        volume=volume_confirmation(df),
        liquidity=liquidity_alignment_score(df, level),
        regime=market_regime_score(df),
        momentum=momentum_score(df),
    )

    last_close = float(df["close"].iloc[-1])
    if sweep_type == "bullish":
        status = "COMPLETED" if last_close >= target else "BREAKOUT" if last_close > level else "FORMING"
    else:
        status = "COMPLETED" if last_close <= target else "BREAKOUT" if last_close < level else "FORMING"
    if (sweep_type == "bullish" and last_close <= invalidation) or (sweep_type == "bearish" and last_close >= invalidation):
        status = "FAILED"

    result.update(
        {
            "status": status,
            "breakout_level": round(level, 4),
            "invalidation_level": round(invalidation, 4),
            "projected_target": round(target, 4),
            "probability": round(probability, 2),
            "points": [
                [len(df) - 4, float(break_bar["high"])],
                [len(df) - 4, float(break_bar["low"])],
            ],
        }
    )
    return result

