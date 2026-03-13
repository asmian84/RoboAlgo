"""Shared utilities for pattern detectors."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PATTERN_STATUS = ("NOT_PRESENT", "FORMING", "READY", "BREAKOUT", "FAILED", "COMPLETED")


def base_result(pattern_name: str) -> dict[str, Any]:
    return {
        "pattern_name": pattern_name,
        "status": "NOT_PRESENT",
        "breakout_level": None,
        "invalidation_level": None,
        "projected_target": None,
        "probability": 0.0,
        "points": [],
    }


def volume_confirmation(df: pd.DataFrame, lookback: int = 20) -> float:
    if "volume" not in df.columns or df["volume"].dropna().empty:
        return 50.0
    tail = df["volume"].astype(float).tail(lookback + 1)
    if len(tail) < 2:
        return 50.0
    last = float(tail.iloc[-1])
    mean = float(tail.iloc[:-1].mean())
    ratio = last / max(mean, 1.0)
    return float(np.clip(40 + ratio * 30, 0, 100))


def momentum_score(df: pd.DataFrame, lookback: int = 14) -> float:
    close = df["close"].astype(float)
    if len(close) < lookback + 2:
        return 50.0
    ret = close.pct_change().tail(lookback).dropna()
    if ret.empty:
        return 50.0
    mean = float(ret.mean())
    std = float(ret.std() or 1e-9)
    z = mean / max(std, 1e-9)
    return float(np.clip(50 + z * 15, 0, 100))


def market_regime_score(df: pd.DataFrame, lookback: int = 30) -> float:
    if len(df) < lookback + 2:
        return 50.0
    close = df["close"].astype(float).tail(lookback + 1)
    vol = close.pct_change().dropna().std()
    trend = (close.iloc[-1] / max(close.iloc[0], 1e-9)) - 1.0
    score = 55 + trend * 120 - (vol or 0.0) * 80
    return float(np.clip(score, 0, 100))


def liquidity_alignment_score(df: pd.DataFrame, breakout_level: float | None) -> float:
    if breakout_level is None or len(df) < 5:
        return 50.0
    high = float(df["high"].astype(float).tail(40).max())
    low = float(df["low"].astype(float).tail(40).min())
    if high <= low:
        return 50.0
    distance = min(abs(high - breakout_level), abs(breakout_level - low)) / max(high - low, 1e-9)
    return float(np.clip(100 - distance * 180, 0, 100))


def composite_probability(
    structure_quality: float,
    volume: float,
    liquidity: float,
    regime: float,
    momentum: float,
) -> float:
    return float(
        np.clip(
            0.40 * structure_quality
            + 0.25 * volume
            + 0.20 * liquidity
            + 0.10 * regime
            + 0.05 * momentum,
            0,
            100,
        )
    )


def status_from_levels(
    close: pd.Series,
    breakout_level: float | None,
    invalidation_level: float | None,
    target: float | None,
    bullish: bool = True,
    ready_buffer: float = 0.005,
) -> str:
    if breakout_level is None or invalidation_level is None or target is None or close.empty:
        return "NOT_PRESENT"

    last = float(close.iloc[-1])
    if bullish:
        if last <= invalidation_level:
            return "FAILED"
        if last >= target:
            return "COMPLETED"
        if last >= breakout_level:
            return "BREAKOUT"
        if last >= breakout_level * (1 - ready_buffer):
            return "READY"
        return "FORMING"

    if last >= invalidation_level:
        return "FAILED"
    if last <= target:
        return "COMPLETED"
    if last <= breakout_level:
        return "BREAKOUT"
    if last <= breakout_level * (1 + ready_buffer):
        return "READY"
    return "FORMING"

