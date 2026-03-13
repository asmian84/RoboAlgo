"""Time Exhaustion Detector — identifies overextended directional moves.

Tracks consecutive directional candles (body and close-to-close) and fires
when the streak exceeds a timeframe-calibrated threshold, signalling that
the move has consumed its fuel and a reversal is increasingly likely.

Timeframe thresholds (midpoints of spec ranges):
    1m → 12,  5m → 8,  15m → 7,  1h → 6,  daily → 5,  weekly → 4

Output format::

    {
        "exhaustion_detected":  bool,
        "direction":            "UP" | "DOWN" | "NONE",
        "exhaustion_strength":  float,   # 0–1
        "candle_streak":        int,
        "close_streak":         int,
        "threshold":            int,
        "timeframe":            str,
        "both_signals":         bool,
    }
"""

from time_exhaustion.exhaustion_engine import (
    TimeExhaustionEngine,
    detect_exhaustion,
    get_threshold_for_timeframe,
)

__all__ = [
    "TimeExhaustionEngine",
    "detect_exhaustion",
    "get_threshold_for_timeframe",
]
