"""Time Exhaustion Detector — identifies when a directional move is overextended.

A move becomes "exhausted" when price has produced too many consecutive
directional candles without correction.  At exhaustion, reversal probability
rises sharply because:
    • Momentum participants are fully committed.
    • The move has consumed the available fuel (stop-loss clusters).
    • Mean-reversion pressure builds.

Detection method
----------------
Two complementary exhaustion signals are computed and combined:

1. **Consecutive candle streak** (primary)
   Count consecutive bars where ``close > open`` (bullish) or
   ``close < open`` (bearish) without a countertrend candle interrupting.
   Compared to a timeframe-aware threshold.

2. **Close-to-close streak** (secondary)
   Count consecutive bars where ``close > prev_close`` (up) or
   ``close < prev_close`` (down).  This catches grinding moves that have
   mixed candle bodies but a sustained directional close sequence.

Both streaks are tracked independently and the larger of the two ratios
(streak / threshold) is used as the primary exhaustion signal.  When both
fire, confidence is elevated.

Timeframe thresholds
--------------------
Shorter timeframes generate more noise so require longer streaks:

    1m   → 12 candles  (midpoint of 10–15)
    5m   → 8  candles  (midpoint of 7–10)
    15m  → 7  candles  (midpoint of 6–8)
    30m  → 6  candles
    1h   → 6  candles  (midpoint of 5–7)
    2h   → 5  candles
    4h   → 5  candles
    daily→ 5  candles  (midpoint of 4–6)
    weekly→ 4 candles

Custom threshold values can be passed directly to override the table.

Output
------
{
    "exhaustion_detected":   bool,
    "direction":             "UP" | "DOWN" | "NONE",
    "exhaustion_strength":   float,   # 0–1  (streak / threshold ratio)
    "candle_streak":         int,     # consecutive body-directional candles
    "close_streak":          int,     # consecutive close-to-close directional bars
    "threshold":             int,     # timeframe-specific threshold used
    "timeframe":             str,
    "both_signals":          bool,    # True when both streak types agree
}

IMPORTANT: No indicator calculations are performed.  Only raw OHLCV price
comparisons (close vs open, close vs prev close) are used.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── Timeframe threshold table ──────────────────────────────────────────────────

_TF_THRESHOLDS: dict[str, int] = {
    "1m":    12,
    "2m":    10,
    "5m":     8,
    "10m":    7,
    "15m":    7,
    "30m":    6,
    "1h":     6,
    "2h":     5,
    "4h":     5,
    "daily":  5,
    "1d":     5,
    "weekly": 4,
    "1w":     4,
    "monthly":3,
}

_DEFAULT_THRESHOLD = 7   # fallback when timeframe not in table


class TimeExhaustionEngine:
    """Detect when a directional move has persisted too long without correction.

    Parameters
    ----------
    timeframe:
        Human-readable timeframe label (e.g. ``"daily"``, ``"1h"``, ``"5m"``).
        Used to look up the candle-streak threshold.
    threshold:
        Override the timeframe-derived threshold.  When *None* (default),
        the threshold is looked up from the timeframe table.
    symbol:
        Optional ticker symbol for logging context.
    """

    def __init__(
        self,
        timeframe: str = "daily",
        threshold: int | None = None,
        symbol: str = "",
    ) -> None:
        self.timeframe = timeframe.lower().strip()
        self.symbol    = symbol
        self._threshold = (
            threshold
            if threshold is not None
            else _TF_THRESHOLDS.get(self.timeframe, _DEFAULT_THRESHOLD)
        )

    # ── Public API ──────────────────────────────────────────────────────────────

    @property
    def threshold(self) -> int:
        return self._threshold

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        """Detect time exhaustion on the most recent bar of ``df``.

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close.
                Must contain at least ``threshold + 2`` bars.

        Returns:
            Exhaustion dict (see module docstring for schema).
        """
        min_bars = self._threshold + 2
        if len(df) < min_bars:
            logger.debug(
                "TimeExhaustionEngine[%s]: insufficient data (%d bars, need ≥%d)",
                self.symbol, len(df), min_bars,
            )
            return self._no_exhaustion()

        candle_streak, candle_dir = _count_candle_streak(df)
        close_streak,  close_dir  = _count_close_streak(df)

        # Use the streak whose direction is agreed upon (or the larger one)
        if candle_dir == close_dir and candle_dir != "NONE":
            direction = candle_dir
        elif candle_streak >= close_streak:
            direction = candle_dir
        else:
            direction = close_dir

        # Primary ratio: how far has the streak exceeded the threshold?
        primary_ratio   = candle_streak / self._threshold
        secondary_ratio = close_streak  / self._threshold

        # Exhaustion detected when either streak reaches the threshold
        exhaustion_detected = (
            candle_streak >= self._threshold or close_streak >= self._threshold
        ) and direction != "NONE"

        # Both signals firing simultaneously raises confidence
        both_signals = (
            candle_streak >= self._threshold
            and close_streak >= self._threshold
            and candle_dir == close_dir
        )

        # Strength: primary ratio capped at 1.0, boosted +0.15 when both signals agree
        base_strength = min(max(primary_ratio, secondary_ratio), 1.5) / 1.5
        strength_bonus = 0.15 if both_signals else 0.0
        exhaustion_strength = round(min(base_strength + strength_bonus, 1.0), 4)

        if not exhaustion_detected:
            exhaustion_strength = round(
                min(max(primary_ratio, secondary_ratio) * 0.6, 0.99), 4
            )
            direction = "NONE"

        result = {
            "exhaustion_detected":  exhaustion_detected,
            "direction":            direction,
            "exhaustion_strength":  exhaustion_strength,
            "candle_streak":        candle_streak,
            "close_streak":         close_streak,
            "threshold":            self._threshold,
            "timeframe":            self.timeframe,
            "both_signals":         both_signals,
        }

        if exhaustion_detected:
            logger.info(
                "TimeExhaustionEngine[%s]: EXHAUSTION dir=%s strength=%.2f "
                "candle_streak=%d close_streak=%d threshold=%d",
                self.symbol, direction, exhaustion_strength,
                candle_streak, close_streak, self._threshold,
            )

        return result

    def scan(
        self,
        df: pd.DataFrame,
        lookback: int = 50,
    ) -> list[dict[str, Any]]:
        """Scan the last ``lookback`` bars and return all exhaustion events found.

        Args:
            df:       OHLCV DataFrame.
            lookback: Number of recent bars to scan.

        Returns:
            List of exhaustion event dicts, each including a ``bar_index`` and
            ``date`` field in addition to the standard exhaustion fields.
        """
        events: list[dict[str, Any]] = []
        start = max(self._threshold + 2, len(df) - lookback)

        for i in range(start, len(df)):
            result = self.run(df.iloc[: i + 1])
            if result["exhaustion_detected"]:
                date_val = df.index[i]
                date_str = (
                    str(date_val.date()) if hasattr(date_val, "date")
                    else str(date_val)
                )
                events.append({**result, "bar_index": i, "date": date_str})

        return events

    def _no_exhaustion(self) -> dict[str, Any]:
        return {
            "exhaustion_detected":  False,
            "direction":            "NONE",
            "exhaustion_strength":  0.0,
            "candle_streak":        0,
            "close_streak":         0,
            "threshold":            self._threshold,
            "timeframe":            self.timeframe,
            "both_signals":         False,
        }


# ── Pure streak counters (no engine calls) ─────────────────────────────────────

def _count_candle_streak(df: pd.DataFrame) -> tuple[int, str]:
    """Count consecutive same-direction candles from the most recent bar.

    A candle is bullish  when ``close > open``.
    A candle is bearish  when ``close < open``.
    A doji (close == open) breaks the streak.

    Returns:
        (streak_length, direction)  where direction is ``"UP"``, ``"DOWN"``,
        or ``"NONE"`` when the streak has length 0 or the last bar is a doji.
    """
    opens  = df["open"].values.astype(float)
    closes = df["close"].values.astype(float)
    n      = len(opens)

    if n == 0:
        return 0, "NONE"

    # Determine direction of the most recent bar
    last_diff = closes[-1] - opens[-1]
    if last_diff > 0:
        target = "UP"
    elif last_diff < 0:
        target = "DOWN"
    else:
        return 0, "NONE"

    count = 0
    for i in range(n - 1, -1, -1):
        diff = closes[i] - opens[i]
        if target == "UP" and diff > 0:
            count += 1
        elif target == "DOWN" and diff < 0:
            count += 1
        else:
            break   # streak broken

    return count, target


def _count_close_streak(df: pd.DataFrame) -> tuple[int, str]:
    """Count consecutive same-direction close-to-close moves from the right.

    A bar is ``"UP"``   when ``close[i] > close[i-1]``.
    A bar is ``"DOWN"`` when ``close[i] < close[i-1]``.
    An unchanged close breaks the streak.

    Returns:
        (streak_length, direction)
    """
    closes = df["close"].values.astype(float)
    n      = len(closes)

    if n < 2:
        return 0, "NONE"

    # Determine direction at the last bar
    last_delta = closes[-1] - closes[-2]
    if last_delta > 0:
        target = "UP"
    elif last_delta < 0:
        target = "DOWN"
    else:
        return 0, "NONE"

    count = 0
    for i in range(n - 1, 0, -1):
        delta = closes[i] - closes[i - 1]
        if target == "UP" and delta > 0:
            count += 1
        elif target == "DOWN" and delta < 0:
            count += 1
        else:
            break

    return count, target


# ── Convenience helpers ────────────────────────────────────────────────────────

def get_threshold_for_timeframe(timeframe: str) -> int:
    """Return the exhaustion threshold for a given timeframe label."""
    return _TF_THRESHOLDS.get(timeframe.lower().strip(), _DEFAULT_THRESHOLD)


def detect_exhaustion(
    df: pd.DataFrame,
    timeframe: str = "daily",
    symbol: str = "",
) -> dict[str, Any]:
    """Functional wrapper around :class:`TimeExhaustionEngine`.

    Args:
        df:        OHLCV DataFrame.
        timeframe: Timeframe label (e.g. ``"daily"``, ``"5m"``).
        symbol:    Optional symbol for logging.

    Returns:
        Exhaustion dict.
    """
    return TimeExhaustionEngine(timeframe=timeframe, symbol=symbol).run(df)
