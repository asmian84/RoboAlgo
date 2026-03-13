"""Liquidity Trap Engine.

Detects stop-hunt style traps:
1) price breaks support/resistance,
2) closes back inside range within 3 candles,
3) volume spike occurs.

trap_score weights:
- level_break: 40%
- reversal_speed: 30%
- wick_dominance: 30%
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import text

from database.connection import get_engine

LOOKBACK_BARS = 40
RETEST_BARS = 3
BREAK_FLOOR_PCT = 0.001
BREAK_CAP_PCT = 0.03
WICK_FLOOR = 0.20
WICK_CAP = 0.60


def _linear_score(value: float, floor: float, cap: float) -> float:
    if value <= floor:
        return 0.0
    if value >= cap:
        return 100.0
    return (value - floor) / (cap - floor) * 100.0


class LiquidityTrapEngine:
    def detect(self, symbol: str, as_of_date: date | None = None) -> dict:
        q = text(
            """
            WITH picked AS (
              SELECT i.id, i.symbol FROM instruments i WHERE i.symbol = :symbol
            ), ranked AS (
              SELECT p.date, p.open, p.high, p.low, p.close, p.volume,
                     ROW_NUMBER() OVER (ORDER BY p.date DESC) AS rn
              FROM price_data p
              JOIN picked s ON s.id = p.instrument_id
              WHERE (:as_of_date IS NULL OR p.date <= :as_of_date)
            )
            SELECT date, open, high, low, close, volume
            FROM ranked
            WHERE rn <= :bars
            ORDER BY date
            """
        )
        df = pd.read_sql_query(
            q,
            get_engine(),
            params={"symbol": symbol.upper(), "as_of_date": as_of_date, "bars": LOOKBACK_BARS + RETEST_BARS + 8},
        )

        if df.empty or len(df) < LOOKBACK_BARS + RETEST_BARS:
            return {"symbol": symbol.upper(), "trap_score": 0.0, "trap_type": "bullish"}

        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        open_ = df["open"].to_numpy(dtype=float)
        volume = df["volume"].fillna(0.0).to_numpy(dtype=float)

        bidx = len(df) - RETEST_BARS - 1
        swing_high = float(np.max(high[:bidx]))
        swing_low = float(np.min(low[:bidx]))

        b_high, b_low = float(high[bidx]), float(low[bidx])
        b_open, b_close = float(open_[bidx]), float(close[bidx])

        trap_type = None
        break_pct = 0.0
        level = 0.0

        if b_high > swing_high * (1.0 + BREAK_FLOOR_PCT):
            trap_type = "bearish"  # trap above resistance then reverse down
            level = swing_high
            break_pct = (b_high - swing_high) / max(swing_high, 1e-9)
        elif b_low < swing_low * (1.0 - BREAK_FLOOR_PCT):
            trap_type = "bullish"  # trap below support then reverse up
            level = swing_low
            break_pct = (swing_low - b_low) / max(swing_low, 1e-9)

        if trap_type is None:
            return {"symbol": symbol.upper(), "trap_score": 0.0, "trap_type": "bullish"}

        # level_break: smaller fake break = stronger trap
        normalized = (break_pct - BREAK_FLOOR_PCT) / max(BREAK_CAP_PCT - BREAK_FLOOR_PCT, 1e-9)
        level_break = max(0.0, 100.0 - normalized * 100.0)

        # reversal_speed: how quickly closes back inside range in next 3 bars
        reversal_speed = 0.0
        for i in range(1, RETEST_BARS + 1):
            c = float(close[bidx + i])
            back_inside = (trap_type == "bearish" and c < level) or (trap_type == "bullish" and c > level)
            if back_inside:
                reversal_speed = max(0.0, 100.0 * (1.0 - (i - 1) / RETEST_BARS))
                break

        # wick_dominance
        rng = max(b_high - b_low, 1e-9)
        if trap_type == "bearish":
            dom_wick = b_high - max(b_open, b_close)
        else:
            dom_wick = min(b_open, b_close) - b_low
        wick_ratio = max(0.0, dom_wick) / rng
        wick_dominance = _linear_score(wick_ratio, WICK_FLOOR, WICK_CAP)

        # volume spike gate (as required in logic)
        baseline_vol = float(np.mean(volume[max(0, bidx - 20) : bidx])) if bidx > 5 else float(np.mean(volume[:bidx + 1]))
        vol_spike = float(volume[bidx]) >= baseline_vol * 1.5 if baseline_vol > 0 else False

        raw_score = 0.4 * level_break + 0.3 * reversal_speed + 0.3 * wick_dominance
        trap_score = float(round(raw_score if vol_spike else raw_score * 0.6, 1))

        return {
            "symbol": symbol.upper(),
            "trap_score": trap_score,
            "trap_type": "bullish" if trap_type == "bullish" else "bearish",
        }
