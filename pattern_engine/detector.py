"""
RoboAlgo - Pattern Detection Engine

Detects candlestick and multi-bar chart patterns from OHLCV + indicator data.

Candlestick Patterns (single/multi-candle):
  Doji, Hammer, Inverted Hammer, Hanging Man, Shooting Star,
  Bullish/Bearish Engulfing, Morning/Evening Star,
  Three White Soldiers, Three Black Crows,
  Bullish/Bearish Harami, Piercing Line, Dark Cloud Cover,
  Bullish/Bearish Marubozu

Chart Patterns (multi-day structural):
  Double Bottom, Double Top, Golden Cross, Death Cross,
  Breakout (resistance break), Breakdown (support break),
  Support Bounce, Resistance Rejection
"""

import logging
from datetime import date as date_type

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tqdm import tqdm

from database.connection import get_session
from database.models import Instrument, PriceData, Indicator, PatternDetection

logger = logging.getLogger(__name__)

# All known patterns — used for catalogue/legend display
PATTERN_CATALOGUE = {
    # ---------- Candlestick patterns ----------
    "Doji":                 {"type": "candlestick", "direction": "neutral",
                             "description": "Open ≈ Close, indicating indecision. Signals potential reversal."},
    "Hammer":               {"type": "candlestick", "direction": "bullish",
                             "description": "Small body at top, long lower shadow. Bullish reversal after downtrend."},
    "Inverted Hammer":      {"type": "candlestick", "direction": "bullish",
                             "description": "Small body at bottom, long upper shadow. Potential bullish reversal."},
    "Hanging Man":          {"type": "candlestick", "direction": "bearish",
                             "description": "Small body at top, long lower shadow in uptrend. Bearish reversal warning."},
    "Shooting Star":        {"type": "candlestick", "direction": "bearish",
                             "description": "Small body at bottom, long upper shadow in uptrend. Bearish reversal."},
    "Bullish Engulfing":    {"type": "candlestick", "direction": "bullish",
                             "description": "Large green candle fully engulfs prior red candle. Strong bullish reversal."},
    "Bearish Engulfing":    {"type": "candlestick", "direction": "bearish",
                             "description": "Large red candle fully engulfs prior green candle. Strong bearish reversal."},
    "Morning Star":         {"type": "candlestick", "direction": "bullish",
                             "description": "3-candle: red → small doji/body → large green. Bullish reversal."},
    "Evening Star":         {"type": "candlestick", "direction": "bearish",
                             "description": "3-candle: green → small doji/body → large red. Bearish reversal."},
    "Three White Soldiers": {"type": "candlestick", "direction": "bullish",
                             "description": "Three consecutive large green candles. Strong bullish continuation."},
    "Three Black Crows":    {"type": "candlestick", "direction": "bearish",
                             "description": "Three consecutive large red candles. Strong bearish continuation."},
    "Bullish Harami":       {"type": "candlestick", "direction": "bullish",
                             "description": "Small green candle inside prior red candle body. Bullish reversal hint."},
    "Bearish Harami":       {"type": "candlestick", "direction": "bearish",
                             "description": "Small red candle inside prior green candle body. Bearish reversal hint."},
    "Piercing Line":        {"type": "candlestick", "direction": "bullish",
                             "description": "Green candle opens below prev low, closes above prev midpoint. Bullish."},
    "Dark Cloud Cover":     {"type": "candlestick", "direction": "bearish",
                             "description": "Red candle opens above prev high, closes below prev midpoint. Bearish."},
    "Bullish Marubozu":     {"type": "candlestick", "direction": "bullish",
                             "description": "Full green candle with tiny/no wicks. Strong buying pressure."},
    "Bearish Marubozu":     {"type": "candlestick", "direction": "bearish",
                             "description": "Full red candle with tiny/no wicks. Strong selling pressure."},
    # ---------- Chart patterns ----------
    "Double Bottom":        {"type": "chart", "direction": "bullish",
                             "description": "Two similar lows separated by a peak. Bullish reversal pattern (W shape)."},
    "Double Top":           {"type": "chart", "direction": "bearish",
                             "description": "Two similar highs separated by a trough. Bearish reversal pattern (M shape)."},
    "Golden Cross":         {"type": "chart", "direction": "bullish",
                             "description": "MA50 crosses above MA200. Long-term bullish trend confirmation."},
    "Death Cross":          {"type": "chart", "direction": "bearish",
                             "description": "MA50 crosses below MA200. Long-term bearish trend confirmation."},
    "Breakout":             {"type": "chart", "direction": "bullish",
                             "description": "Price breaks above 20-day resistance. Bullish momentum signal."},
    "Breakdown":            {"type": "chart", "direction": "bearish",
                             "description": "Price breaks below 20-day support. Bearish momentum signal."},
    "Support Bounce":       {"type": "chart", "direction": "bullish",
                             "description": "Price tests 20-day low and bounces. Bullish continuation signal."},
    "Resistance Rejection": {"type": "chart", "direction": "bearish",
                             "description": "Price tests 20-day high and reverses. Bearish continuation signal."},
    # ---------- Behavioral patterns ----------
    "Banister":             {"type": "behavioral", "direction": "bullish",
                             "description": "Descending staircase of 3+ lower highs/lows (sellers grip the banister) then breaks above last swing high. Herd exhaustion → contrarian reversal entry."},
    "Capitulation Bottom":  {"type": "behavioral", "direction": "bullish",
                             "description": "Rapid >15% drop with volume spike, followed by a green reversal candle with long lower wick. Loss aversion selling exhausted → smart money accumulation."},
    "Wave 4 Pullback":      {"type": "behavioral", "direction": "bullish",
                             "description": "After a strong 3-wave advance (Elliott), a 8-15% pullback to MA50 with RSI resetting to 40-55. Anchoring bias creates a dip-buying opportunity."},
    "ABC Correction":       {"type": "behavioral", "direction": "bullish",
                             "description": "Three-wave corrective decline: A down → B up (partial recovery) → C down to new lows near A length. Recency bias (bears think trend continues) creates the C-wave reversal."},
    "Staircase Uptrend":    {"type": "behavioral", "direction": "bullish",
                             "description": "3+ ascending higher lows over 20+ bars. Price is at a new higher-low support zone. Herd momentum confirms continuation — buy the step."},
    "Herd Exhaustion":      {"type": "behavioral", "direction": "bearish",
                             "description": "RSI >75 + price above Bollinger upper band + 3+ consecutive up days with declining volume. FOMO buyers exhausted, distribution phase. Take profits / reduce."},
    "Dead Cat Bounce":      {"type": "behavioral", "direction": "bearish",
                             "description": "Weak recovery rally after a major sell-off: 3-7% bounce on declining volume, RSI stays below 50. Recency bias (bulls hope for recovery) — bounce will fail, trend continues down."},
    "FOMO Breakout":        {"type": "behavioral", "direction": "bullish",
                             "description": "Price breaks 20-day high with volume >1.5× average, RSI 50-70. Herd momentum pattern — crowd chases the breakout. Continuation trade with trail stop."},
    "V-Bottom Reversal":    {"type": "behavioral", "direction": "bullish",
                             "description": "Sharp flush down 5-15% intraday or over 2-3 days, RSI < 35, then strong bullish close or gap-up recovery. Panic selling exhausted in one move — best for leveraged ETFs and penny stocks."},
    "Distribution Top":     {"type": "behavioral", "direction": "bearish",
                             "description": "Topping pattern: 3+ days of high-volume selling near 20-day high, RSI divergence (price makes new high but RSI doesn't). Smart money distributing to retail buyers."},
}


def _body(row) -> float:
    return abs(row["close"] - row["open"])


def _range(row) -> float:
    return row["high"] - row["low"]


def _upper_shadow(row) -> float:
    return row["high"] - max(row["open"], row["close"])


def _lower_shadow(row) -> float:
    return min(row["open"], row["close"]) - row["low"]


def _is_green(row) -> bool:
    return row["close"] >= row["open"]


def _is_red(row) -> bool:
    return row["close"] < row["open"]


def detect_candlestick_patterns(df: pd.DataFrame) -> list[dict]:
    """
    Detect single and multi-candle patterns from OHLCV DataFrame.
    df must have columns: date, open, high, low, close
    Returns list of {date, pattern_name, direction, strength, price_level}
    """
    results = []
    n = len(df)
    if n < 3:
        return results

    rows = df.reset_index(drop=True)

    for i in range(2, n):
        c0 = rows.iloc[i - 2]   # 2 bars ago
        c1 = rows.iloc[i - 1]   # previous bar
        c2 = rows.iloc[i]       # current bar

        r2 = _range(c2)
        if r2 <= 0:
            continue

        b2 = _body(c2)
        us2 = _upper_shadow(c2)
        ls2 = _lower_shadow(c2)
        dt = c2["date"]
        mid_c2 = (c2["open"] + c2["close"]) / 2

        b1 = _body(c1)
        r1 = _range(c1)
        b0 = _body(c0)

        # ── Single-candle patterns ──────────────────────────────────────

        # Doji: body < 5% of range
        if b2 / r2 < 0.05:
            results.append({"date": dt, "pattern_name": "Doji", "direction": "neutral",
                             "strength": round(1 - b2 / r2, 2), "price_level": c2["close"]})

        # Hammer / Hanging Man: lower shadow >= 2× body, upper shadow < body, body in top 1/3
        elif (ls2 >= 2 * b2 and us2 < b2 and
              b2 > 0 and min(c2["open"], c2["close"]) > c2["low"] + r2 * 0.55):
            str_ = round(min(ls2 / r2, 1.0), 2)
            # Hammer = after downtrend (last 5 closes trending down), else Hanging Man
            trend_down = rows.iloc[max(0, i - 5):i]["close"].is_monotonic_decreasing if i >= 5 else False
            name = "Hammer" if trend_down else "Hanging Man"
            direction = "bullish" if name == "Hammer" else "bearish"
            results.append({"date": dt, "pattern_name": name, "direction": direction,
                             "strength": str_, "price_level": c2["low"]})

        # Inverted Hammer / Shooting Star: upper shadow >= 2× body, lower shadow < body
        elif (us2 >= 2 * b2 and ls2 < b2 and b2 > 0 and
              max(c2["open"], c2["close"]) < c2["high"] - r2 * 0.55):
            str_ = round(min(us2 / r2, 1.0), 2)
            trend_down = rows.iloc[max(0, i - 5):i]["close"].is_monotonic_decreasing if i >= 5 else False
            name = "Inverted Hammer" if trend_down else "Shooting Star"
            direction = "bullish" if name == "Inverted Hammer" else "bearish"
            results.append({"date": dt, "pattern_name": name, "direction": direction,
                             "strength": str_, "price_level": c2["high"]})

        # Bullish Marubozu: green, upper + lower shadow < 2% of range each
        if _is_green(c2) and us2 / r2 < 0.02 and ls2 / r2 < 0.02 and b2 / r2 > 0.90:
            results.append({"date": dt, "pattern_name": "Bullish Marubozu", "direction": "bullish",
                             "strength": round(b2 / r2, 2), "price_level": c2["close"]})

        # Bearish Marubozu: red, tiny wicks
        elif _is_red(c2) and us2 / r2 < 0.02 and ls2 / r2 < 0.02 and b2 / r2 > 0.90:
            results.append({"date": dt, "pattern_name": "Bearish Marubozu", "direction": "bearish",
                             "strength": round(b2 / r2, 2), "price_level": c2["close"]})

        # ── Two-candle patterns ─────────────────────────────────────────
        if i < 1 or r1 <= 0:
            continue

        # Bullish Engulfing: c1 red, c2 green, c2 body engulfs c1 body
        if (_is_red(c1) and _is_green(c2) and b1 > 0 and
                c2["open"] < c1["close"] and c2["close"] > c1["open"]):
            str_ = round(min(b2 / b1, 2.0) / 2, 2)
            results.append({"date": dt, "pattern_name": "Bullish Engulfing", "direction": "bullish",
                             "strength": str_, "price_level": c2["close"]})

        # Bearish Engulfing: c1 green, c2 red, c2 body engulfs c1 body
        elif (_is_green(c1) and _is_red(c2) and b1 > 0 and
              c2["open"] > c1["close"] and c2["close"] < c1["open"]):
            str_ = round(min(b2 / b1, 2.0) / 2, 2)
            results.append({"date": dt, "pattern_name": "Bearish Engulfing", "direction": "bearish",
                             "strength": str_, "price_level": c2["close"]})

        # Bullish Harami: c1 red (large), c2 green (small), c2 body inside c1 body
        elif (_is_red(c1) and _is_green(c2) and b1 > 0 and b2 < b1 * 0.5 and
              c2["open"] > c1["close"] and c2["close"] < c1["open"]):
            results.append({"date": dt, "pattern_name": "Bullish Harami", "direction": "bullish",
                             "strength": round(1 - b2 / b1, 2), "price_level": c2["close"]})

        # Bearish Harami: c1 green (large), c2 red (small), c2 body inside c1 body
        elif (_is_green(c1) and _is_red(c2) and b1 > 0 and b2 < b1 * 0.5 and
              c2["open"] < c1["close"] and c2["close"] > c1["open"]):
            results.append({"date": dt, "pattern_name": "Bearish Harami", "direction": "bearish",
                             "strength": round(1 - b2 / b1, 2), "price_level": c2["close"]})

        # Piercing Line: c1 red, c2 green, c2 opens below c1 low, closes above c1 midpoint
        elif (_is_red(c1) and _is_green(c2) and b1 > 0 and
              c2["open"] < c1["low"] and c2["close"] > (c1["open"] + c1["close"]) / 2):
            results.append({"date": dt, "pattern_name": "Piercing Line", "direction": "bullish",
                             "strength": 0.75, "price_level": c2["close"]})

        # Dark Cloud Cover: c1 green, c2 red, c2 opens above c1 high, closes below c1 midpoint
        elif (_is_green(c1) and _is_red(c2) and b1 > 0 and
              c2["open"] > c1["high"] and c2["close"] < (c1["open"] + c1["close"]) / 2):
            results.append({"date": dt, "pattern_name": "Dark Cloud Cover", "direction": "bearish",
                             "strength": 0.75, "price_level": c2["close"]})

        # ── Three-candle patterns ───────────────────────────────────────
        if i < 2 or b0 <= 0:
            continue

        # Morning Star: c0 large red, c1 small body (doji-like), c2 large green
        if (_is_red(c0) and b0 > 0 and
                b1 < b0 * 0.3 and                            # small middle
                _is_green(c2) and b2 > b0 * 0.5 and          # large green
                c2["close"] > c0["open"] * 0.97):             # closes into c0 body
            results.append({"date": dt, "pattern_name": "Morning Star", "direction": "bullish",
                             "strength": 0.85, "price_level": c2["close"]})

        # Evening Star: c0 large green, c1 small body, c2 large red
        elif (_is_green(c0) and b0 > 0 and
              b1 < b0 * 0.3 and
              _is_red(c2) and b2 > b0 * 0.5 and
              c2["close"] < c0["open"] * 1.03):
            results.append({"date": dt, "pattern_name": "Evening Star", "direction": "bearish",
                             "strength": 0.85, "price_level": c2["close"]})

        # Three White Soldiers: 3 consecutive green candles, each closing higher
        elif (_is_green(c0) and _is_green(c1) and _is_green(c2) and
              c1["close"] > c0["close"] and c2["close"] > c1["close"] and
              b0 > 0 and b1 > 0 and b2 > 0 and
              b0 / _range(c0) > 0.5 and b1 / _range(c1) > 0.5 and b2 / r2 > 0.5):
            results.append({"date": dt, "pattern_name": "Three White Soldiers", "direction": "bullish",
                             "strength": 0.90, "price_level": c2["close"]})

        # Three Black Crows: 3 consecutive red candles, each closing lower
        elif (_is_red(c0) and _is_red(c1) and _is_red(c2) and
              c1["close"] < c0["close"] and c2["close"] < c1["close"] and
              b0 > 0 and b1 > 0 and b2 > 0 and
              b0 / _range(c0) > 0.5 and b1 / _range(c1) > 0.5 and b2 / r2 > 0.5):
            results.append({"date": dt, "pattern_name": "Three Black Crows", "direction": "bearish",
                             "strength": 0.90, "price_level": c2["close"]})

    return results


def detect_chart_patterns(df: pd.DataFrame, ind_df: pd.DataFrame) -> list[dict]:
    """
    Detect multi-day chart patterns.
    df: OHLCV DataFrame (date, open, high, low, close)
    ind_df: Indicators DataFrame (date, ma50, ma200)
    Returns list of pattern dicts.
    """
    results = []
    if len(df) < 30:
        return results

    df = df.copy().reset_index(drop=True)

    # ── Golden Cross / Death Cross (MA50 vs MA200) ─────────────────────
    if len(ind_df) >= 2:
        ind = ind_df.dropna(subset=["ma50", "ma200"]).reset_index(drop=True)
        for j in range(1, len(ind)):
            prev = ind.iloc[j - 1]
            curr = ind.iloc[j]
            dt = curr["date"]
            # Golden Cross: MA50 crosses above MA200
            if prev["ma50"] <= prev["ma200"] and curr["ma50"] > curr["ma200"]:
                close_val = df[df["date"] == dt]["close"].values
                price = float(close_val[0]) if len(close_val) else curr["ma50"]
                results.append({"date": dt, "pattern_name": "Golden Cross", "direction": "bullish",
                                 "strength": 0.90, "price_level": round(price, 2)})
            # Death Cross: MA50 crosses below MA200
            elif prev["ma50"] >= prev["ma200"] and curr["ma50"] < curr["ma200"]:
                close_val = df[df["date"] == dt]["close"].values
                price = float(close_val[0]) if len(close_val) else curr["ma50"]
                results.append({"date": dt, "pattern_name": "Death Cross", "direction": "bearish",
                                 "strength": 0.90, "price_level": round(price, 2)})

    # ── Breakout / Breakdown / Bounce / Rejection (20-day rolling) ─────
    window = 20
    tol = 0.015  # 1.5% tolerance for "near" support/resistance

    for i in range(window, len(df)):
        curr = df.iloc[i]
        window_slice = df.iloc[i - window:i]
        resistance = window_slice["high"].max()
        support = window_slice["low"].min()
        dt = curr["date"]
        close = curr["close"]

        # Breakout: close > 20-day high (new high)
        if close > resistance * (1 + tol):
            results.append({"date": dt, "pattern_name": "Breakout", "direction": "bullish",
                             "strength": round(min((close - resistance) / resistance * 20, 1.0), 2),
                             "price_level": round(resistance, 2)})

        # Breakdown: close < 20-day low (new low)
        elif close < support * (1 - tol):
            results.append({"date": dt, "pattern_name": "Breakdown", "direction": "bearish",
                             "strength": round(min((support - close) / support * 20, 1.0), 2),
                             "price_level": round(support, 2)})

        # Support Bounce: low touched near support but close is up ≥ 0.5% from low
        elif (abs(curr["low"] - support) / support < tol and
              curr["close"] > curr["low"] * 1.005 and curr["close"] >= curr["open"]):
            results.append({"date": dt, "pattern_name": "Support Bounce", "direction": "bullish",
                             "strength": 0.65, "price_level": round(support, 2)})

        # Resistance Rejection: high touched near resistance but close is down ≥ 0.5% from high
        elif (abs(curr["high"] - resistance) / resistance < tol and
              curr["close"] < curr["high"] * 0.995 and curr["close"] <= curr["open"]):
            results.append({"date": dt, "pattern_name": "Resistance Rejection", "direction": "bearish",
                             "strength": 0.65, "price_level": round(resistance, 2)})

    # ── Double Bottom ──────────────────────────────────────────────────
    # Look for two similar lows separated by 10-40 bars, with an intermediate high
    for i in range(40, len(df)):
        curr_low = df.iloc[i]["low"]
        curr_date = df.iloc[i]["date"]
        # Search for a prior low within ±3% of current low, 10-40 bars ago
        for lookback in range(10, 41):
            if i - lookback < 0:
                break
            prior = df.iloc[i - lookback]
            if abs(prior["low"] - curr_low) / curr_low < 0.03:
                # Check there's a peak between them higher than both lows + 3%
                between = df.iloc[i - lookback + 1:i]
                if len(between) == 0:
                    continue
                peak = between["high"].max()
                if peak > curr_low * 1.03:
                    # Confirm current candle is green (bounce)
                    if df.iloc[i]["close"] >= df.iloc[i]["open"]:
                        results.append({
                            "date": curr_date,
                            "pattern_name": "Double Bottom",
                            "direction": "bullish",
                            "strength": round(1 - abs(prior["low"] - curr_low) / curr_low, 2),
                            "price_level": round(curr_low, 2),
                        })
                        break  # only record once per bar

    # ── Double Top ─────────────────────────────────────────────────────
    for i in range(40, len(df)):
        curr_high = df.iloc[i]["high"]
        curr_date = df.iloc[i]["date"]
        for lookback in range(10, 41):
            if i - lookback < 0:
                break
            prior = df.iloc[i - lookback]
            if abs(prior["high"] - curr_high) / curr_high < 0.03:
                between = df.iloc[i - lookback + 1:i]
                if len(between) == 0:
                    continue
                trough = between["low"].min()
                if trough < curr_high * 0.97:
                    if df.iloc[i]["close"] <= df.iloc[i]["open"]:
                        results.append({
                            "date": curr_date,
                            "pattern_name": "Double Top",
                            "direction": "bearish",
                            "strength": round(1 - abs(prior["high"] - curr_high) / curr_high, 2),
                            "price_level": round(curr_high, 2),
                        })
                        break

    return results


def _swing_highs_lows(df: pd.DataFrame, window: int = 3) -> tuple[list, list]:
    """Find swing highs and lows using a symmetric window."""
    highs, lows = [], []
    n = len(df)
    for i in range(window, n - window):
        hi = df.iloc[i]["high"]
        lo = df.iloc[i]["low"]
        dt = df.iloc[i]["date"]
        if all(hi >= df.iloc[i - k]["high"] for k in range(1, window + 1)) and \
           all(hi >= df.iloc[i + k]["high"] for k in range(1, window + 1)):
            highs.append((i, hi, dt))
        if all(lo <= df.iloc[i - k]["low"] for k in range(1, window + 1)) and \
           all(lo <= df.iloc[i + k]["low"] for k in range(1, window + 1)):
            lows.append((i, lo, dt))
    return highs, lows


def detect_behavioral_patterns(df: pd.DataFrame, ind_df: pd.DataFrame) -> list[dict]:
    """
    Detect behavioral crowd-psychology patterns.
    Based on: Herd Mentality, Loss Aversion, Anchoring, Recency Bias extremes.
    """
    results = []
    if len(df) < 35:
        return results

    df = df.copy().reset_index(drop=True)

    # Build fast lookup maps for indicators
    rsi_map    = {}
    vol_map    = {}
    bb_up_map  = {}
    bb_lo_map  = {}
    ma50_map   = {}
    if ind_df is not None and len(ind_df):
        for _, row in ind_df.iterrows():
            d = row["date"]
            rsi_map[d]   = row.get("rsi")
            bb_up_map[d] = row.get("bb_upper")
            bb_lo_map[d] = row.get("bb_lower")
            ma50_map[d]  = row.get("ma50")

    # 20-day average volume
    df["avg_vol_20"] = df["volume"].rolling(20).mean()

    swing_highs, swing_lows = _swing_highs_lows(df, window=3)

    for i in range(35, len(df)):
        curr   = df.iloc[i]
        dt     = curr["date"]
        close  = curr["close"]
        open_  = curr["open"]
        vol    = curr["volume"]
        avg_v  = curr["avg_vol_20"] if not pd.isna(curr["avg_vol_20"]) else vol
        rsi    = rsi_map.get(dt)
        ma50   = ma50_map.get(dt)
        bb_up  = bb_up_map.get(dt)
        bb_lo  = bb_lo_map.get(dt)

        # ── 1. BANISTER (Descending Staircase Reversal) ───────────────────
        # 3+ descending swing lows AND swing highs → price breaks above last swing high
        recent_sh = [sh for sh in swing_highs if i - 35 <= sh[0] < i]
        recent_sl = [sl for sl in swing_lows  if i - 35 <= sl[0] < i]
        if len(recent_sh) >= 3 and len(recent_sl) >= 3:
            sh_vals = [sh[1] for sh in recent_sh[-4:]]
            sl_vals = [sl[1] for sl in recent_sl[-4:]]
            desc_highs = all(sh_vals[k] < sh_vals[k-1] for k in range(1, len(sh_vals)))
            desc_lows  = all(sl_vals[k] < sl_vals[k-1] for k in range(1, len(sl_vals)))
            if desc_highs and desc_lows:
                # RSI was oversold at some point during the descent
                rsi_in_window = [rsi_map.get(df.iloc[j]["date"]) for j in range(i - 35, i)]
                rsi_in_window = [r for r in rsi_in_window if r is not None]
                min_rsi = min(rsi_in_window) if rsi_in_window else 50
                if min_rsi < 42:
                    last_sh = recent_sh[-1][1]   # last swing high = banister rail
                    if close > last_sh and close >= open_:
                        total_decline = (sh_vals[0] - sl_vals[-1]) / sh_vals[0]
                        strength = round(min(0.55 + total_decline * 2.5, 1.0), 2)
                        results.append({
                            "date": dt, "pattern_name": "Banister",
                            "direction": "bullish", "strength": strength,
                            "price_level": round(last_sh, 2),
                        })

        # ── 2. CAPITULATION BOTTOM ────────────────────────────────────────
        # >15% drop in 10 bars + volume spike + green reversal candle
        if i >= 10:
            hi_10  = df.iloc[i - 10:i]["close"].max()
            drop   = (hi_10 - close) / hi_10 if hi_10 > 0 else 0
            lo_wick = min(open_, close) - curr["low"]
            body    = abs(close - open_)
            if (drop > 0.15 and vol > avg_v * 1.8 and
                    close >= open_ and lo_wick > body * 0.5):
                strength = round(min(0.5 + drop * 2.5, 1.0), 2)
                results.append({
                    "date": dt, "pattern_name": "Capitulation Bottom",
                    "direction": "bullish", "strength": strength,
                    "price_level": round(curr["low"], 2),
                })

        # ── 3. WAVE 4 PULLBACK ────────────────────────────────────────────
        # Strong 20-bar advance (+15%+) → 8-15% pullback to/near MA50 → RSI reset
        if i >= 20 and ma50 is not None and rsi is not None:
            base_close  = df.iloc[i - 20]["close"]
            peak_close  = df.iloc[i - 20:i]["close"].max()
            advance_pct = (peak_close - base_close) / base_close if base_close > 0 else 0
            pullback_pct = (peak_close - close) / peak_close if peak_close > 0 else 0
            near_ma50   = abs(close - ma50) / ma50 < 0.04
            if (advance_pct > 0.15 and 0.07 < pullback_pct < 0.20 and
                    near_ma50 and 38 < rsi < 58 and close >= open_):
                strength = round(0.60 + advance_pct * 0.8, 2)
                strength = min(strength, 0.95)
                results.append({
                    "date": dt, "pattern_name": "Wave 4 Pullback",
                    "direction": "bullish", "strength": strength,
                    "price_level": round(ma50, 2),
                })

        # ── 4. ABC CORRECTION ─────────────────────────────────────────────
        # A down → B up (38-62% retrace of A) → C down ≈ A in length
        if len(recent_sh) >= 2 and len(recent_sl) >= 2 and rsi is not None:
            # A: peak → first swing low after peak
            peak_sh = recent_sh[-2]  # (idx, high, date)
            b_sl    = recent_sl[-2]  # swing low after peak = bottom of A
            b_sh    = recent_sh[-1]  # recovery = top of B
            c_sl    = recent_sl[-1]  # current bottom = C
            a_len   = peak_sh[1] - b_sl[1]
            b_retrace = (b_sh[1] - b_sl[1]) / a_len if a_len > 0 else 0
            c_len   = b_sh[1] - c_sl[1]
            # C should be roughly equal to A (80-130% of A)
            c_ratio = c_len / a_len if a_len > 0 else 0
            # Order: peak_sh → b_sl → b_sh → c_sl (sequential)
            in_order = (peak_sh[0] < b_sl[0] < b_sh[0] < c_sl[0])
            if (in_order and 0.35 < b_retrace < 0.68 and
                    0.75 < c_ratio < 1.35 and rsi < 40 and
                    close >= open_ and c_sl[0] >= i - 5):
                results.append({
                    "date": dt, "pattern_name": "ABC Correction",
                    "direction": "bullish",
                    "strength": round(min(0.55 + abs(c_ratio - 1.0) * -0.3 + 0.3, 0.92), 2),
                    "price_level": round(c_sl[1], 2),
                })

        # ── 5. STAIRCASE UPTREND (higher lows) ───────────────────────────
        # 3+ ascending swing lows; currently at a new swing low = buy the step
        if len(recent_sl) >= 3:
            sl_vals_asc = [sl[1] for sl in recent_sl[-4:]]
            asc_lows = all(sl_vals_asc[k] > sl_vals_asc[k-1] for k in range(1, len(sl_vals_asc)))
            last_sl_idx = recent_sl[-1][0]
            at_step = (i - last_sl_idx) <= 5   # within 5 bars of the last swing low
            if asc_lows and at_step and close >= open_:
                strength = round(min(0.55 + len(recent_sl) * 0.05, 0.88), 2)
                results.append({
                    "date": dt, "pattern_name": "Staircase Uptrend",
                    "direction": "bullish", "strength": strength,
                    "price_level": round(recent_sl[-1][1], 2),
                })

        # ── 6. HERD EXHAUSTION (FOMO Top) ────────────────────────────────
        # RSI>75 + price above BB upper + 3 consecutive up days with declining volume
        if rsi is not None and bb_up is not None and rsi > 74 and close > bb_up:
            if i >= 3:
                last3 = df.iloc[i - 3:i + 1]
                consecutive_up  = all(last3.iloc[k]["close"] > last3.iloc[k - 1]["close"]
                                      for k in range(1, 4))
                volume_declining = (last3.iloc[3]["volume"] < last3.iloc[2]["volume"] <
                                    last3.iloc[1]["volume"])
                if consecutive_up and volume_declining:
                    strength = round(min((rsi - 74) / 26 + 0.55, 0.92), 2)
                    results.append({
                        "date": dt, "pattern_name": "Herd Exhaustion",
                        "direction": "bearish", "strength": strength,
                        "price_level": round(bb_up, 2),
                    })

        # ── 7. DEAD CAT BOUNCE ────────────────────────────────────────────
        # Major drop (>15% in 15 bars) → weak 3-7% bounce on declining volume → RSI < 48
        if i >= 15 and rsi is not None and rsi < 48:
            hi_15  = df.iloc[i - 15:i - 3]["close"].max()
            lo_3   = df.iloc[i - 3:i]["close"].min()
            drop_15 = (hi_15 - lo_3) / hi_15 if hi_15 > 0 else 0
            bounce  = (close - lo_3) / lo_3 if lo_3 > 0 else 0
            vol_bounce  = df.iloc[i - 3:i + 1]["volume"].mean()
            vol_selloff = df.iloc[i - 15:i - 3]["volume"].mean()
            if (drop_15 > 0.15 and 0.03 < bounce < 0.09 and
                    vol_bounce < vol_selloff * 0.85 and close < open_):
                results.append({
                    "date": dt, "pattern_name": "Dead Cat Bounce",
                    "direction": "bearish",
                    "strength": round(min(0.5 + drop_15 * 2, 0.90), 2),
                    "price_level": round(close, 2),
                })

        # ── 8. FOMO BREAKOUT ─────────────────────────────────────────────
        # New 20-day high + volume > 1.5× avg + RSI 50-70 (healthy, not extreme)
        if i >= 20 and rsi is not None and 50 < rsi < 72:
            resistance_20 = df.iloc[i - 20:i]["high"].max()
            ret_5d = (close - df.iloc[i - 5]["close"]) / df.iloc[i - 5]["close"] if i >= 5 else 0
            if (close > resistance_20 * 1.005 and
                    vol > avg_v * 1.45 and
                    ret_5d < 0.15 and  # not already parabolic
                    close >= open_):
                strength = round(min(0.55 + (vol / avg_v - 1.45) * 0.2, 0.92), 2)
                results.append({
                    "date": dt, "pattern_name": "FOMO Breakout",
                    "direction": "bullish", "strength": strength,
                    "price_level": round(resistance_20, 2),
                })

        # ── 9. V-BOTTOM REVERSAL ─────────────────────────────────────────
        # Sharp flush 5-15% over 1-3 days → strong bullish recovery candle
        # Captures single-day capitulation spikes common in leveraged ETFs
        if i >= 3 and rsi is not None and rsi < 38:
            low_3d  = df.iloc[i - 3:i + 1]["low"].min()
            high_3d = df.iloc[i - 3:i + 1]["high"].max()
            flush   = (high_3d - low_3d) / high_3d if high_3d > 0 else 0
            ref_close = df.iloc[i - 3]["close"]
            drawdown = (ref_close - close) / ref_close if ref_close > 0 else 0

            # Today's candle must be bullish with body > 50% of range
            candle_range = high - low
            body = abs(close - open_)
            bullish_close = close > open_ and (candle_range > 0 and body / candle_range > 0.45)

            if (flush >= 0.05 and drawdown >= 0.04 and
                    bullish_close and vol >= avg_v * 1.2):
                strength = round(min(0.55 + flush * 3, 0.95), 2)
                results.append({
                    "date": dt, "pattern_name": "V-Bottom Reversal",
                    "direction": "bullish", "strength": strength,
                    "price_level": round(low_3d, 2),
                })

        # ── 10. DISTRIBUTION TOP ─────────────────────────────────────────
        # 3+ high-volume bars near 20-day high + RSI divergence (not confirming)
        if i >= 22 and rsi is not None and rsi > 58:
            hi_20 = df.iloc[i - 20:i + 1]["high"].max()
            # Price is near the 20-day high
            near_high = close >= hi_20 * 0.97

            if near_high:
                # Count high-volume days in last 5 bars where close < open (selling)
                recent = df.iloc[i - 4:i + 1]
                hv_selling = sum(
                    1 for _, r in recent.iterrows()
                    if r["volume"] >= avg_v * 1.2 and r["close"] < r["open"]
                )
                # RSI should be flat or declining while price is at highs
                rsi_3d_ago = ind_df["rsi"].iloc[i - 3] if i < len(ind_df) else None
                rsi_diverging = rsi_3d_ago is not None and rsi <= rsi_3d_ago

                if hv_selling >= 2 and rsi_diverging and vol >= avg_v * 1.1:
                    strength = round(min(0.52 + hv_selling * 0.1, 0.88), 2)
                    results.append({
                        "date": dt, "pattern_name": "Distribution Top",
                        "direction": "bearish", "strength": strength,
                        "price_level": round(hi_20, 2),
                    })

    return results


class PatternDetector:
    """Detects and stores all patterns for instruments."""

    def detect_for_symbol(self, symbol: str) -> list[dict]:
        session = get_session()
        try:
            instrument_id = session.execute(
                select(Instrument.id).where(Instrument.symbol == symbol)
            ).scalar()
            if instrument_id is None:
                return []

            price_df = pd.read_sql(
                select(PriceData.date, PriceData.open, PriceData.high,
                       PriceData.low, PriceData.close, PriceData.volume)
                .where(PriceData.instrument_id == instrument_id)
                .order_by(PriceData.date),
                session.bind,
            )
            if price_df.empty or len(price_df) < 10:
                return []
            price_df["date"] = pd.to_datetime(price_df["date"]).dt.date

            ind_df = pd.read_sql(
                select(Indicator.date, Indicator.ma50, Indicator.ma200,
                       Indicator.rsi, Indicator.bb_upper, Indicator.bb_lower)
                .where(Indicator.instrument_id == instrument_id)
                .order_by(Indicator.date),
                session.bind,
            )
            ind_df["date"] = pd.to_datetime(ind_df["date"]).dt.date

            cs_patterns         = detect_candlestick_patterns(price_df)
            chart_patterns      = detect_chart_patterns(price_df, ind_df)
            behavioral_patterns = detect_behavioral_patterns(price_df, ind_df)

            all_patterns = cs_patterns + chart_patterns + behavioral_patterns
            for p in all_patterns:
                p["instrument_id"] = instrument_id
                p["pattern_type"] = PATTERN_CATALOGUE.get(p["pattern_name"], {}).get("type", "candlestick")
                # Coerce numpy scalars → native Python floats for PostgreSQL
                p["strength"] = float(p["strength"]) if p["strength"] is not None else None
                p["price_level"] = float(p["price_level"]) if p["price_level"] is not None else None

            return all_patterns
        finally:
            session.close()

    def compute_and_store(self, symbol: str | None = None):
        """Detect and store patterns for one or all instruments."""
        session = get_session()
        try:
            if symbol:
                instruments = session.execute(
                    select(Instrument).where(Instrument.symbol == symbol)
                ).scalars().all()
            else:
                instruments = session.execute(select(Instrument)).scalars().all()

            total = 0
            for inst in tqdm(instruments, desc="Detecting patterns"):
                patterns = self.detect_for_symbol(inst.symbol)
                if patterns:
                    self._store(session, patterns)
                    total += len(patterns)

            logger.info(f"Stored {total} patterns across all instruments")
            return total
        finally:
            session.close()

    def _store(self, session, patterns: list[dict]):
        for i in range(0, len(patterns), 500):
            batch = patterns[i:i + 500]
            session.execute(
                pg_insert(PatternDetection).values(batch).on_conflict_do_nothing()
            )
        session.commit()

    def get_patterns_for_symbol(self, symbol: str, limit: int = 500) -> list[dict]:
        """Return most recent patterns for a symbol."""
        session = get_session()
        try:
            instrument_id = session.execute(
                select(Instrument.id).where(Instrument.symbol == symbol)
            ).scalar()
            if not instrument_id:
                return []

            rows = session.execute(
                select(PatternDetection)
                .where(PatternDetection.instrument_id == instrument_id)
                .order_by(PatternDetection.date.desc())
                .limit(limit)
            ).scalars().all()

            return [
                {
                    "date":         str(r.date),
                    "pattern_name": r.pattern_name,
                    "pattern_type": r.pattern_type,
                    "direction":    r.direction,
                    "strength":     r.strength,
                    "price_level":  r.price_level,
                }
                for r in rows
            ]
        finally:
            session.close()
