"""
Candlestick pattern detector — TA-Lib powered (61 patterns).

Uses compiled C-library TA-Lib for rigorous, battle-tested candlestick
detection. Falls back to a minimal pure-Python set if TA-Lib is unavailable.

TA-Lib CDL functions return:
  +100  → bullish signal
  -100  → bearish signal
     0  → no pattern on that bar

Age decay:
  age=0 (current bar)  → 100% confidence
  age=1 (prior close)  → 60%  (pattern hasn't resolved, edge drops sharply)
  age=2                → 25%  (below 65% floor in service.py → filtered out)
  age≥3                → expired entirely (FAILED status)
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── TA-Lib CDL catalogue ──────────────────────────────────────────────────────
# (talib_func_name, friendly_name, base_confidence)
# base_confidence reflects pattern rarity and historical reliability.
# Multi-candle patterns score highest; single-candle scores are lower.
_CDL_CATALOGUE: list[tuple[str, str, float]] = [
    # ── Single-candle ────────────────────────────────────────────────────────
    ("CDLDOJI",             "Doji",                                    68.0),
    ("CDLDRAGONFLYDOJI",    "Dragonfly Doji",                          74.0),
    ("CDLGRAVESTONEDOJI",   "Gravestone Doji",                         74.0),
    ("CDLLONGLEGGEDDOJI",   "Long-Legged Doji",                        70.0),
    ("CDLRICKSHAWMAN",      "Rickshaw Man",                            68.0),
    ("CDLHAMMER",           "Hammer",                                  76.0),
    ("CDLHANGINGMAN",       "Hanging Man",                             75.0),
    ("CDLINVERTEDHAMMER",   "Inverted Hammer",                         73.0),
    ("CDLSHOOTINGSTAR",     "Shooting Star",                           76.0),
    ("CDLMARUBOZU",         "Marubozu",                                78.0),
    ("CDLCLOSINGMARUBOZU",  "Closing Marubozu",                        77.0),
    ("CDLHIGHWAVE",         "High Wave",                               67.0),
    ("CDLSPINNINGTOP",      "Spinning Top",                            65.0),
    ("CDLSHORTLINE",        "Short Line",                              63.0),
    ("CDLLONGLINE",         "Long Line",                               68.0),
    ("CDLBELTHOLD",         "Belt Hold",                               74.0),
    ("CDLDOJISTAR",         "Doji Star",                               73.0),
    ("CDLTAKURI",           "Takuri",                                  75.0),
    # ── Two-candle ───────────────────────────────────────────────────────────
    ("CDLENGULFING",        "Engulfing",                               82.0),
    ("CDLHARAMI",           "Harami",                                  75.0),
    ("CDLHARAMICROSS",      "Harami Cross",                            76.0),
    ("CDLPIERCING",         "Piercing Line",                           78.0),
    ("CDLDARKCLOUDCOVER",   "Dark Cloud Cover",                        78.0),
    ("CDLKICKING",          "Kicking",                                 83.0),
    ("CDLKICKINGBYLENGTH",  "Kicking By Length",                       82.0),
    ("CDLCOUNTERATTACK",    "Counterattack",                           75.0),
    ("CDLINNECK",           "In-Neck Pattern",                         70.0),
    ("CDLONNECK",           "On-Neck Pattern",                         70.0),
    ("CDLTHRUSTING",        "Thrusting Pattern",                       71.0),
    ("CDLGAPSIDESIDEWHITE", "Gap Side-by-Side White Lines",            73.0),
    ("CDLMATCHINGLOW",      "Matching Low",                            74.0),
    ("CDLHOMINGPIGEON",     "Homing Pigeon",                           73.0),
    ("CDLSEPARATINGLINES",  "Separating Lines",                        75.0),
    # ── Three-candle ─────────────────────────────────────────────────────────
    ("CDLMORNINGSTAR",      "Morning Star",                            86.0),
    ("CDLEVENINGSTAR",      "Evening Star",                            86.0),
    ("CDLMORNINGDOJISTAR",  "Morning Doji Star",                       88.0),
    ("CDLEVENINGDOJISTAR",  "Evening Doji Star",                       88.0),
    ("CDL3WHITESOLDIERS",   "Three White Soldiers",                    89.0),
    ("CDL3BLACKCROWS",      "Three Black Crows",                       89.0),
    ("CDL3INSIDE",          "Three Inside Up/Down",                    80.0),
    ("CDL3OUTSIDE",         "Three Outside Up/Down",                   81.0),
    ("CDL3LINESTRIKE",      "Three-Line Strike",                       79.0),
    ("CDLABANDONEDBABY",    "Abandoned Baby",                          91.0),
    ("CDLBREAKAWAY",        "Breakaway",                               80.0),
    ("CDLADVANCEBLOCK",     "Advance Block",                           78.0),
    ("CDLSTALLEDPATTERN",   "Stalled Pattern",                         77.0),
    ("CDLTRISTAR",          "Tristar Pattern",                         82.0),
    ("CDLIDENTICAL3CROWS",  "Identical Three Crows",                   87.0),
    ("CDL2CROWS",           "Two Crows",                               80.0),
    ("CDLUPSIDEGAP2CROWS",  "Upside Gap Two Crows",                    81.0),
    ("CDLXSIDEGAP3METHODS", "Up/Down Gap Three Methods",               82.0),
    # ── Multi-candle / rare ──────────────────────────────────────────────────
    ("CDL3STARSINSOUTH",    "Three Stars In The South",                87.0),
    ("CDLCONCEALBABYSWALL", "Concealing Baby Swallow",                 86.0),
    ("CDLHIKKAKE",          "Hikkake Pattern",                         78.0),
    ("CDLHIKKAKEMOD",       "Modified Hikkake Pattern",                79.0),
    ("CDLLADDERBOTTOM",     "Ladder Bottom",                           85.0),
    ("CDLMATHOLD",          "Mat Hold",                                84.0),
    ("CDLRISEFALL3METHODS", "Rising/Falling Three Methods",            83.0),
    ("CDLUNIQUE3RIVER",     "Unique Three River",                      82.0),
    ("CDLSTICKSANDWICH",    "Stick Sandwich",                          79.0),
    ("CDLTASUKIGAP",        "Tasuki Gap",                              78.0),
]


def _atr(df: pd.DataFrame, n: int = 14) -> float:
    if len(df) < 2:
        return float(df['close'].iloc[-1]) * 0.02
    hi = df['high'].astype(float).tail(n)
    lo = df['low'].astype(float).tail(n)
    return float((hi - lo).mean()) or float(df['close'].iloc[-1]) * 0.02


def _status_from_age(age: int) -> str:
    if age == 0:
        return "FORMING"
    if age <= 2:
        return "READY"
    return "FAILED"


def detect(symbol: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect up to 61 candlestick patterns using TA-Lib.

    Scans only the last 5 bars — candlestick signals are time-sensitive.
    Returns one entry per pattern found; older bars get confidence-decay applied.
    TA-Lib requires the full series for its lookback-period calibration.
    """
    if len(df) < 10:
        return []

    try:
        import talib  # noqa: PLC0415
    except ImportError:
        logger.warning("TA-Lib not available — falling back to minimal detector")
        return _fallback_detect(symbol, df)

    df = df.reset_index(drop=True)
    n = len(df)
    atr = _atr(df)

    # TA-Lib needs float64 numpy arrays
    op  = df['open'].astype(float).to_numpy()
    hi  = df['high'].astype(float).to_numpy()
    lo  = df['low'].astype(float).to_numpy()
    cl  = df['close'].astype(float).to_numpy()

    # Scan window: last 5 bars only
    scan_start = max(0, n - 5)

    # pattern_name → (bar_idx, direction, base_conf)
    found: dict[str, tuple[int, str, float]] = {}

    for func_name, friendly_name, base_conf in _CDL_CATALOGUE:
        fn = getattr(talib, func_name, None)
        if fn is None:
            continue
        try:
            result = fn(op, hi, lo, cl)
        except Exception:
            continue

        # Find the most recent signal within our 5-bar window
        for i in range(n - 1, scan_start - 1, -1):
            val = result[i]
            if val == 100:
                direction = "bullish"
            elif val == -100:
                direction = "bearish"
            else:
                continue

            # Keep only first (most recent) occurrence per pattern
            found.setdefault(friendly_name, (i, direction, base_conf))
            break

    # Build output records
    results: list[dict[str, Any]] = []
    for pattern_name, (bar_idx, direction, base_conf) in found.items():
        age = n - 1 - bar_idx
        status = _status_from_age(age)
        if status == "FAILED":
            continue  # expired, skip entirely

        row   = df.iloc[bar_idx]
        price = float(row['close'])
        high  = float(row['high'])
        low   = float(row['low'])

        if direction == "bullish":
            breakout     = round(high  * 1.002, 4)
            invalidation = round(low   - atr,   4)
            target       = round(price + 2.0 * atr, 4)
        elif direction == "bearish":
            breakout     = round(low   * 0.998, 4)
            invalidation = round(high  + atr,   4)
            target       = round(price - 2.0 * atr, 4)
        else:
            breakout     = price
            invalidation = price - atr
            target       = price + atr

        # Age decay — confidence drops sharply after the signal bar closes
        age_mult   = 1.0 if age == 0 else (0.60 if age == 1 else 0.25)
        final_conf = round(base_conf * age_mult, 1)

        results.append({
            "pattern_name":      pattern_name,
            "pattern_category":  "candlestick",
            "status":            status,
            "direction":         direction,
            "confidence":        final_conf,
            "breakout_level":    breakout,
            "invalidation_level": invalidation,
            "target":            target,
            "points":            [[bar_idx, price]],
        })

    return results


# ── Fallback (no TA-Lib) ─────────────────────────────────────────────────────

def _fallback_detect(symbol: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    """Minimal 6-pattern fallback when TA-Lib is unavailable."""
    def body(o, c): return abs(c - o)
    def rng(h, l):  return h - l
    def ushadow(h, o, c): return h - max(o, c)
    def lshadow(o, c, l): return min(o, c) - l

    df   = df.reset_index(drop=True)
    n    = len(df)
    atr  = _atr(df)
    found: dict[str, tuple[int, str, float]] = {}

    for i in range(max(2, n - 5), n):
        c2 = df.iloc[i]
        c1 = df.iloc[i - 1]
        o2, h2, l2, cl2 = float(c2['open']), float(c2['high']), float(c2['low']), float(c2['close'])
        o1, h1, l1, cl1 = float(c1['open']), float(c1['high']), float(c1['low']), float(c1['close'])
        b2, r2, us2, ls2 = body(o2, cl2), rng(h2, l2), ushadow(h2, o2, cl2), lshadow(o2, cl2, l2)
        b1, r1 = body(o1, cl1), rng(h1, l1)
        if r2 <= 0 or r1 <= 0: continue
        if b2/r2 < 0.05:
            found.setdefault("Doji", (i, "neutral", 68.0))
        if ls2 >= 2*b2 and us2 < b2 and b2 > 0:
            found.setdefault("Hammer", (i, "bullish", 76.0))
        if us2 >= 2*b2 and ls2 < b2 and b2 > 0:
            found.setdefault("Shooting Star", (i, "bearish", 76.0))
        if cl1 < o1 and cl2 >= o2 and o2 < cl1 and cl2 > o1:
            found.setdefault("Bullish Engulfing", (i, "bullish", 82.0))
        if cl1 >= o1 and cl2 < o2 and o2 > cl1 and cl2 < o1:
            found.setdefault("Bearish Engulfing", (i, "bearish", 82.0))

    results = []
    for name, (idx, direction, base_conf) in found.items():
        age  = n - 1 - idx
        if age >= 3: continue
        row  = df.iloc[idx]
        price, high, low = float(row['close']), float(row['high']), float(row['low'])
        age_mult   = 1.0 if age == 0 else (0.60 if age == 1 else 0.25)
        results.append({
            "pattern_name":      name,
            "pattern_category":  "candlestick",
            "status":            _status_from_age(age),
            "direction":         direction,
            "confidence":        round(base_conf * age_mult, 1),
            "breakout_level":    round((high if direction == "bullish" else low) * (1.002 if direction == "bullish" else 0.998), 4),
            "invalidation_level": round(low - atr if direction == "bullish" else high + atr, 4),
            "target":            round(price + (2*atr if direction == "bullish" else -2*atr), 4),
            "points":            [[idx, price]],
        })
    return results
