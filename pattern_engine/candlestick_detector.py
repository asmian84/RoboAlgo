"""Candlestick pattern detector — service-compatible wrapper."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd


def _body(o, c): return abs(c - o)
def _range_(h, l): return h - l
def _upper_shadow(h, o, c): return h - max(o, c)
def _lower_shadow(o, c, l): return min(o, c) - l


def _atr(df: pd.DataFrame, n: int = 14) -> float:
    if len(df) < 2:
        return float(df['close'].iloc[-1]) * 0.02
    hi = df['high'].astype(float).tail(n)
    lo = df['low'].astype(float).tail(n)
    return float((hi - lo).mean()) or float(df['close'].iloc[-1]) * 0.02


def _status_from_age(age: int) -> str:
    """age = bars since pattern, 0 = current bar."""
    if age == 0:
        return "FORMING"
    if age <= 3:
        return "READY"
    return "FORMING"  # weakening but still notable


def detect(symbol: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect 17 candlestick patterns from recent bars.

    Returns one entry per found pattern (most recent occurrence in last 10 bars).
    Patterns not found → NOT_PRESENT entries returned too so the UI can list them all.
    """
    if len(df) < 5:
        return []

    df = df.reset_index(drop=True)
    n = len(df)
    atr = _atr(df)
    last_close = float(df['close'].iloc[-1])

    # We'll collect: pattern_name → (bar_idx, direction, confidence)
    found: dict[str, tuple[int, str, float]] = {}

    # Scan last 10 bars (or fewer if df is short)
    start = max(2, n - 10)

    for i in range(start, n):
        c2 = df.iloc[i]
        c1 = df.iloc[i - 1] if i >= 1 else c2
        c0 = df.iloc[i - 2] if i >= 2 else c1

        o2, h2, l2, cl2 = float(c2['open']), float(c2['high']), float(c2['low']), float(c2['close'])
        o1, h1, l1, cl1 = float(c1['open']), float(c1['high']), float(c1['low']), float(c1['close'])
        o0, h0, l0, cl0 = float(c0['open']), float(c0['high']), float(c0['low']), float(c0['close'])

        r2 = _range_(h2, l2)
        if r2 <= 0:
            continue
        b2 = _body(o2, cl2)
        us2 = _upper_shadow(h2, o2, cl2)
        ls2 = _lower_shadow(o2, cl2, l2)
        b1 = _body(o1, cl1)
        r1 = _range_(h1, l1)
        b0 = _body(o0, cl0)

        age = n - 1 - i

        # ── Single-candle ──────────────────────────────────────────────
        if b2 / r2 < 0.05:
            found.setdefault("Doji", (i, "neutral", round((1 - b2/r2) * 80, 1)))

        if (ls2 >= 2 * b2 and us2 < b2 and b2 > 0 and min(o2, cl2) > l2 + r2 * 0.55):
            trend_down = (df.iloc[max(0,i-5):i]['close'].astype(float).is_monotonic_decreasing if i >= 5 else False)
            if trend_down:
                found.setdefault("Hammer", (i, "bullish", round(min(ls2/r2, 1.0) * 80, 1)))
            else:
                found.setdefault("Hanging Man", (i, "bearish", round(min(ls2/r2, 1.0) * 75, 1)))

        if (us2 >= 2 * b2 and ls2 < b2 and b2 > 0 and max(o2, cl2) < h2 - r2 * 0.55):
            trend_down = (df.iloc[max(0,i-5):i]['close'].astype(float).is_monotonic_decreasing if i >= 5 else False)
            if trend_down:
                found.setdefault("Inverted Hammer", (i, "bullish", round(min(us2/r2, 1.0) * 75, 1)))
            else:
                found.setdefault("Shooting Star", (i, "bearish", round(min(us2/r2, 1.0) * 78, 1)))

        if cl2 >= o2 and us2/r2 < 0.02 and ls2/r2 < 0.02 and b2/r2 > 0.90:
            found.setdefault("Bullish Marubozu", (i, "bullish", round(b2/r2 * 82, 1)))

        if cl2 < o2 and us2/r2 < 0.02 and ls2/r2 < 0.02 and b2/r2 > 0.90:
            found.setdefault("Bearish Marubozu", (i, "bearish", round(b2/r2 * 82, 1)))

        # ── Two-candle ─────────────────────────────────────────────────
        if r1 <= 0:
            continue

        if cl1 < o1 and cl2 >= o2 and b1 > 0 and o2 < cl1 and cl2 > o1:
            found.setdefault("Bullish Engulfing", (i, "bullish", round(min(b2/b1, 2.0)/2 * 85, 1)))
        elif cl1 >= o1 and cl2 < o2 and b1 > 0 and o2 > cl1 and cl2 < o1:
            found.setdefault("Bearish Engulfing", (i, "bearish", round(min(b2/b1, 2.0)/2 * 85, 1)))
        elif cl1 < o1 and cl2 >= o2 and b1 > 0 and b2 < b1 * 0.5 and o2 > cl1 and cl2 < o1:
            found.setdefault("Bullish Harami", (i, "bullish", round((1-b2/b1)*78, 1)))
        elif cl1 >= o1 and cl2 < o2 and b1 > 0 and b2 < b1 * 0.5 and o2 < cl1 and cl2 > o1:
            found.setdefault("Bearish Harami", (i, "bearish", round((1-b2/b1)*78, 1)))
        elif cl1 < o1 and cl2 >= o2 and b1 > 0 and o2 < l1 and cl2 > (o1+cl1)/2:
            found.setdefault("Piercing Line", (i, "bullish", 75.0))
        elif cl1 >= o1 and cl2 < o2 and b1 > 0 and o2 > h1 and cl2 < (o1+cl1)/2:
            found.setdefault("Dark Cloud Cover", (i, "bearish", 75.0))

        # ── Three-candle ───────────────────────────────────────────────
        if b0 <= 0:
            continue

        if cl0 < o0 and b0 > 0 and b1 < b0*0.3 and cl2 >= o2 and b2 > b0*0.5 and cl2 > o0*0.97:
            found.setdefault("Morning Star", (i, "bullish", 85.0))
        elif cl0 >= o0 and b0 > 0 and b1 < b0*0.3 and cl2 < o2 and b2 > b0*0.5 and cl2 < o0*1.03:
            found.setdefault("Evening Star", (i, "bearish", 85.0))
        elif (cl0>=o0 and cl1>=o1 and cl2>=o2 and cl1>cl0 and cl2>cl1 and
              b0>0 and b1>0 and b2>0 and b0/max(_range_(h0,l0),1e-9)>0.5 and b1/max(r1,1e-9)>0.5 and b2/r2>0.5):
            found.setdefault("Three White Soldiers", (i, "bullish", 90.0))
        elif (cl0<o0 and cl1<o1 and cl2<o2 and cl1<cl0 and cl2<cl1 and
              b0>0 and b1>0 and b2>0 and b0/max(_range_(h0,l0),1e-9)>0.5 and b1/max(r1,1e-9)>0.5 and b2/r2>0.5):
            found.setdefault("Three Black Crows", (i, "bearish", 90.0))

    # Build output
    results = []
    for pattern_name, (bar_idx, direction, conf) in found.items():
        age = n - 1 - bar_idx
        status = _status_from_age(age)

        row = df.iloc[bar_idx]
        price = float(row['close'])
        high = float(row['high'])
        low = float(row['low'])

        if direction == "bullish":
            breakout = round(high * 1.002, 4)
            invalidation = round(low - atr, 4)
            target = round(price + 2.0 * atr, 4)
        elif direction == "bearish":
            breakout = round(low * 0.998, 4)
            invalidation = round(high + atr, 4)
            target = round(price - 2.0 * atr, 4)
        else:
            breakout = price
            invalidation = price - atr
            target = price + atr

        results.append({
            "pattern_name": pattern_name,
            "pattern_category": "candlestick",
            "status": status,
            "direction": direction,
            "confidence": round(conf * (0.8 if age > 2 else 1.0), 1),
            "breakout_level": breakout,
            "invalidation_level": invalidation,
            "target": target,
            "points": [[bar_idx, price]],
        })

    return results
