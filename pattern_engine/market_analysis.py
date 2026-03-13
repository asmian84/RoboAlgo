"""Market analysis pattern detector.

Implements macro-level market analysis patterns:
  1. Dow Theory Trend Classification
  2. Elliott Wave Impulse Detection
  3. Intermarket Analysis Signals (proxy, single-ticker)
  4. Market Breadth / Strength Signals
  5. Wyckoff Price Cycle Analysis
  6. Price Action / Relative Strength Patterns

Public API
----------
detect(symbol: str, df: pd.DataFrame) -> list[dict]
    df must have columns: date, open, high, low, close, volume
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PATTERN_CATEGORY = "market_analysis"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _base(name: str) -> dict[str, Any]:
    return {
        "pattern_name": name,
        "pattern_category": PATTERN_CATEGORY,
        "status": "NOT_PRESENT",
        "direction": "neutral",
        "confidence": 0.0,
        "breakout_level": None,
        "invalidation_level": None,
        "target": None,
        "points": [],
    }


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    """Simple ATR approximation using high-low range."""
    return float((df["high"] - df["low"]).astype(float).tail(period).mean())


def _levels(current_price: float, atr: float, direction: str) -> tuple[float | None, float | None, float | None]:
    """Return (breakout_level, invalidation_level, target) for standard ATR-based levels."""
    if atr <= 0:
        return None, None, None
    if direction == "bullish":
        return (
            round(current_price, 4),
            round(current_price - 1.5 * atr, 4),
            round(current_price + 2.5 * atr, 4),
        )
    if direction == "bearish":
        return (
            round(current_price, 4),
            round(current_price + 1.5 * atr, 4),
            round(current_price - 2.5 * atr, 4),
        )
    return None, None, None


def _swing_points(df: pd.DataFrame, window: int = 5) -> list[tuple[int, float, str]]:
    """Return alternating swing highs and lows as [(idx, price, 'high'|'low'), ...]."""
    n = len(df)
    raw: list[tuple[int, float, str]] = []
    for i in range(window, n - window):
        hi = float(df.iloc[i]["high"])
        lo = float(df.iloc[i]["low"])
        is_sh = all(
            hi >= float(df.iloc[i + k]["high"])
            for k in range(-window, window + 1)
            if k != 0
        )
        is_sl = all(
            lo <= float(df.iloc[i + k]["low"])
            for k in range(-window, window + 1)
            if k != 0
        )
        if is_sh:
            raw.append((i, hi, "high"))
        elif is_sl:
            raw.append((i, lo, "low"))

    # Enforce strict alternation (keep last occurrence when adjacent types match)
    alternating: list[tuple[int, float, str]] = []
    for pt in raw:
        if alternating and alternating[-1][2] == pt[2]:
            # Replace with whichever extreme is more extreme
            prev = alternating[-1]
            if pt[2] == "high" and pt[1] >= prev[1]:
                alternating[-1] = pt
            elif pt[2] == "low" and pt[1] <= prev[1]:
                alternating[-1] = pt
        else:
            alternating.append(pt)
    return alternating


def _recent_swing_points(df: pd.DataFrame, window: int = 5, lookback: int = 60) -> list[tuple[int, float, str]]:
    """Swing points restricted to the tail portion of df."""
    sub = df.tail(lookback).reset_index(drop=True)
    pts = _swing_points(sub, window)
    # Map local indices back to global df indices
    offset = len(df) - len(sub)
    return [(idx + offset, price, kind) for idx, price, kind in pts]


# ---------------------------------------------------------------------------
# 1. Dow Theory
# ---------------------------------------------------------------------------


def _detect_dow_theory(df: pd.DataFrame) -> list[dict]:
    results: list[dict] = []
    atr = _atr(df)
    current_price = float(df["close"].iloc[-1])
    last_bar = len(df) - 1

    swings = _recent_swing_points(df, window=5, lookback=60)
    if len(swings) < 4:
        return results

    last4 = swings[-4:]
    pts = [[int(idx), float(price)] for idx, price, _ in last4]

    highs = [(idx, price) for idx, price, kind in swings if kind == "high"]
    lows = [(idx, price) for idx, price, kind in swings if kind == "low"]

    def _is_recent(idx: int) -> bool:
        return idx >= last_bar - 19

    # ── Primary Uptrend ──────────────────────────────────────────────────────
    if len(highs) >= 2 and len(lows) >= 2:
        recent_highs = highs[-4:] if len(highs) >= 4 else highs
        recent_lows = lows[-4:] if len(lows) >= 4 else lows
        hh = all(recent_highs[i][1] > recent_highs[i - 1][1] for i in range(1, len(recent_highs)))
        hl = all(recent_lows[i][1] > recent_lows[i - 1][1] for i in range(1, len(recent_lows)))
        if hh and hl:
            confirm = len(recent_highs) + len(recent_lows)
            conf = float(np.clip(72 + (confirm - 4) * 2.5, 72, 82))
            r = _base("Dow Primary Uptrend")
            r["direction"] = "bullish"
            r["confidence"] = conf
            r["status"] = "READY"
            r["points"] = pts
            bk, inv, tgt = _levels(current_price, atr, "bullish")
            r["breakout_level"] = bk
            r["invalidation_level"] = inv
            r["target"] = tgt
            results.append(r)

    # ── Primary Downtrend ────────────────────────────────────────────────────
    if len(highs) >= 2 and len(lows) >= 2:
        recent_highs = highs[-4:] if len(highs) >= 4 else highs
        recent_lows = lows[-4:] if len(lows) >= 4 else lows
        lh = all(recent_highs[i][1] < recent_highs[i - 1][1] for i in range(1, len(recent_highs)))
        ll = all(recent_lows[i][1] < recent_lows[i - 1][1] for i in range(1, len(recent_lows)))
        if lh and ll:
            confirm = len(recent_highs) + len(recent_lows)
            conf = float(np.clip(72 + (confirm - 4) * 2.5, 72, 82))
            r = _base("Dow Primary Downtrend")
            r["direction"] = "bearish"
            r["confidence"] = conf
            r["status"] = "READY"
            r["points"] = pts
            bk, inv, tgt = _levels(current_price, atr, "bearish")
            r["breakout_level"] = bk
            r["invalidation_level"] = inv
            r["target"] = tgt
            results.append(r)

    # ── Bullish Reversal Signal: after downtrend, higher swing low ───────────
    if len(lows) >= 3:
        # Confirm prior downtrend using last 3 lows (lower lows)
        prior_lows = lows[-3:]
        prior_downtrend = all(prior_lows[i][1] < prior_lows[i - 1][1] for i in range(1, len(prior_lows) - 1)) if len(prior_lows) >= 3 else False
        # Actually: check if latest low is HIGHER than the one before it
        if len(lows) >= 2:
            latest_low = lows[-1]
            prev_low = lows[-2]
            if latest_low[1] > prev_low[1] and _is_recent(latest_low[0]):
                # Ensure we had a downtrend before: check that prev_low was lower than the one before
                had_downtrend = len(lows) >= 3 and lows[-2][1] < lows[-3][1]
                if had_downtrend:
                    r = _base("Dow Trend Reversal Signal Bullish")
                    r["direction"] = "bullish"
                    r["confidence"] = 68.0
                    r["status"] = "FORMING"
                    r["points"] = [[int(prev_low[0]), float(prev_low[1])], [int(latest_low[0]), float(latest_low[1])]]
                    bk, inv, tgt = _levels(current_price, atr, "bullish")
                    r["breakout_level"] = bk
                    r["invalidation_level"] = inv
                    r["target"] = tgt
                    results.append(r)

    # ── Bearish Reversal Signal: after uptrend, lower swing high ─────────────
    if len(highs) >= 2:
        latest_high = highs[-1]
        prev_high = highs[-2]
        if latest_high[1] < prev_high[1] and _is_recent(latest_high[0]):
            had_uptrend = len(highs) >= 3 and highs[-2][1] > highs[-3][1]
            if had_uptrend:
                r = _base("Dow Trend Reversal Signal Bearish")
                r["direction"] = "bearish"
                r["confidence"] = 68.0
                r["status"] = "FORMING"
                r["points"] = [[int(prev_high[0]), float(prev_high[1])], [int(latest_high[0]), float(latest_high[1])]]
                bk, inv, tgt = _levels(current_price, atr, "bearish")
                r["breakout_level"] = bk
                r["invalidation_level"] = inv
                r["target"] = tgt
                results.append(r)

    return results


# ---------------------------------------------------------------------------
# 2. Elliott Wave Impulse Detection
# ---------------------------------------------------------------------------


def _detect_elliott_wave(df: pd.DataFrame) -> list[dict]:
    results: list[dict] = []
    atr = _atr(df)
    current_price = float(df["close"].iloc[-1])
    last_bar = len(df) - 1

    swings = _recent_swing_points(df, window=3, lookback=min(len(df), 120))
    if len(swings) < 6:
        return results

    def _is_recent(idx: int) -> bool:
        return idx >= last_bar - 30

    def _retrace(leg_start: float, leg_end: float, retrace_end: float) -> float:
        leg = abs(leg_end - leg_start)
        if leg == 0:
            return 0.0
        ret = abs(retrace_end - leg_end)
        return ret / leg

    def _check_bullish_5wave(pts: list[tuple[int, float, str]]) -> dict | None:
        """Check whether 5 alternating swing points form a valid bullish impulse."""
        if len(pts) < 5:
            return None
        # pts[0]=low, pts[1]=high, pts[2]=low, pts[3]=high, pts[4]=low or high
        # For a 5-wave bullish impulse we need: low, high, low, high, low (0-4) driving up
        # Wave labels: W0=base, W1 top, W2 bottom, W3 top, W4 bottom, W5 top
        # We look for: L,H,L,H,L,H  → 6 points
        if len(pts) < 6:
            return None
        kinds = [k for _, _, k in pts[-6:]]
        if kinds[0] != "low":
            return None
        expected = ["low", "high", "low", "high", "low", "high"]
        if kinds != expected:
            return None

        p = pts[-6:]
        p0_idx, p0_price, _ = p[0]  # Wave 0 base
        p1_idx, p1_price, _ = p[1]  # Wave 1 top
        p2_idx, p2_price, _ = p[2]  # Wave 2 bottom
        p3_idx, p3_price, _ = p[3]  # Wave 3 top
        p4_idx, p4_price, _ = p[4]  # Wave 4 bottom
        p5_idx, p5_price, _ = p[5]  # Wave 5 top

        w1 = p1_price - p0_price
        w3 = p3_price - p2_price
        w5 = p5_price - p4_price

        if w1 <= 0 or w3 <= 0:
            return None

        # Rule: Wave 3 is not shortest
        if w3 < w1 and w3 < w5:
            return None

        # Rule: Wave 2 retraces 38-62% of Wave 1
        w2_ret = _retrace(p0_price, p1_price, p2_price)
        if not (0.30 <= w2_ret <= 0.70):
            return None

        # Rule: Wave 4 does not overlap Wave 1 top
        if p4_price < p1_price:
            return None

        return {
            "points": [[int(idx), float(price)] for idx, price, _ in p],
            "w1": w1, "w3": w3, "w5": w5,
            "p5_idx": p5_idx,
        }

    def _check_bullish_partial(pts: list[tuple[int, float, str]]) -> dict | None:
        """Check for bullish 4-point sequence (may be in Wave 3 or Wave 2)."""
        if len(pts) < 4:
            return None
        kinds = [k for _, _, k in pts[-4:]]
        if kinds != ["low", "high", "low", "high"]:
            return None
        p = pts[-4:]
        p0_idx, p0_price, _ = p[0]
        p1_idx, p1_price, _ = p[1]
        p2_idx, p2_price, _ = p[2]
        p3_idx, p3_price, _ = p[3]

        w1 = p1_price - p0_price
        if w1 <= 0:
            return None
        w2_ret = _retrace(p0_price, p1_price, p2_price)
        if not (0.30 <= w2_ret <= 0.70):
            return None
        return {
            "points": [[int(idx), float(price)] for idx, price, _ in p],
            "p3_idx": p3_idx,
        }

    def _check_w2_entry(pts: list[tuple[int, float, str]]) -> dict | None:
        """3-point check: possibly in Wave 2 pullback."""
        if len(pts) < 3:
            return None
        kinds = [k for _, _, k in pts[-3:]]
        if kinds != ["low", "high", "low"]:
            return None
        p = pts[-3:]
        p0_idx, p0_price, _ = p[0]
        p1_idx, p1_price, _ = p[1]
        p2_idx, p2_price, _ = p[2]
        w1 = p1_price - p0_price
        if w1 <= 0:
            return None
        w2_ret = _retrace(p0_price, p1_price, p2_price)
        if not (0.35 <= w2_ret <= 0.68):
            return None
        return {
            "points": [[int(idx), float(price)] for idx, price, _ in p],
            "p2_idx": p2_idx,
        }

    # ── 5-wave complete / Wave 5 in progress ─────────────────────────────────
    five = _check_bullish_5wave(swings)
    if five and _is_recent(five["p5_idx"]):
        r = _base("Elliott Wave 5 in Progress")
        r["direction"] = "neutral"
        r["confidence"] = 65.0
        r["status"] = "FORMING"
        r["points"] = five["points"]
        r["breakout_level"] = None
        r["invalidation_level"] = None
        r["target"] = None
        results.append(r)

    # ── Wave 3 Upswing (4 points, in Wave 3 territory) ───────────────────────
    partial = _check_bullish_partial(swings)
    if partial and _is_recent(partial["p3_idx"]):
        r = _base("Elliott Wave 3 Upswing")
        r["direction"] = "bullish"
        r["confidence"] = 72.0
        r["status"] = "FORMING"
        r["points"] = partial["points"]
        bk, inv, tgt = _levels(current_price, atr, "bullish")
        r["breakout_level"] = bk
        r["invalidation_level"] = inv
        r["target"] = tgt
        results.append(r)

    # ── Wave 2 Pullback Entry ─────────────────────────────────────────────────
    w2 = _check_w2_entry(swings)
    if w2 and _is_recent(w2["p2_idx"]):
        r = _base("Elliott Wave 2 Pullback Entry")
        r["direction"] = "bullish"
        r["confidence"] = 68.0
        r["status"] = "FORMING"
        r["points"] = w2["points"]
        bk, inv, tgt = _levels(current_price, atr, "bullish")
        r["breakout_level"] = bk
        r["invalidation_level"] = inv
        r["target"] = tgt
        results.append(r)

    # ── Bearish mirror: 5-wave impulse down ──────────────────────────────────
    def _check_bearish_5wave(pts: list[tuple[int, float, str]]) -> dict | None:
        if len(pts) < 6:
            return None
        kinds = [k for _, _, k in pts[-6:]]
        if kinds != ["high", "low", "high", "low", "high", "low"]:
            return None
        p = pts[-6:]
        p0_idx, p0_price, _ = p[0]
        p1_idx, p1_price, _ = p[1]
        p2_idx, p2_price, _ = p[2]
        p3_idx, p3_price, _ = p[3]
        p4_idx, p4_price, _ = p[4]
        p5_idx, p5_price, _ = p[5]

        w1 = p0_price - p1_price
        w3 = p2_price - p3_price
        w5 = p4_price - p5_price

        if w1 <= 0 or w3 <= 0:
            return None
        if w3 < w1 and w3 < w5:
            return None
        w2_ret = abs(p2_price - p1_price) / max(w1, 1e-9)
        if not (0.30 <= w2_ret <= 0.70):
            return None
        if p4_price > p1_price:
            return None
        return {
            "points": [[int(idx), float(price)] for idx, price, _ in p],
            "p5_idx": p5_idx,
        }

    five_bear = _check_bearish_5wave(swings)
    if five_bear and _is_recent(five_bear["p5_idx"]):
        r = _base("Elliott Wave 5 in Progress")
        r["direction"] = "neutral"
        r["confidence"] = 65.0
        r["status"] = "FORMING"
        r["points"] = five_bear["points"]
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# 3. Intermarket Analysis Signals (proxy)
# ---------------------------------------------------------------------------


def _detect_intermarket(df: pd.DataFrame) -> list[dict]:
    results: list[dict] = []
    close = df["close"].astype(float)
    last_bar = len(df) - 1
    current_price = float(close.iloc[-1])
    atr = _atr(df)

    if len(close) < 22:
        return results

    price_20d_ago = float(close.iloc[-21])
    price_5d_ago = float(close.iloc[-6])
    price_now = float(close.iloc[-1])

    change_20d = (price_now - price_20d_ago) / max(price_20d_ago, 1e-9)
    change_5d = (price_now - price_5d_ago) / max(price_5d_ago, 1e-9)

    # Flight to safety: Risk-Off Bounce
    if change_20d < -0.08 and change_5d > 0.02:
        r = _base("Risk-Off Bounce")
        r["direction"] = "bullish"
        r["confidence"] = 65.0
        r["status"] = "FORMING"
        r["points"] = [[last_bar - 20, round(price_20d_ago, 4)], [last_bar, round(price_now, 4)]]
        bk, inv, tgt = _levels(current_price, atr, "bullish")
        r["breakout_level"] = bk
        r["invalidation_level"] = inv
        r["target"] = tgt
        results.append(r)

    # Extended Rally Caution
    if change_20d > 0.15 and change_5d < -0.03:
        r = _base("Extended Rally Caution")
        r["direction"] = "bearish"
        r["confidence"] = 65.0
        r["status"] = "FORMING"
        r["points"] = [[last_bar - 20, round(price_20d_ago, 4)], [last_bar, round(price_now, 4)]]
        bk, inv, tgt = _levels(current_price, atr, "bearish")
        r["breakout_level"] = bk
        r["invalidation_level"] = inv
        r["target"] = tgt
        results.append(r)

    # Volatility Expansion (ATR > 1.5× 20-bar average)
    atr_14 = float((df["high"] - df["low"]).astype(float).tail(14).mean())
    atr_20 = float((df["high"] - df["low"]).astype(float).tail(20).mean())
    if atr_20 > 0 and atr_14 > 1.5 * atr_20:
        r = _base("Volatility Expansion")
        r["direction"] = "neutral"
        r["confidence"] = 62.0
        r["status"] = "FORMING"
        r["points"] = [[last_bar, round(current_price, 4)]]
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# 4. Market Breadth / Strength Signals
# ---------------------------------------------------------------------------


def _detect_breadth(df: pd.DataFrame) -> list[dict]:
    results: list[dict] = []
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    last_bar = len(df) - 1
    current_price = float(close.iloc[-1])
    atr = _atr(df)

    if len(df) < 6:
        return results

    tail_close = close.tail(5)
    tail_vol = volume.tail(5)

    price_rising = float(tail_close.iloc[-1]) > float(tail_close.iloc[0])
    vol_rising = float(tail_vol.mean()) > float(volume.tail(20).mean()) if len(volume) >= 20 else float(tail_vol.iloc[-1]) > float(tail_vol.iloc[0])

    pts = [[last_bar, round(current_price, 4)]]

    if price_rising and vol_rising:
        r = _base("Healthy Uptrend")
        r["direction"] = "bullish"
        r["confidence"] = 67.0
        r["status"] = "FORMING"
        r["points"] = pts
        bk, inv, tgt = _levels(current_price, atr, "bullish")
        r["breakout_level"] = bk
        r["invalidation_level"] = inv
        r["target"] = tgt
        results.append(r)

    elif price_rising and not vol_rising:
        r = _base("Diverging Volume Warning")
        r["direction"] = "bearish"
        r["confidence"] = 65.0
        r["status"] = "FORMING"
        r["points"] = pts
        bk, inv, tgt = _levels(current_price, atr, "bearish")
        r["breakout_level"] = bk
        r["invalidation_level"] = inv
        r["target"] = tgt
        results.append(r)

    elif not price_rising and vol_rising:
        r = _base("Distribution Warning")
        r["direction"] = "bearish"
        r["confidence"] = 68.0
        r["status"] = "FORMING"
        r["points"] = pts
        bk, inv, tgt = _levels(current_price, atr, "bearish")
        r["breakout_level"] = bk
        r["invalidation_level"] = inv
        r["target"] = tgt
        results.append(r)

    else:
        r = _base("Healthy Pullback")
        r["direction"] = "bullish"
        r["confidence"] = 65.0
        r["status"] = "FORMING"
        r["points"] = pts
        bk, inv, tgt = _levels(current_price, atr, "bullish")
        r["breakout_level"] = bk
        r["invalidation_level"] = inv
        r["target"] = tgt
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# 5. Wyckoff Price Cycle Analysis
# ---------------------------------------------------------------------------


def _detect_wyckoff(df: pd.DataFrame) -> list[dict]:
    results: list[dict] = []
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    last_bar = len(df) - 1
    current_price = float(close.iloc[-1])
    atr = _atr(df)

    # Support and resistance from rolling 20-bar high/low
    support = float(low.tail(20).min())
    resistance = float(high.tail(20).max())
    avg_vol = float(volume.tail(30).mean()) if len(volume) >= 30 else float(volume.mean())

    # ── Phase A: Selling Climax + Automatic Rally ────────────────────────────
    # Sharp drop (>3% in 3 bars) + volume spike + immediate bounce
    if len(df) >= 8:
        window = close.tail(8)
        wvol = volume.tail(8)
        wlow = low.tail(8)
        whigh = high.tail(8)

        drop_start = float(window.iloc[0])
        drop_end = float(window.iloc[4])
        drop_pct = (drop_start - drop_end) / max(drop_start, 1e-9)
        vol_spike = float(wvol.iloc[3]) > 1.8 * avg_vol
        bounce = float(window.iloc[-1]) > float(window.iloc[4]) * 1.01

        if drop_pct > 0.03 and vol_spike and bounce:
            r = _base("Wyckoff Selling Climax")
            r["direction"] = "bullish"
            r["confidence"] = 72.0
            r["status"] = "FORMING"
            sc_idx = last_bar - 4
            r["points"] = [
                [last_bar - 7, round(drop_start, 4)],
                [sc_idx, round(drop_end, 4)],
                [last_bar, round(current_price, 4)],
            ]
            bk, inv, tgt = _levels(current_price, atr, "bullish")
            r["breakout_level"] = bk
            r["invalidation_level"] = inv
            r["target"] = tgt
            results.append(r)

    # ── Phase B: Accumulation — choppy, range-bound, elevated volume ─────────
    if len(df) >= 20:
        tail_close = close.tail(20)
        c_max = float(tail_close.max())
        c_min = float(tail_close.min())
        range_pct = (c_max - c_min) / max(c_min, 1e-9)
        avg_vol_20 = float(volume.tail(20).mean())
        avg_vol_prior = float(volume.tail(40).head(20).mean()) if len(volume) >= 40 else avg_vol

        if range_pct < 0.08 and avg_vol_20 > avg_vol_prior * 0.9:
            r = _base("Wyckoff Accumulation Phase B")
            r["direction"] = "neutral"
            r["confidence"] = 65.0
            r["status"] = "FORMING"
            r["points"] = [
                [last_bar - 19, round(float(close.iloc[-20]), 4)],
                [last_bar, round(current_price, 4)],
            ]
            results.append(r)

    # ── Phase C: Spring — undercut of support then closes back above ─────────
    if len(df) >= 5:
        # Check last 3 bars for spring
        for offset in range(0, min(4, len(df) - 1)):
            bar_idx = last_bar - offset
            if bar_idx < 3:
                break
            bar_low = float(df.iloc[bar_idx]["low"])
            bar_close = float(df.iloc[bar_idx]["close"])
            if bar_low < support and bar_close > support:
                # Confirm recovery in subsequent bars (if not current bar)
                r = _base("Wyckoff Spring")
                r["direction"] = "bullish"
                r["confidence"] = 78.0
                r["status"] = "FORMING"
                r["points"] = [
                    [bar_idx, round(bar_low, 4)],
                    [last_bar, round(current_price, 4)],
                ]
                bk, inv, tgt = _levels(current_price, atr, "bullish")
                r["breakout_level"] = bk
                r["invalidation_level"] = inv
                r["target"] = tgt
                results.append(r)
                break

    # ── Phase D/E: Sign of Strength — breakout above resistance, high volume ─
    if len(df) >= 5:
        breakout_bar = close.tail(5)
        breakout_vol = volume.tail(5)
        just_broke_out = any(
            float(breakout_bar.iloc[i]) > resistance * 0.999
            and float(breakout_vol.iloc[i]) > avg_vol * 1.3
            for i in range(len(breakout_bar))
        )
        if just_broke_out:
            r = _base("Wyckoff Sign of Strength")
            r["direction"] = "bullish"
            r["confidence"] = 75.0
            r["status"] = "BREAKOUT"
            r["points"] = [
                [last_bar - 4, round(float(close.iloc[-5]), 4)],
                [last_bar, round(current_price, 4)],
            ]
            bk, inv, tgt = _levels(current_price, atr, "bullish")
            r["breakout_level"] = round(resistance, 4)
            r["invalidation_level"] = inv
            r["target"] = tgt
            results.append(r)

    # ── Distribution: UTAD — upthrust then sharp reversal ───────────────────
    if len(df) >= 5:
        tail_high = high.tail(5)
        tail_close_5 = close.tail(5)
        spike_then_reverse = any(
            float(tail_high.iloc[i]) > resistance * 1.001
            and float(tail_close_5.iloc[i]) < resistance
            for i in range(len(tail_high))
        )
        if spike_then_reverse:
            r = _base("Wyckoff Upthrust")
            r["direction"] = "bearish"
            r["confidence"] = 75.0
            r["status"] = "FORMING"
            r["points"] = [
                [last_bar - 4, round(float(close.iloc[-5]), 4)],
                [last_bar, round(current_price, 4)],
            ]
            bk, inv, tgt = _levels(current_price, atr, "bearish")
            r["breakout_level"] = round(resistance, 4)
            r["invalidation_level"] = inv
            r["target"] = tgt
            results.append(r)

    # ── Distribution: Sign of Weakness — breakdown below support, high volume ─
    if len(df) >= 5:
        breakdown_close = close.tail(5)
        breakdown_vol = volume.tail(5)
        just_broke_down = any(
            float(breakdown_close.iloc[i]) < support * 1.001
            and float(breakdown_vol.iloc[i]) > avg_vol * 1.3
            for i in range(len(breakdown_close))
        )
        if just_broke_down:
            r = _base("Wyckoff Sign of Weakness")
            r["direction"] = "bearish"
            r["confidence"] = 75.0
            r["status"] = "BREAKOUT"
            r["points"] = [
                [last_bar - 4, round(float(close.iloc[-5]), 4)],
                [last_bar, round(current_price, 4)],
            ]
            bk, inv, tgt = _levels(current_price, atr, "bearish")
            r["breakout_level"] = round(support, 4)
            r["invalidation_level"] = inv
            r["target"] = tgt
            results.append(r)

    return results


# ---------------------------------------------------------------------------
# 6. Price Action / Relative Strength Patterns
# ---------------------------------------------------------------------------


def _detect_relative_strength(df: pd.DataFrame) -> list[dict]:
    results: list[dict] = []
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    last_bar = len(df) - 1
    current_price = float(close.iloc[-1])
    current_high = float(high.iloc[-1])
    current_low = float(low.iloc[-1])
    atr = _atr(df)

    # 52-week window (252 trading days) or entire df if shorter
    year_window = min(len(df), 252)
    high_52w = float(high.tail(year_window).max())
    low_52w = float(low.tail(year_window).min())

    prev_high_52w = float(high.tail(year_window).iloc[:-1].max()) if year_window > 1 else high_52w

    # ── 52-Week High Breakout ────────────────────────────────────────────────
    new_high_today = current_high > prev_high_52w
    near_52w_high = current_price >= high_52w * 0.98

    if new_high_today:
        r = _base("52-Week High Breakout")
        r["direction"] = "bullish"
        r["confidence"] = 75.0
        r["status"] = "BREAKOUT"
        r["points"] = [[last_bar, round(current_price, 4)]]
        r["breakout_level"] = round(high_52w, 4)
        _, inv, tgt = _levels(current_price, atr, "bullish")
        r["invalidation_level"] = inv
        r["target"] = tgt
        results.append(r)
    elif near_52w_high:
        r = _base("Near 52-Week High")
        r["direction"] = "bullish"
        r["confidence"] = 68.0
        r["status"] = "FORMING"
        r["points"] = [[last_bar, round(current_price, 4)]]
        r["breakout_level"] = round(high_52w, 4)
        _, inv, tgt = _levels(current_price, atr, "bullish")
        r["invalidation_level"] = inv
        r["target"] = tgt
        results.append(r)

    # ── 52-Week Low Reversal ─────────────────────────────────────────────────
    near_52w_low = current_price <= low_52w * 1.02
    # Reversal: price was near or below 52w low in last 3 bars but is now recovering
    if len(df) >= 4:
        prior_lows = low.tail(4).iloc[:-1]
        was_near_low = float(prior_lows.min()) <= low_52w * 1.005
        reversing = current_price > float(close.tail(4).iloc[-2])
        if near_52w_low and was_near_low and reversing:
            r = _base("52-Week Low Reversal")
            r["direction"] = "bullish"
            r["confidence"] = 70.0
            r["status"] = "FORMING"
            r["points"] = [[last_bar - 3, round(float(close.iloc[-4]), 4)], [last_bar, round(current_price, 4)]]
            r["breakout_level"] = round(low_52w, 4)
            _, inv, tgt = _levels(current_price, atr, "bullish")
            r["invalidation_level"] = inv
            r["target"] = tgt
            results.append(r)

    # ── 20-Day High Breakout ─────────────────────────────────────────────────
    high_20d = float(high.tail(21).iloc[:-1].max()) if len(high) >= 21 else float(high.tail(20).max())
    if current_high > high_20d:
        r = _base("20-Day High Breakout")
        r["direction"] = "bullish"
        r["confidence"] = 68.0
        r["status"] = "BREAKOUT"
        r["points"] = [[last_bar, round(current_price, 4)]]
        r["breakout_level"] = round(high_20d, 4)
        _, inv, tgt = _levels(current_price, atr, "bullish")
        r["invalidation_level"] = inv
        r["target"] = tgt
        results.append(r)

    # ── 20-Day Low Breakdown ─────────────────────────────────────────────────
    low_20d = float(low.tail(21).iloc[:-1].min()) if len(low) >= 21 else float(low.tail(20).min())
    if current_low < low_20d:
        r = _base("20-Day Low Breakdown")
        r["direction"] = "bearish"
        r["confidence"] = 68.0
        r["status"] = "BREAKOUT"
        r["points"] = [[last_bar, round(current_price, 4)]]
        r["breakout_level"] = round(low_20d, 4)
        _, inv, tgt = _levels(current_price, atr, "bearish")
        r["invalidation_level"] = inv
        r["target"] = tgt
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect(symbol: str, df: pd.DataFrame) -> list[dict]:
    """Detect market analysis patterns in df for the given symbol.

    Parameters
    ----------
    symbol : str
        Ticker symbol (informational only — not used in computation).
    df : pd.DataFrame
        OHLCV DataFrame with columns: date, open, high, low, close, volume.
        Must have at least 50 rows.

    Returns
    -------
    list[dict]
        List of pattern dicts.  Each dict contains the keys defined in the
        service contract.  Returns [] if df is too short or on any exception.
    """
    try:
        required = {"open", "high", "low", "close", "volume"}
        if df is None or len(df) < 50:
            return []
        if not required.issubset(set(df.columns)):
            return []

        df = df.reset_index(drop=True)
        last_bar = len(df) - 1

        # Run each detector group
        all_candidates: list[dict] = []
        all_candidates.extend(_detect_dow_theory(df))
        all_candidates.extend(_detect_elliott_wave(df))
        all_candidates.extend(_detect_intermarket(df))
        all_candidates.extend(_detect_breadth(df))
        all_candidates.extend(_detect_wyckoff(df))
        all_candidates.extend(_detect_relative_strength(df))

        # Keep only patterns whose last point falls within last 20 bars;
        # if points is empty, keep unconditionally (single-bar signals).
        def _is_recent(pattern: dict) -> bool:
            pts = pattern.get("points", [])
            if not pts:
                return True
            last_pt_idx = max(p[0] for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2)
            return last_pt_idx >= last_bar - 19

        recent = [p for p in all_candidates if _is_recent(p)]

        # Deduplicate — keep most recent occurrence per pattern_name
        found: dict[str, dict] = {}
        for pattern in recent:
            name = pattern["pattern_name"]
            if name not in found:
                found[name] = pattern
            else:
                # Prefer the one whose last point is more recent
                existing_pts = found[name].get("points", [])
                new_pts = pattern.get("points", [])
                existing_last = max((p[0] for p in existing_pts if isinstance(p, (list, tuple)) and len(p) >= 2), default=-1)
                new_last = max((p[0] for p in new_pts if isinstance(p, (list, tuple)) and len(p) >= 2), default=-1)
                if new_last >= existing_last:
                    found[name] = pattern

        return list(found.values())

    except Exception:
        return []
