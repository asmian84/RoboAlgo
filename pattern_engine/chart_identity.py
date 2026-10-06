from __future__ import annotations

import numpy as np
import pandas as pd


def detect_chart_identity(symbol: str, df: pd.DataFrame) -> list[str]:
    """Return 1-3 short identity labels for the chart. Always returns at least one."""
    if df is None or len(df) < 5:
        return ["Sideways"]

    labels: list[str] = []

    o = df["open"].values
    h = df["high"].values
    lo = df["low"].values
    c = df["close"].values
    has_volume = "volume" in df.columns
    vol = df["volume"].values if has_volume else None

    # ------------------------------------------------------------------
    # Candlestick patterns (last 3 bars)
    # ------------------------------------------------------------------
    _candle_label = _detect_candlestick(o, h, lo, c)
    if _candle_label:
        labels.append(_candle_label)

    # ------------------------------------------------------------------
    # Trend state (last 20 bars)
    # ------------------------------------------------------------------
    if len(labels) < 3:
        _trend = _detect_trend(c, h, lo)
        if _trend:
            labels.append(_trend)

    # ------------------------------------------------------------------
    # Volume features
    # ------------------------------------------------------------------
    if len(labels) < 3 and has_volume and vol is not None and len(vol) >= 20:
        _vol = _detect_volume(vol)
        if _vol:
            labels.append(_vol)

    # ------------------------------------------------------------------
    # Gap detection
    # ------------------------------------------------------------------
    if len(labels) < 3 and len(df) >= 2:
        _gap = _detect_gap(h, lo)
        if _gap:
            labels.append(_gap)

    return labels if labels else ["Sideways"]


# ======================================================================
# Candlestick helpers
# ======================================================================

def _detect_candlestick(o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray) -> str | None:
    """Check the last 3 bars for candlestick patterns (priority order)."""
    n = len(o)
    if n < 2:
        return None

    # Current bar values
    oc = o[-1]
    hc = h[-1]
    lc = lo[-1]
    cc = c[-1]
    body = abs(cc - oc)
    rng = hc - lc

    # Previous bar values
    op = o[-2]
    cp = c[-2]
    body_prev = abs(cp - op)

    upper_shadow = hc - max(oc, cc)
    lower_shadow = min(oc, cc) - lc

    # --- 3-bar patterns (need at least 3 bars) ---
    if n >= 3:
        o3 = o[-3]
        c3 = c[-3]
        body3 = abs(c3 - o3)
        avg_body = (body3 + body_prev + body) / 3.0 if (body3 + body_prev + body) > 0 else 1.0

        # Morning Star: big red, small body, big green
        if (c3 < o3 and body3 > avg_body
                and body_prev < 0.3 * body3
                and cc > oc and body > avg_body):
            return "Morning Star"

        # Evening Star: big green, small body, big red
        if (c3 > o3 and body3 > avg_body
                and body_prev < 0.3 * body3
                and cc < oc and body > avg_body):
            return "Evening Star"

    # --- 2-bar patterns ---
    # Bullish Engulfing
    if (cp < op and cc > oc
            and oc <= cp and cc >= op
            and body > body_prev):
        return "Bullish Engulfing"

    # Bearish Engulfing
    if (cp > op and cc < oc
            and oc >= cp and cc <= op
            and body > body_prev):
        return "Bearish Engulfing"

    # --- 1-bar patterns ---
    if rng <= 0:
        return None

    # Doji
    if body <= 0.1 * rng:
        return "Doji"

    # Marubozu: shadows < 10% of range
    if upper_shadow < 0.1 * rng and lower_shadow < 0.1 * rng and body > 0:
        return "Marubozu"

    # Hammer: lower shadow >= 2x body, small upper shadow, downtrend context
    if lower_shadow >= 2 * body and upper_shadow <= 0.3 * rng:
        if n >= 5 and c[-5] > cc:
            return "Hammer"

    # Shooting Star: upper shadow >= 2x body, small lower shadow, uptrend context
    if upper_shadow >= 2 * body and lower_shadow <= 0.3 * rng:
        if n >= 5 and c[-5] < cc:
            return "Shooting Star"

    # Spinning Top: small body, both shadows > body
    if upper_shadow > body and lower_shadow > body and body < 0.3 * rng:
        return "Spinning Top"

    return None


# ======================================================================
# Trend helpers
# ======================================================================

def _detect_trend(c: np.ndarray, h: np.ndarray, lo: np.ndarray) -> str | None:
    """Classify trend state from the last 20 bars."""
    n = len(c)

    # Breakout / Breakdown (check first, more specific)
    if n >= 21:
        prev_high = np.max(h[-21:-1])
        prev_low = np.min(lo[-21:-1])
        if c[-1] > prev_high:
            return "Breakout"
        if c[-1] < prev_low:
            return "Breakdown"

    # SMA-based trend
    if n >= 20:
        sma20 = np.mean(c[-20:])
        sma20_prev = np.mean(c[-21:-1]) if n >= 21 else np.mean(c[-20:-1])
        slope = sma20 - sma20_prev

        if c[-1] > sma20 and slope > 0:
            return "Uptrend"
        if c[-1] < sma20 and slope < 0:
            return "Downtrend"

    # Consolidation: last 10 bars range < 0.5 * ATR14
    if n >= 14:
        rng_10 = np.max(h[-10:]) - np.min(lo[-10:]) if n >= 10 else np.max(h) - np.min(lo)
        # Simple ATR14
        tr_vals = np.maximum(
            h[-14:] - lo[-14:],
            np.maximum(
                np.abs(h[-14:] - np.concatenate([[c[-15]], c[-14:-1]])) if n >= 15
                else h[-14:] - lo[-14:],
                np.abs(lo[-14:] - np.concatenate([[c[-15]], c[-14:-1]])) if n >= 15
                else h[-14:] - lo[-14:],
            ),
        )
        atr14 = np.mean(tr_vals)
        if atr14 > 0 and rng_10 < 0.5 * atr14:
            return "Consolidation"

    return None


# ======================================================================
# Volume helpers
# ======================================================================

def _detect_volume(vol: np.ndarray) -> str | None:
    """Classify volume on the last bar vs 20-bar average."""
    if len(vol) < 20:
        return None
    avg20 = np.mean(vol[-21:-1]) if len(vol) >= 21 else np.mean(vol[-20:-1])
    if avg20 <= 0:
        return None
    last = vol[-1]
    if last > 3 * avg20:
        return "Climax Volume"
    if last > 2 * avg20:
        return "High Volume"
    if last < 0.3 * avg20:
        return "Volume Dry-Up"
    return None


# ======================================================================
# Gap helpers
# ======================================================================

def _detect_gap(h: np.ndarray, lo: np.ndarray) -> str | None:
    """Detect gap up/down between the last two bars."""
    if len(h) < 2:
        return None
    if lo[-1] > h[-2]:
        return "Gap Up"
    if h[-1] < lo[-2]:
        return "Gap Down"
    return None
