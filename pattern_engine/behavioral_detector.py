"""Behavioral crowd-psychology pattern detector — service-compatible."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def _compute_bb_upper(close: pd.Series, period: int = 20, std: float = 2.0) -> pd.Series:
    ma = close.rolling(period).mean()
    sigma = close.rolling(period).std()
    return ma + std * sigma


def _atr(df: pd.DataFrame, n: int = 14) -> float:
    hi = df['high'].astype(float).tail(n)
    lo = df['low'].astype(float).tail(n)
    return float((hi - lo).mean()) or float(df['close'].iloc[-1]) * 0.02


def _swing_highs_lows(df: pd.DataFrame, window: int = 3):
    highs, lows = [], []
    n = len(df)
    for i in range(window, n - window):
        hi = float(df.iloc[i]['high'])
        lo = float(df.iloc[i]['low'])
        if all(hi >= float(df.iloc[i-k]['high']) for k in range(1, window+1)) and \
           all(hi >= float(df.iloc[i+k]['high']) for k in range(1, window+1)):
            highs.append((i, hi))
        if all(lo <= float(df.iloc[i-k]['low']) for k in range(1, window+1)) and \
           all(lo <= float(df.iloc[i+k]['low']) for k in range(1, window+1)):
            lows.append((i, lo))
    return highs, lows


def detect(symbol: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect behavioral/crowd-psychology patterns."""
    if len(df) < 35:
        return []

    df = df.reset_index(drop=True).copy()
    n = len(df)
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)
    atr = _atr(df)

    # Compute indicators inline
    rsi_series = _compute_rsi(close)
    bb_upper_series = _compute_bb_upper(close)
    ma50_series = close.rolling(50).mean()
    avg_vol_20 = volume.rolling(20).mean()

    swing_highs, swing_lows = _swing_highs_lows(df)

    found: dict[str, tuple[int, str, float]] = {}

    # Scan last 15 bars for behavioral signals
    start = max(35, n - 15)

    for i in range(start, n):
        curr_close = float(close.iloc[i])
        curr_open = float(df.iloc[i]['open'])
        curr_high = float(high.iloc[i])
        curr_low = float(low.iloc[i])
        curr_vol = float(volume.iloc[i])
        avg_v = float(avg_vol_20.iloc[i]) if not np.isnan(avg_vol_20.iloc[i]) else curr_vol
        rsi = float(rsi_series.iloc[i]) if not np.isnan(rsi_series.iloc[i]) else 50.0
        bb_up = float(bb_upper_series.iloc[i]) if not np.isnan(bb_upper_series.iloc[i]) else curr_high
        ma50 = float(ma50_series.iloc[i]) if not np.isnan(ma50_series.iloc[i]) else curr_close

        recent_sh = [(idx, v) for idx, v in swing_highs if i - 35 <= idx < i]
        recent_sl = [(idx, v) for idx, v in swing_lows  if i - 35 <= idx < i]

        # 1. Banister (descending staircase reversal)
        if len(recent_sh) >= 3 and len(recent_sl) >= 3:
            sh_vals = [v for _, v in recent_sh[-4:]]
            sl_vals = [v for _, v in recent_sl[-4:]]
            if (all(sh_vals[k] < sh_vals[k-1] for k in range(1, len(sh_vals))) and
                    all(sl_vals[k] < sl_vals[k-1] for k in range(1, len(sl_vals)))):
                rsi_window = [float(rsi_series.iloc[j]) for j in range(max(0,i-35), i) if not np.isnan(rsi_series.iloc[j])]
                if rsi_window and min(rsi_window) < 42:
                    last_sh_v = recent_sh[-1][1]
                    if curr_close > last_sh_v and curr_close >= curr_open:
                        total_decline = (sh_vals[0] - sl_vals[-1]) / sh_vals[0] if sh_vals[0] > 0 else 0
                        conf = round(min(55 + total_decline * 250, 88), 1)
                        found.setdefault("Banister", (i, "bullish", conf))

        # 2. Capitulation Bottom
        if i >= 10:
            hi_10 = float(close.iloc[i-10:i].max())
            drop = (hi_10 - curr_close) / hi_10 if hi_10 > 0 else 0
            lo_wick = min(curr_open, curr_close) - curr_low
            body = abs(curr_close - curr_open)
            if (drop > 0.15 and curr_vol > avg_v * 1.8 and
                    curr_close >= curr_open and lo_wick > body * 0.5):
                conf = round(min(50 + drop * 250, 88), 1)
                found.setdefault("Capitulation Bottom", (i, "bullish", conf))

        # 3. Wave 4 Pullback
        if i >= 20 and rsi is not None:
            base_close = float(close.iloc[i-20])
            peak_close = float(close.iloc[i-20:i].max())
            advance_pct = (peak_close - base_close) / base_close if base_close > 0 else 0
            pullback_pct = (peak_close - curr_close) / peak_close if peak_close > 0 else 0
            near_ma50 = abs(curr_close - ma50) / ma50 < 0.04 if ma50 > 0 else False
            if (advance_pct > 0.15 and 0.07 < pullback_pct < 0.20 and near_ma50 and
                    38 < rsi < 58 and curr_close >= curr_open):
                conf = round(min(60 + advance_pct * 80, 88), 1)
                found.setdefault("Wave 4 Pullback", (i, "bullish", conf))

        # 4. ABC Correction
        if len(recent_sh) >= 2 and len(recent_sl) >= 2:
            peak_sh = recent_sh[-2]
            b_sl = recent_sl[-2]
            b_sh = recent_sh[-1]
            c_sl = recent_sl[-1]
            a_len = peak_sh[1] - b_sl[1]
            b_retrace = (b_sh[1] - b_sl[1]) / a_len if a_len > 0 else 0
            c_len = b_sh[1] - c_sl[1]
            c_ratio = c_len / a_len if a_len > 0 else 0
            in_order = (peak_sh[0] < b_sl[0] < b_sh[0] < c_sl[0])
            if (in_order and 0.35 < b_retrace < 0.68 and 0.75 < c_ratio < 1.35 and
                    rsi < 40 and curr_close >= curr_open and c_sl[0] >= i - 5):
                conf = round(min(55 + (1 - abs(c_ratio-1.0)) * 25, 85), 1)
                found.setdefault("ABC Correction", (i, "bullish", conf))

        # 5. Staircase Uptrend
        if len(recent_sl) >= 3:
            sl_vals_asc = [v for _, v in recent_sl[-4:]]
            asc_lows = all(sl_vals_asc[k] > sl_vals_asc[k-1] for k in range(1, len(sl_vals_asc)))
            last_sl_idx = recent_sl[-1][0]
            at_step = (i - last_sl_idx) <= 5
            if asc_lows and at_step and curr_close >= curr_open:
                conf = round(min(55 + len(recent_sl) * 5, 88), 1)
                found.setdefault("Staircase Uptrend", (i, "bullish", conf))

        # 6. Herd Exhaustion
        if rsi > 74 and curr_close > bb_up and i >= 3:
            last3 = df.iloc[i-3:i+1]
            cl3 = last3['close'].astype(float)
            vl3 = last3['volume'].astype(float)
            cons_up = all(cl3.iloc[k] > cl3.iloc[k-1] for k in range(1, 4))
            vol_decl = vl3.iloc[3] < vl3.iloc[2] < vl3.iloc[1]
            if cons_up and vol_decl:
                conf = round(min((rsi - 74) / 26 * 30 + 55, 88), 1)
                found.setdefault("Herd Exhaustion", (i, "bearish", conf))

        # 7. Dead Cat Bounce
        if i >= 15 and rsi < 48:
            hi_15 = float(close.iloc[i-15:i-3].max())
            lo_3 = float(close.iloc[i-3:i].min())
            drop_15 = (hi_15 - lo_3) / hi_15 if hi_15 > 0 else 0
            bounce = (curr_close - lo_3) / lo_3 if lo_3 > 0 else 0
            vol_bounce = float(volume.iloc[i-3:i+1].mean())
            vol_selloff = float(volume.iloc[i-15:i-3].mean())
            if (drop_15 > 0.15 and 0.03 < bounce < 0.09 and
                    vol_bounce < vol_selloff * 0.85 and curr_close < curr_open):
                conf = round(min(50 + drop_15 * 200, 88), 1)
                found.setdefault("Dead Cat Bounce", (i, "bearish", conf))

        # 8. FOMO Breakout
        if i >= 20 and 50 < rsi < 72:
            resist_20 = float(high.iloc[i-20:i].max())
            ret_5d = (curr_close - float(close.iloc[i-5])) / float(close.iloc[i-5]) if i >= 5 and float(close.iloc[i-5]) > 0 else 0
            if (curr_close > resist_20 * 1.005 and curr_vol > avg_v * 1.45 and
                    ret_5d < 0.15 and curr_close >= curr_open):
                conf = round(min(55 + (curr_vol/avg_v - 1.45) * 20, 88), 1)
                found.setdefault("FOMO Breakout", (i, "bullish", conf))

        # 9. V-Bottom Reversal
        if i >= 3 and rsi < 38:
            low_3d = float(low.iloc[i-3:i+1].min())
            high_3d = float(high.iloc[i-3:i+1].max())
            flush = (high_3d - low_3d) / high_3d if high_3d > 0 else 0
            ref_close = float(close.iloc[i-3])
            drawdown = (ref_close - curr_close) / ref_close if ref_close > 0 else 0
            candle_range = curr_high - curr_low
            body = abs(curr_close - curr_open)
            bull_cl = curr_close > curr_open and (candle_range > 0 and body/candle_range > 0.45)
            if flush >= 0.05 and drawdown >= 0.04 and bull_cl and curr_vol >= avg_v * 1.2:
                conf = round(min(55 + flush * 300, 92), 1)
                found.setdefault("V-Bottom Reversal", (i, "bullish", conf))

        # 10. Distribution Top
        if i >= 22 and rsi > 58:
            hi_20 = float(high.iloc[i-20:i+1].max())
            near_high = curr_close >= hi_20 * 0.97
            if near_high:
                recent_bars = df.iloc[i-4:i+1]
                hv_selling = sum(
                    1 for _, r in recent_bars.iterrows()
                    if float(r['volume']) >= avg_v * 1.2 and float(r['close']) < float(r['open'])
                )
                rsi_3d_ago = float(rsi_series.iloc[i-3]) if i >= 3 and not np.isnan(rsi_series.iloc[i-3]) else rsi
                rsi_diverging = rsi <= rsi_3d_ago
                if hv_selling >= 2 and rsi_diverging and curr_vol >= avg_v * 1.1:
                    conf = round(min(52 + hv_selling * 10, 85), 1)
                    found.setdefault("Distribution Top", (i, "bearish", conf))

    # Build output
    results = []
    for pattern_name, (bar_idx, direction, conf) in found.items():
        age = n - 1 - bar_idx
        if age == 0:
            status = "FORMING"
        elif age <= 4:
            status = "READY"
        else:
            status = "FORMING"

        price = float(close.iloc[bar_idx])
        h_price = float(high.iloc[bar_idx])
        l_price = float(low.iloc[bar_idx])

        if direction == "bullish":
            breakout = round(h_price * 1.002, 4)
            invalidation = round(l_price - atr * 1.5, 4)
            target = round(price + 2.5 * atr, 4)
        elif direction == "bearish":
            breakout = round(l_price * 0.998, 4)
            invalidation = round(h_price + atr * 1.5, 4)
            target = round(price - 2.5 * atr, 4)
        else:
            breakout = price
            invalidation = price - atr
            target = price + atr

        # Decay confidence by age
        age_factor = max(0.6, 1.0 - age * 0.08)

        results.append({
            "pattern_name": pattern_name,
            "pattern_category": "behavioral",
            "status": status,
            "direction": direction,
            "confidence": round(conf * age_factor, 1),
            "breakout_level": breakout,
            "invalidation_level": invalidation,
            "target": target,
            "points": [[bar_idx, price]],
        })

    return results
