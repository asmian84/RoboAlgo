"""Strategy pattern detector based on StockCharts methodology.

Exposes:
    detect(symbol, df) -> list[dict]

Each dict conforms to the service contract with keys:
    pattern_name, pattern_category, status, direction, confidence,
    breakout_level, invalidation_level, target, points
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper indicators
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, 1e-9))


def _atr(df: pd.DataFrame, n: int = 14) -> float:
    tr = (df["high"].astype(float) - df["low"].astype(float)).tail(n)
    return float(tr.mean()) if not tr.empty else 0.0


def _bb(close: pd.Series, n: int = 20, std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (lower, middle, upper) Bollinger Bands."""
    ma = close.rolling(n).mean()
    sig = close.rolling(n).std()
    return ma - std * sig, ma, ma + std * sig


def _stoch(df: pd.DataFrame, k: int = 14, d: int = 3) -> tuple[pd.Series, pd.Series]:
    lo = df["low"].astype(float).rolling(k).min()
    hi = df["high"].astype(float).rolling(k).max()
    k_pct = 100 * (df["close"].astype(float) - lo) / (hi - lo + 1e-9)
    d_pct = k_pct.rolling(d).mean()
    return k_pct, d_pct


def _cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    typical = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3
    sma = typical.rolling(n).mean()
    mean_dev = typical.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (typical - sma) / (0.015 * mean_dev.replace(0, 1e-9))


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (macd_line, signal_line, histogram)."""
    fast = _ema(close, 12)
    slow = _ema(close, 26)
    macd_line = fast - slow
    signal = _ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist


# ---------------------------------------------------------------------------
# Status helper
# ---------------------------------------------------------------------------

def _age_status(age: int) -> str:
    """Convert bar age (0 = current) to status label."""
    if age == 0:
        return "FORMING"
    if 1 <= age <= 3:
        return "READY"
    return "FORMING"


def _make_result(
    pattern_name: str,
    direction: str,
    confidence: float,
    status: str,
    points: list[list[Any]],
    breakout_level: float | None = None,
    invalidation_level: float | None = None,
    target: float | None = None,
) -> dict[str, Any]:
    return {
        "pattern_name": pattern_name,
        "pattern_category": "strategy",
        "status": status,
        "direction": direction,
        "confidence": round(float(np.clip(confidence, 0.0, 100.0)), 2),
        "breakout_level": breakout_level,
        "invalidation_level": invalidation_level,
        "target": target,
        "points": points,
    }


# ---------------------------------------------------------------------------
# Individual strategy detectors
# ---------------------------------------------------------------------------

def _detect_bb_squeeze(
    df: pd.DataFrame,
    close: pd.Series,
    atr: float,
) -> list[dict[str, Any]]:
    """Bollinger Band Squeeze (BB inside Keltner Channels)."""
    results: list[dict[str, Any]] = []
    if len(df) < 25:
        return results

    bb_lower, bb_mid, bb_upper = _bb(close, n=20, std=2.0)
    ema20 = _ema(close, 20)

    # ATR14 series for Keltner
    high_s = df["high"].astype(float)
    low_s = df["low"].astype(float)
    tr_s = (high_s - low_s)
    atr14_s = tr_s.rolling(14).mean()
    kc_lower = ema20 - 1.5 * atr14_s
    kc_upper = ema20 + 1.5 * atr14_s

    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)

    # Scan last 15 bars for a squeeze release
    scan_start = max(0, len(df) - 15)
    for i in range(len(df) - 1, scan_start - 1, -1):
        # Squeeze must be OFF at bar i (release)
        if squeeze_on.iloc[i]:
            continue
        # Count consecutive ON bars ending at i-1
        count = 0
        j = i - 1
        while j >= 0 and squeeze_on.iloc[j]:
            count += 1
            j -= 1
        if count < 5:
            continue

        # Found a squeeze release
        first_squeeze_idx = j + 1  # first bar of the squeeze run
        release_idx = i
        age = (len(df) - 1) - release_idx

        cur_close = float(close.iloc[i])
        mid_val = float(bb_mid.iloc[i]) if not pd.isna(bb_mid.iloc[i]) else cur_close
        direction = "bullish" if cur_close > mid_val else "bearish"

        conf = min(72 + 2 * count, 88)
        status = _age_status(age)

        if direction == "bullish":
            tgt = cur_close + 2 * atr
            inv = cur_close - atr
        else:
            tgt = cur_close - 2 * atr
            inv = cur_close + atr

        pts = [
            [int(first_squeeze_idx), float(close.iloc[first_squeeze_idx])],
            [int(release_idx), cur_close],
        ]
        results.append(_make_result(
            "BB Squeeze Breakout", direction, conf, status, pts,
            breakout_level=cur_close,
            invalidation_level=inv,
            target=tgt,
        ))
        break  # one per scan

    return results


def _detect_rsi2(
    df: pd.DataFrame,
    close: pd.Series,
) -> list[dict[str, Any]]:
    """RSI(2) short-term mean reversion strategy."""
    results: list[dict[str, Any]] = []
    if len(df) < 202:
        return results

    rsi2 = _rsi(close, n=2).fillna(50)
    ma200 = close.rolling(200).mean()

    scan_start = max(0, len(df) - 15)
    found: dict[str, tuple[int, str, float]] = {}

    for i in range(scan_start, len(df)):
        r = float(rsi2.iloc[i])
        c = float(close.iloc[i])
        m200 = float(ma200.iloc[i]) if not pd.isna(ma200.iloc[i]) else 0.0
        if m200 == 0.0:
            continue

        if r < 10 and c > m200:
            conf = min(70 + (10 - r), 88)
            age = (len(df) - 1) - i
            status = "BREAKOUT" if age == 0 else _age_status(age)
            found["RSI(2) Oversold Setup"] = (i, "bullish", conf)

        if r > 90 and c > m200:
            conf = min(70 + (r - 90), 88)
            age = (len(df) - 1) - i
            status = "BREAKOUT" if age == 0 else _age_status(age)
            found["RSI(2) Overbought Exit"] = (i, "bearish", conf)

    for name, (idx, direction, conf) in found.items():
        age = (len(df) - 1) - idx
        status = "BREAKOUT" if age == 0 else _age_status(age)
        cur_close = float(close.iloc[idx])
        results.append(_make_result(
            name, direction, conf, status,
            points=[[int(idx), cur_close]],
        ))
    return results


def _detect_stochastic_pop(
    df: pd.DataFrame,
    close: pd.Series,
) -> list[dict[str, Any]]:
    """Stochastic Pop (cross above 80) and Stochastic Drop (cross below 20)."""
    results: list[dict[str, Any]] = []
    if len(df) < 20:
        return results

    k, _d = _stoch(df)

    scan_start = max(1, len(df) - 15)
    found_pop: tuple[int, float] | None = None
    found_drop: tuple[int, float] | None = None

    for i in range(scan_start, len(df)):
        ki = float(k.iloc[i]) if not pd.isna(k.iloc[i]) else 50.0
        ki_prev = float(k.iloc[i - 1]) if not pd.isna(k.iloc[i - 1]) else 50.0

        # Stochastic Pop: K crosses above 80
        if ki > 80 and ki_prev <= 80:
            found_pop = (i, ki)

        # Stochastic Drop: K crosses below 20
        if ki < 20 and ki_prev >= 20:
            found_drop = (i, ki)

    if found_pop:
        idx, ki_val = found_pop
        conf = float(np.clip(65 + min(ki_val - 80, 15), 0, 100))
        age = (len(df) - 1) - idx
        status = _age_status(age)
        results.append(_make_result(
            "Stochastic Pop", "bullish", conf, status,
            points=[[int(idx), float(close.iloc[idx])]],
        ))

    if found_drop:
        idx, ki_val = found_drop
        conf = float(np.clip(65 + min(20 - ki_val, 15), 0, 100))
        age = (len(df) - 1) - idx
        status = _age_status(age)
        results.append(_make_result(
            "Stochastic Drop", "bearish", conf, status,
            points=[[int(idx), float(close.iloc[idx])]],
        ))

    return results


def _detect_cci_correction(
    df: pd.DataFrame,
    close: pd.Series,
) -> list[dict[str, Any]]:
    """CCI Correction Strategy — cross above -100 (bullish) or below +100 (bearish)."""
    results: list[dict[str, Any]] = []
    if len(df) < 25:
        return results

    cci = _cci(df, n=20)

    scan_start = max(1, len(df) - 15)
    found_bull: int | None = None
    found_bear: int | None = None

    for i in range(scan_start, len(df)):
        c_now = float(cci.iloc[i]) if not pd.isna(cci.iloc[i]) else 0.0
        c_prev = float(cci.iloc[i - 1]) if not pd.isna(cci.iloc[i - 1]) else 0.0

        # Cross above -100 from below
        if c_now > -100 and c_prev <= -100:
            found_bull = i

        # Cross below +100 from above
        if c_now < 100 and c_prev >= 100:
            found_bear = i

    if found_bull is not None:
        age = (len(df) - 1) - found_bull
        status = _age_status(age)
        results.append(_make_result(
            "CCI Correction Bullish", "bullish", 68.0, status,
            points=[[int(found_bull), float(close.iloc[found_bull])]],
        ))

    if found_bear is not None:
        age = (len(df) - 1) - found_bear
        status = _age_status(age)
        results.append(_make_result(
            "CCI Correction Bearish", "bearish", 68.0, status,
            points=[[int(found_bear), float(close.iloc[found_bear])]],
        ))

    return results


def _detect_ma_strategies(
    df: pd.DataFrame,
    close: pd.Series,
    atr: float,
) -> list[dict[str, Any]]:
    """Golden Cross, Death Cross, MA Support/Resistance signals."""
    results: list[dict[str, Any]] = []
    if len(df) < 55:
        return results

    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean() if len(df) >= 200 else pd.Series(
        [float("nan")] * len(close), index=close.index
    )

    last_close = float(close.iloc[-1])
    last_idx = len(df) - 1

    # --- Golden Cross / Death Cross (within last 5 bars) ---
    cross_lookback = min(5, len(df) - 1)
    for i in range(len(df) - cross_lookback - 1, len(df) - 1):
        if i < 1:
            continue
        m50_now = float(ma50.iloc[i + 1]) if not pd.isna(ma50.iloc[i + 1]) else None
        m50_prev = float(ma50.iloc[i]) if not pd.isna(ma50.iloc[i]) else None
        m200_now = float(ma200.iloc[i + 1]) if not pd.isna(ma200.iloc[i + 1]) else None
        m200_prev = float(ma200.iloc[i]) if not pd.isna(ma200.iloc[i]) else None

        if None in (m50_now, m50_prev, m200_now, m200_prev):
            continue

        age = last_idx - (i + 1)

        # Golden Cross
        if m50_now > m200_now and m50_prev <= m200_prev:
            status = _age_status(age)
            results.append(_make_result(
                "Golden Cross Setup", "bullish", 78.0, status,
                points=[[int(i + 1), float(close.iloc[i + 1])]],
                breakout_level=m50_now,
                target=float(close.iloc[i + 1]) + 2 * atr,
            ))

        # Death Cross
        if m50_now < m200_now and m50_prev >= m200_prev:
            status = _age_status(age)
            results.append(_make_result(
                "Death Cross Setup", "bearish", 78.0, status,
                points=[[int(i + 1), float(close.iloc[i + 1])]],
                breakout_level=m50_now,
                target=float(close.iloc[i + 1]) - 2 * atr,
            ))

    # --- MA Support / Resistance (current bar) ---
    m50_last = float(ma50.iloc[-1]) if not pd.isna(ma50.iloc[-1]) else None
    m200_last = float(ma200.iloc[-1]) if not pd.isna(ma200.iloc[-1]) else None
    open_last = float(df["open"].astype(float).iloc[-1])

    if m50_last is not None:
        pct_dist_50 = abs(last_close - m50_last) / max(m50_last, 1e-9)
        # MA50 Support Bounce: price touched MA50, now above it, bullish candle
        if pct_dist_50 < 0.005 and last_close > open_last and last_close > m50_last:
            results.append(_make_result(
                "MA50 Support Bounce", "bullish", 68.0, "READY",
                points=[[last_idx, last_close]],
                breakout_level=m50_last,
                target=last_close + 2 * atr,
                invalidation_level=m50_last - atr,
            ))
        # MA50 Resistance Rejection: price hit MA50 from below, rejected (bearish candle below MA50)
        elif pct_dist_50 < 0.005 and last_close < open_last and last_close < m50_last:
            results.append(_make_result(
                "MA50 Resistance Rejection", "bearish", 68.0, "READY",
                points=[[last_idx, last_close]],
                breakout_level=m50_last,
                target=last_close - 2 * atr,
                invalidation_level=m50_last + atr,
            ))

    if m200_last is not None:
        pct_dist_200 = abs(last_close - m200_last) / max(m200_last, 1e-9)
        if pct_dist_200 < 0.005 and last_close > open_last and last_close > m200_last:
            results.append(_make_result(
                "MA200 Support Bounce", "bullish", 72.0, "READY",
                points=[[last_idx, last_close]],
                breakout_level=m200_last,
                target=last_close + 2 * atr,
                invalidation_level=m200_last - atr,
            ))
        elif pct_dist_200 < 0.005 and last_close < open_last and last_close < m200_last:
            results.append(_make_result(
                "MA200 Resistance Rejection", "bearish", 72.0, "READY",
                points=[[last_idx, last_close]],
                breakout_level=m200_last,
                target=last_close - 2 * atr,
                invalidation_level=m200_last + atr,
            ))

    return results


def _detect_moving_momentum(
    df: pd.DataFrame,
    close: pd.Series,
) -> list[dict[str, Any]]:
    """Moving Momentum — ROC-based signals."""
    results: list[dict[str, Any]] = []
    if len(df) < 15:
        return results

    roc10 = (close / close.shift(10).replace(0, float("nan")) - 1) * 100

    scan_start = max(1, len(df) - 15)
    found: dict[str, tuple[int, str, float]] = {}

    for i in range(scan_start, len(df)):
        r_now = float(roc10.iloc[i]) if not pd.isna(roc10.iloc[i]) else 0.0
        r_prev = float(roc10.iloc[i - 1]) if not pd.isna(roc10.iloc[i - 1]) else 0.0

        # Cross above 0
        if r_now > 0 and r_prev <= 0:
            found.setdefault("Positive Momentum Cross", (i, "bullish", 65.0))

        # Cross below 0
        if r_now < 0 and r_prev >= 0:
            found.setdefault("Negative Momentum Cross", (i, "bearish", 65.0))

    # Strong Upward Momentum: ROC > 5% for 3+ consecutive bars
    consecutive = 0
    strong_start: int | None = None
    for i in range(max(0, len(df) - 15), len(df)):
        r = float(roc10.iloc[i]) if not pd.isna(roc10.iloc[i]) else 0.0
        if r > 5.0:
            if consecutive == 0:
                strong_start = i
            consecutive += 1
        else:
            consecutive = 0
            strong_start = None
    if consecutive >= 3 and strong_start is not None:
        found.setdefault("Strong Upward Momentum", (len(df) - 1, "bullish", 70.0))

    for name, (idx, direction, conf) in found.items():
        age = (len(df) - 1) - idx
        status = _age_status(age)
        results.append(_make_result(
            name, direction, conf, status,
            points=[[int(idx), float(close.iloc[idx])]],
        ))

    return results


def _detect_nr7(
    df: pd.DataFrame,
    close: pd.Series,
    atr: float,
) -> list[dict[str, Any]]:
    """NR7 — today's range is the narrowest of the last 7 bars."""
    results: list[dict[str, Any]] = []
    if len(df) < 8:
        return results

    high_s = df["high"].astype(float)
    low_s = df["low"].astype(float)
    rng = high_s - low_s

    scan_start = max(6, len(df) - 15)
    found: tuple[int, str, float] | None = None

    rsi14 = _rsi(close, n=14).fillna(50)

    for i in range(scan_start, len(df)):
        today_range = float(rng.iloc[i])
        window_ranges = rng.iloc[i - 6: i + 1]
        if today_range != float(window_ranges.min()):
            continue

        # NR7 found
        r = float(rsi14.iloc[i]) if not pd.isna(rsi14.iloc[i]) else 50.0
        if r > 50:
            direction, conf = "bullish", 68.0
        elif r < 50:
            direction, conf = "bearish", 68.0
        else:
            direction, conf = "neutral", 65.0

        found = (i, direction, conf)

    if found is not None:
        idx, direction, conf = found
        age = (len(df) - 1) - idx
        status = _age_status(age)
        cur_close = float(close.iloc[idx])
        tgt = cur_close + 2 * atr if direction == "bullish" else cur_close - 2 * atr if direction == "bearish" else None
        results.append(_make_result(
            "NR7 Compression", direction, conf, status,
            points=[[int(idx), cur_close]],
            target=tgt,
            invalidation_level=cur_close - atr if direction == "bullish" else cur_close + atr if direction == "bearish" else None,
        ))

    return results


def _detect_gap_patterns(
    df: pd.DataFrame,
    close: pd.Series,
) -> list[dict[str, Any]]:
    """Gap Up / Gap Down and Gap Fill Reversal."""
    results: list[dict[str, Any]] = []
    if len(df) < 3:
        return results

    open_s = df["open"].astype(float)
    scan_start = max(1, len(df) - 15)

    found: dict[str, tuple[int, str, float]] = {}

    for i in range(scan_start, len(df)):
        prev_close = float(close.iloc[i - 1])
        cur_open = float(open_s.iloc[i])
        cur_close = float(close.iloc[i])

        if prev_close <= 0:
            continue

        gap_pct = (cur_open - prev_close) / prev_close * 100

        if gap_pct > 1.0:
            # Gap Up
            conf = float(np.clip(65 + gap_pct * 10, 0, 85))
            found.setdefault("Gap Up", (i, "bullish", conf))
            # Gap Fill: closed below previous close
            if cur_close < prev_close:
                found["Gap Fill Reversal (from Gap Up)"] = (i, "bearish", 72.0)

        elif gap_pct < -1.0:
            # Gap Down
            conf = float(np.clip(65 + abs(gap_pct) * 10, 0, 85))
            found.setdefault("Gap Down", (i, "bearish", conf))
            # Gap Fill: closed above previous close
            if cur_close > prev_close:
                found["Gap Fill Reversal (from Gap Down)"] = (i, "bullish", 72.0)

    for name, (idx, direction, conf) in found.items():
        age = (len(df) - 1) - idx
        status = _age_status(age)
        results.append(_make_result(
            name, direction, conf, status,
            points=[[int(idx), float(close.iloc[idx])]],
        ))

    return results


def _detect_ichimoku(
    df: pd.DataFrame,
    close: pd.Series,
    atr: float,
) -> list[dict[str, Any]]:
    """Ichimoku Cloud signals."""
    results: list[dict[str, Any]] = []
    if len(df) < 53:
        return results

    high_s = df["high"].astype(float)
    low_s = df["low"].astype(float)

    tenkan = (high_s.rolling(9).max() + low_s.rolling(9).min()) / 2
    kijun = (high_s.rolling(26).max() + low_s.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high_s.rolling(52).max() + low_s.rolling(52).min()) / 2).shift(26)
    # chikou: close shifted back 26 bars — use close 26 bars ago as proxy for current bar
    chikou = close.shift(26)  # not used in signals below but computed for completeness

    last_idx = len(df) - 1
    last_close = float(close.iloc[-1])

    def safe(s: pd.Series, i: int) -> float | None:
        v = s.iloc[i] if 0 <= i < len(s) else float("nan")
        return None if pd.isna(v) else float(v)

    found: dict[str, tuple[int, str, float]] = {}

    scan_start = max(1, len(df) - 15)

    for i in range(scan_start, len(df)):
        tk_now = safe(tenkan, i)
        tk_prev = safe(tenkan, i - 1)
        kj_now = safe(kijun, i)
        kj_prev = safe(kijun, i - 1)
        sa = safe(senkou_a, i)
        sb = safe(senkou_b, i)
        c = float(close.iloc[i])

        if None in (tk_now, tk_prev, kj_now, kj_prev):
            continue

        age = last_idx - i

        # TK Cross Bullish
        if tk_now > kj_now and tk_prev <= kj_prev:  # type: ignore[operator]
            found.setdefault("Ichimoku TK Cross Bullish", (i, "bullish", 70.0))

        # TK Cross Bearish
        if tk_now < kj_now and tk_prev >= kj_prev:  # type: ignore[operator]
            found.setdefault("Ichimoku TK Cross Bearish", (i, "bearish", 70.0))

        # Price above / below cloud
        if sa is not None and sb is not None:
            cloud_top = max(sa, sb)
            cloud_bot = min(sa, sb)

            if c > cloud_top:
                found["Above Ichimoku Cloud"] = (i, "bullish", 72.0)
            elif c < cloud_bot:
                found["Below Ichimoku Cloud"] = (i, "bearish", 72.0)

            # Cloud Breakout: price just crossed above cloud from below
            if i >= 1:
                c_prev = float(close.iloc[i - 1])
                sa_prev = safe(senkou_a, i - 1)
                sb_prev = safe(senkou_b, i - 1)
                if sa_prev is not None and sb_prev is not None:
                    prev_cloud_top = max(sa_prev, sb_prev)
                    if c > cloud_top and c_prev <= prev_cloud_top:
                        found["Ichimoku Cloud Breakout"] = (i, "bullish", 78.0)

        # Kijun Bounce: price within 0.5% of kijun
        if kj_now is not None:
            pct = abs(c - kj_now) / max(abs(kj_now), 1e-9)  # type: ignore[arg-type]
            if pct < 0.005:
                found.setdefault("Kijun Support Bounce", (i, "bullish", 68.0))

    for name, (idx, direction, conf) in found.items():
        age = last_idx - idx
        status = _age_status(age)
        cur_close = float(close.iloc[idx])
        results.append(_make_result(
            name, direction, conf, status,
            points=[[int(idx), cur_close]],
            target=cur_close + 2 * atr if direction == "bullish" else cur_close - 2 * atr,
            invalidation_level=cur_close - atr if direction == "bullish" else cur_close + atr,
        ))

    return results


def _detect_swing_charting(
    df: pd.DataFrame,
    close: pd.Series,
) -> list[dict[str, Any]]:
    """Swing Charting — HH/HL/LL/LH pattern detection."""
    results: list[dict[str, Any]] = []
    if len(df) < 20:
        return results

    high_s = df["high"].astype(float).values
    low_s = df["low"].astype(float).values
    n = len(high_s)

    # Find swing points using a simple local extremum approach (window=3)
    swings: list[tuple[int, float, str]] = []  # (idx, price, 'H' or 'L')
    for i in range(1, n - 1):
        if high_s[i] > high_s[i - 1] and high_s[i] > high_s[i + 1]:
            swings.append((i, high_s[i], "H"))
        elif low_s[i] < low_s[i - 1] and low_s[i] < low_s[i + 1]:
            swings.append((i, low_s[i], "L"))

    # Deduplicate consecutive same-type swings, keep the most extreme
    deduped: list[tuple[int, float, str]] = []
    for s in swings:
        if deduped and deduped[-1][2] == s[2]:
            prev = deduped[-1]
            if s[2] == "H" and s[1] > prev[1]:
                deduped[-1] = s
            elif s[2] == "L" and s[1] < prev[1]:
                deduped[-1] = s
        else:
            deduped.append(s)

    if len(deduped) < 3:
        return results

    # Analyse last 3 swing points (must be within last 30 bars)
    recent = [s for s in deduped if s[0] >= n - 30]
    if len(recent) < 3:
        recent = deduped[-3:]

    s1, s2, s3 = recent[-3], recent[-2], recent[-1]
    age = (len(df) - 1) - s3[0]
    if age > 15:
        return results

    pts = [[int(s1[0]), float(s1[1])], [int(s2[0]), float(s2[1])], [int(s3[0]), float(s3[1])]]
    status = _age_status(age)

    # HH → HL → HH: Uptrend continuation
    if s1[2] == "H" and s2[2] == "L" and s3[2] == "H":
        if s3[1] > s1[1] and s2[1] > (s1[1] - (s1[1] - s2[1]) * 0.5):
            results.append(_make_result(
                "Swing Uptrend Continuation", "bullish", 67.0, status, pts,
            ))

    # LL → LH → LL: Downtrend continuation
    elif s1[2] == "L" and s2[2] == "H" and s3[2] == "L":
        if s3[1] < s1[1]:
            results.append(_make_result(
                "Swing Downtrend Continuation", "bearish", 67.0, status, pts,
            ))

    # LL → HL (reversal signal): prior low, now higher low forming
    elif s1[2] == "L" and s2[2] == "H" and s3[2] == "L" and s3[1] > s1[1]:
        results.append(_make_result(
            "Swing Low Reversal Signal", "bullish", 65.0, status, pts,
        ))

    # Explicit check for HL after LL
    if len(deduped) >= 4:
        s0 = deduped[-4]
        if s0[2] == "L" and s1[2] == "H" and s2[2] == "L" and s3[2] == "H":
            if s2[1] > s0[1]:  # Higher low
                age2 = (len(df) - 1) - s3[0]
                if age2 <= 15:
                    pts2 = [
                        [int(s0[0]), float(s0[1])],
                        [int(s1[0]), float(s1[1])],
                        [int(s2[0]), float(s2[1])],
                        [int(s3[0]), float(s3[1])],
                    ]
                    results.append(_make_result(
                        "Swing Low Reversal Signal", "bullish", 65.0, _age_status(age2), pts2,
                    ))

    return results


def _detect_sector_rotation(
    df: pd.DataFrame,
    close: pd.Series,
) -> list[dict[str, Any]]:
    """Sector Rotation / Relative Strength proxy."""
    results: list[dict[str, Any]] = []
    if len(df) < 25:
        return results

    last = float(close.iloc[-1])
    c20 = float(close.iloc[-21]) if len(df) > 20 else None
    c5 = float(close.iloc[-6]) if len(df) > 5 else None

    if c20 is None or c5 is None or c20 <= 0 or c5 <= 0:
        return results

    roc20 = (last / c20 - 1) * 100
    roc5 = (last / c5 - 1) * 100
    last_idx = len(df) - 1

    signals: list[tuple[str, str, float]] = []

    if roc20 > 10 and roc5 > 2:
        signals.append(("Momentum Leader", "bullish", 68.0))
    if roc20 < -10 and roc5 < -2:
        signals.append(("Laggard Breakdown", "bearish", 68.0))
    if roc20 < -5 and roc5 > 1:
        signals.append(("Sector Rotation In", "bullish", 65.0))
    if roc20 > 5 and roc5 < -1:
        signals.append(("Sector Rotation Out", "bearish", 65.0))

    for name, direction, conf in signals:
        results.append(_make_result(
            name, direction, conf, "READY",
            points=[[last_idx, last]],
        ))

    return results


def _detect_elder_impulse(
    df: pd.DataFrame,
    close: pd.Series,
) -> list[dict[str, Any]]:
    """Elder Impulse System — EMA13 slope + MACD histogram direction."""
    results: list[dict[str, Any]] = []
    if len(df) < 30:
        return results

    ema13 = _ema(close, 13)
    _macd_line, _signal, hist = _macd(close)

    if len(ema13) < 2 or len(hist) < 2:
        return results

    e_now = float(ema13.iloc[-1]) if not pd.isna(ema13.iloc[-1]) else 0.0
    e_prev = float(ema13.iloc[-2]) if not pd.isna(ema13.iloc[-2]) else 0.0
    h_now = float(hist.iloc[-1]) if not pd.isna(hist.iloc[-1]) else 0.0
    h_prev = float(hist.iloc[-2]) if not pd.isna(hist.iloc[-2]) else 0.0

    ema_up = e_now > e_prev
    ema_down = e_now < e_prev
    hist_up = h_now > h_prev
    hist_down = h_now < h_prev

    last_idx = len(df) - 1
    last_close = float(close.iloc[-1])

    if ema_up and hist_up:
        results.append(_make_result(
            "Elder Impulse Bullish", "bullish", 68.0, "READY",
            points=[[last_idx, last_close]],
        ))
    elif ema_down and hist_down:
        results.append(_make_result(
            "Elder Impulse Bearish", "bearish", 68.0, "READY",
            points=[[last_idx, last_close]],
        ))

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(symbol: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect strategy patterns in the given OHLCV DataFrame.

    Parameters
    ----------
    symbol:
        Ticker symbol (used for logging / context only).
    df:
        DataFrame with columns: date, open, high, low, close, volume.
        Must have at least 50 rows; returns [] otherwise.

    Returns
    -------
    list[dict]
        Each dict contains: pattern_name, pattern_category, status, direction,
        confidence, breakout_level, invalidation_level, target, points.
    """
    try:
        if df is None or len(df) < 50:
            return []

        # Ensure numeric types; reset index to positional integers
        df = df.copy()
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.reset_index(drop=True)

        close = df["close"].astype(float)
        atr = _atr(df, n=14)

        results: list[dict[str, Any]] = []

        results.extend(_detect_bb_squeeze(df, close, atr))
        results.extend(_detect_rsi2(df, close))
        results.extend(_detect_stochastic_pop(df, close))
        results.extend(_detect_cci_correction(df, close))
        results.extend(_detect_ma_strategies(df, close, atr))
        results.extend(_detect_moving_momentum(df, close))
        results.extend(_detect_nr7(df, close, atr))
        results.extend(_detect_gap_patterns(df, close))
        results.extend(_detect_ichimoku(df, close, atr))
        results.extend(_detect_swing_charting(df, close))
        results.extend(_detect_sector_rotation(df, close))
        results.extend(_detect_elder_impulse(df, close))

        # Filter out NOT_PRESENT (only return actionable signals)
        results = [r for r in results if r.get("status", "NOT_PRESENT") != "NOT_PRESENT"]

        return results

    except Exception:  # noqa: BLE001
        return []
