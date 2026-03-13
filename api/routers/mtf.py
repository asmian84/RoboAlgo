"""
RoboAlgo - Multi-Timeframe Confidence Analysis
Computes technical confidence scores across 8 timeframes for any symbol.
Downloads intraday data on-demand via yfinance (no DB storage).
"""

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter

from config.settings import (
    INDEX_DRIVERS,
    LEVERAGED_ETF_PAIRS,
    UNDERLYING_LEADERS,
)

router = APIRouter()

# ── Build ETF → underlying stock ticker mapping ───────────────────────────────
_ETF_TO_UNDERLYING: dict[str, str] = {}

for _bull, _bear, _desc, *_rest in LEVERAGED_ETF_PAIRS:
    if _rest and _rest[0]:
        _u = _rest[0]
        _ETF_TO_UNDERLYING[_bull] = _u
        if _bear:
            _ETF_TO_UNDERLYING[_bear] = _u

for _s in INDEX_DRIVERS + UNDERLYING_LEADERS:
    _ETF_TO_UNDERLYING[_s] = _s

# ── Timeframe config ──────────────────────────────────────────────────────────
TIMEFRAMES = [
    {"name": "1m",   "interval": "1m",   "period": "7d",   "resample": None,  "bars": 40},
    {"name": "5m",   "interval": "5m",   "period": "60d",  "resample": None,  "bars": 40},
    {"name": "15m",  "interval": "15m",  "period": "60d",  "resample": None,  "bars": 40},
    {"name": "30m",  "interval": "30m",  "period": "60d",  "resample": None,  "bars": 40},
    {"name": "1h",   "interval": "1h",   "period": "730d", "resample": None,  "bars": 40},
    {"name": "2h",   "interval": "1h",   "period": "730d", "resample": "2h",  "bars": 40},
    {"name": "4h",   "interval": "1h",   "period": "730d", "resample": "4h",  "bars": 40},
    {"name": "1d",   "interval": "1d",   "period": "max",  "resample": None,  "bars": 60},
    {"name": "1w",   "interval": "1wk",  "period": "max",  "resample": None,  "bars": 52},
    {"name": "1M",   "interval": "1mo",  "period": "max",  "resample": None,  "bars": 24},
]


def _get_underlying(symbol: str) -> str:
    return _ETF_TO_UNDERLYING.get(symbol.upper(), symbol.upper())


def _download(symbol: str, interval: str, period: str) -> pd.DataFrame | None:
    try:
        df = yf.Ticker(symbol).history(interval=interval, period=period, auto_adjust=True)
        if df.empty:
            return None
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        date_col = "datetime" if "datetime" in df.columns else "date"
        df = df.rename(columns={date_col: "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        cols = [c for c in ["datetime", "open", "high", "low", "close", "volume"] if c in df.columns]
        return df[cols].dropna(subset=["close"])
    except Exception:
        return None


def _resample_df(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    df = df.set_index("datetime")
    r = df.resample(freq).agg({"open": "first", "high": "max", "low": "min",
                                "close": "last", "volume": "sum"}).dropna()
    return r.reset_index()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(series: pd.Series, fast=12, slow=26, sig=9):
    ef = series.ewm(span=fast, adjust=False).mean()
    es = series.ewm(span=slow, adjust=False).mean()
    macd = ef - es
    signal = macd.ewm(span=sig, adjust=False).mean()
    return macd, signal, macd - signal


def _bb(series: pd.Series, period=20, std=2.0):
    ma = series.rolling(period).mean()
    sd = series.rolling(period).std()
    upper = ma + std * sd
    lower = ma - std * sd
    pos = (series - lower) / (upper - lower).replace(0, np.nan)
    return upper, ma, lower, pos.clip(0, 1)


def _score(df: pd.DataFrame, bars: int) -> dict:
    """Compute bullish confidence score 0-100 from technical indicators."""
    if df is None or len(df) < bars:
        return {"confidence": None, "signal": "no_data", "details": {}}

    close = df["close"].astype(float)

    # RSI: lower RSI → more bullish opportunity
    rsi_s = _rsi(close)
    rsi_val = float(rsi_s.iloc[-1]) if not pd.isna(rsi_s.iloc[-1]) else 50.0
    rsi_score = max(0.0, min(1.0, (70.0 - rsi_val) / 60.0))

    # MACD histogram: positive → bullish
    _, _, hist = _macd(close)
    hist_val = float(hist.iloc[-1]) if not pd.isna(hist.iloc[-1]) else 0.0
    hist_std = float(hist.std()) if hist.std() > 0 else 1.0
    macd_score = max(0.0, min(1.0, 0.5 + (hist_val / hist_std) * 0.25))

    # Bollinger Bands: near lower band → bullish
    bb_upper_s, bb_mid_s, bb_lower_s, bb_pos = _bb(close)
    bb_val    = float(bb_pos.iloc[-1])     if not pd.isna(bb_pos.iloc[-1])     else 0.5
    bb_lower  = float(bb_lower_s.iloc[-1]) if not pd.isna(bb_lower_s.iloc[-1]) else None
    bb_mid    = float(bb_mid_s.iloc[-1])   if not pd.isna(bb_mid_s.iloc[-1])   else None
    bb_upper  = float(bb_upper_s.iloc[-1]) if not pd.isna(bb_upper_s.iloc[-1]) else None
    bb_score  = 1.0 - bb_val

    # MA trend: above short & long MAs → bullish
    n = len(close)
    ma20_s = close.rolling(min(20, n // 3)).mean()
    ma50_s = close.rolling(min(50, n // 2)).mean()
    ma20   = float(ma20_s.iloc[-1]) if not pd.isna(ma20_s.iloc[-1]) else None
    ma50   = float(ma50_s.iloc[-1]) if not pd.isna(ma50_s.iloc[-1]) else None
    cur    = float(close.iloc[-1])
    ma_scores = []
    if ma20 is not None:
        ma_scores.append(1.0 if cur > ma20 else 0.0)
    if ma50 is not None:
        ma_scores.append(1.0 if cur > ma50 else 0.0)
    ma_score = float(np.mean(ma_scores)) if ma_scores else 0.5

    # Momentum: 5-bar return
    nb = min(5, n - 1)
    ret = (cur - float(close.iloc[-nb - 1])) / float(close.iloc[-nb - 1]) if nb > 0 else 0.0
    mom_score = max(0.0, min(1.0, 0.5 + ret * 5.0))

    # ATR (14-period true range)
    atr_val = None
    if "high" in df.columns and "low" in df.columns:
        high = df["high"].astype(float)
        low  = df["low"].astype(float)
        tr   = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr_s   = tr.rolling(14).mean()
        atr_val = float(atr_s.iloc[-1]) if len(atr_s) >= 14 and not pd.isna(atr_s.iloc[-1]) else None

    # Weighted combination
    confidence = (
        0.25 * rsi_score +
        0.25 * macd_score +
        0.20 * bb_score +
        0.20 * ma_score +
        0.10 * mom_score
    ) * 100.0

    signal = "bullish" if confidence >= 60 else ("bearish" if confidence <= 40 else "neutral")

    return {
        "confidence": round(confidence, 1),
        "signal": signal,
        "details": {
            "rsi":        round(rsi_val, 1),
            "macd_hist":  round(hist_val, 5),
            "bb_position": round(bb_val, 3),
            "above_ma":   round(ma_score, 2),
            "momentum":   round(mom_score, 2),
            # Price levels for horizon trade plans
            "last_price": round(cur, 4),
            "bb_lower":   round(bb_lower, 4) if bb_lower is not None else None,
            "bb_mid":     round(bb_mid,   4) if bb_mid   is not None else None,
            "bb_upper":   round(bb_upper, 4) if bb_upper is not None else None,
            "ma20":       round(ma20,     4) if ma20     is not None else None,
            "ma50":       round(ma50,     4) if ma50     is not None else None,
            "atr":        round(atr_val,  4) if atr_val  is not None else None,
        },
    }


@router.get("/{symbol}")
def get_mtf(symbol: str):
    """Multi-timeframe confidence for a symbol (uses underlying stock ticker)."""
    symbol = symbol.upper()
    underlying = _get_underlying(symbol)

    results = []
    for tf in TIMEFRAMES:
        df = _download(underlying, tf["interval"], tf["period"])
        if df is not None and tf["resample"]:
            df = _resample_df(df, tf["resample"])
        scored = _score(df, tf["bars"])
        results.append({"timeframe": tf["name"], **scored})

    return {"symbol": symbol, "underlying": underlying, "timeframes": results}
