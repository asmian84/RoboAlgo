"""
Price data and combined chart endpoints.

For symbols that exist in the database the DB path is used (fast, pre-computed
indicators).  For any other ticker a yfinance fallback fetches full daily history
(period="max", back to IPO / 1980s) and computes all indicators on-the-fly so
that ANY publicly-traded stock, ETF, index, or crypto pair works without needing
to be in the instruments table.
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from data_engine.loader import DataLoader

router = APIRouter()
loader = DataLoader()


# ── Helpers ───────────────────────────────────────────────────────────────────

def df_to_records(df: pd.DataFrame, limit: int = 500) -> list[dict]:
    """Convert a date-indexed DataFrame to JSON-safe list of dicts (NaN → null)."""
    if df.empty:
        return []
    # limit=0 means no cap — return all rows
    df = (df if limit == 0 else df.tail(limit)).reset_index()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


def _compute_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all standard RoboAlgo indicators from OHLCV prices in-memory.
    Returns a DataFrame with the same index as `prices`.

    Columns produced match the Indicator model:
      rsi, atr, macd_line, macd_signal, macd_histogram,
      bb_upper, bb_middle, bb_lower, bb_width, ma50, ma200
    """
    df = prices.copy()
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # ── RSI (14, Wilder / EWM form) ───────────────────────────────────────────
    delta     = close.diff()
    gain      = delta.clip(lower=0)
    loss      = (-delta).clip(lower=0)
    avg_gain  = gain.ewm(com=13, adjust=False).mean()
    avg_loss  = loss.ewm(com=13, adjust=False).mean()
    rs        = avg_gain / avg_loss.replace(0, float("nan"))
    df["rsi"] = 100 - 100 / (1 + rs)

    # ── ATR (14, EWM form) ────────────────────────────────────────────────────
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=14, adjust=False).mean()

    # ── MACD (12 / 26 / 9) ───────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line          = ema12 - ema26
    macd_signal        = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_line"]      = macd_line
    df["macd_signal"]    = macd_signal
    df["macd_histogram"] = macd_line - macd_signal

    # ── Bollinger Bands (20 / 2σ) ─────────────────────────────────────────────
    bb_mid          = close.rolling(20).mean()
    bb_std          = close.rolling(20).std(ddof=0)
    df["bb_upper"]  = bb_mid + 2 * bb_std
    df["bb_middle"] = bb_mid
    df["bb_lower"]  = bb_mid - 2 * bb_std
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / bb_mid.replace(0, float("nan"))

    # ── Moving Averages ────────────────────────────────────────────────────────
    df["ma50"]  = close.rolling(50).mean()
    df["ma200"] = close.rolling(200).mean()

    ind_cols = [
        "rsi", "atr",
        "macd_line", "macd_signal", "macd_histogram",
        "bb_upper", "bb_middle", "bb_lower", "bb_width",
        "ma50", "ma200",
    ]
    return df[ind_cols]


def _yfinance_fetch(symbol: str, limit: int = 500, yf_interval: str = "1d") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch OHLCV from yfinance for any bar resolution (1d / 1wk / 1mo).
    Returns (prices_df, indicators_df) both indexed by timezone-naive date.
    Returns (empty, empty) if the symbol is unknown or has no data.
    """
    try:
        import yfinance as yf

        if yf_interval == "1wk":
            period = "max"   # Full weekly history back to IPO
        elif yf_interval == "1mo":
            period = "max"   # Full monthly history back to IPO
        else:
            period = "max"  # Full daily history back to IPO

        hist   = yf.Ticker(symbol).history(period=period, interval=yf_interval, auto_adjust=True)

        if hist.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Strip timezone so strftime works uniformly
        if hist.index.tz is not None:
            hist.index = hist.index.tz_convert(None)

        hist.index.name = "date"
        hist = hist.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })

        prices     = hist[["open", "high", "low", "close", "volume"]].copy()
        indicators = _compute_indicators(prices)

        return prices, indicators

    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def _alphavantage_fetch(symbol: str, limit: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Alpha Vantage TIME_SERIES_DAILY fallback — covers some tickers yfinance misses
    (certain OTC, international ADRs, newer listings).
    Rate-limited: only called when yfinance returns empty.
    Returns (empty, empty) on failure or rate-limit hit.
    """
    try:
        import httpx

        AV_KEY  = "UPSSIQEAMIJFI6RS"
        AV_BASE = "https://www.alphavantage.co/query"

        with httpx.Client(timeout=10) as client:
            r = client.get(AV_BASE, params={
                "function":   "TIME_SERIES_DAILY_ADJUSTED",
                "symbol":     symbol,
                "outputsize": "full",
                "apikey":     AV_KEY,
            })

        data = r.json()
        ts   = data.get("Time Series (Daily)")
        if not ts:
            return pd.DataFrame(), pd.DataFrame()

        rows = []
        for date_str, vals in sorted(ts.items()):
            rows.append({
                "date":   pd.Timestamp(date_str),
                "open":   float(vals["1. open"]),
                "high":   float(vals["2. high"]),
                "low":    float(vals["3. low"]),
                "close":  float(vals["5. adjusted close"]),
                "volume": float(vals["6. volume"]),
            })

        if not rows:
            return pd.DataFrame(), pd.DataFrame()

        prices = pd.DataFrame(rows).set_index("date")
        prices.index.name = "date"

        indicators = _compute_indicators(prices)
        return prices, indicators

    except Exception:
        return pd.DataFrame(), pd.DataFrame()


# ── Endpoints ─────────────────────────────────────────────────────────────────

def _fetch_any(symbol: str, limit: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Data fetching strategy: Alpha Vantage first (full history) → yfinance → DB cache.

    For limit=0 (full-history / 'All' TF):
      - Alpha Vantage with outputsize="full" returns ALL available data back to IPO
      - Fallback to yfinance if AV is rate-limited

    For limit != 0 (partial history):
      - Use DB cache if available (fastest)
      - Fallback to yfinance, then Alpha Vantage
    """
    # Full-history requests: use Alpha Vantage (complete data) or yfinance as fallback
    if limit == 0:
        # 1. Alpha Vantage with full output (back to IPO)
        prices, indicators = _alphavantage_fetch(symbol, 0)
        if not prices.empty:
            return prices, indicators

        # 2. yfinance as fallback (if AV is rate-limited)
        prices, indicators = _yfinance_fetch(symbol, 0)
        return prices, indicators

    # Partial history: use DB cache if available (fast), then fallback
    # 1. Database (pre-computed, fastest)
    prices = loader.get_prices(symbol)
    if not prices.empty:
        indicators = loader.get_indicators(symbol)
        if indicators.empty:
            indicators = _compute_indicators(prices)
        return prices, indicators

    # 2. yfinance
    prices, indicators = _yfinance_fetch(symbol, limit)
    if not prices.empty:
        return prices, indicators

    # 3. Alpha Vantage
    prices, indicators = _alphavantage_fetch(symbol, limit)
    return prices, indicators


@router.get("/prices/{symbol}")
def get_prices(symbol: str, limit: int = 500):
    """OHLCV price data for a symbol — DB → yfinance → Alpha Vantage."""
    sym = symbol.upper()
    prices, _ = _fetch_any(sym, limit)
    if prices.empty:
        raise HTTPException(404, f"No price data for {symbol}")
    return df_to_records(prices, limit)


@router.get("/chart/{symbol}")
def get_chart_data(symbol: str, limit: int = 0, interval: str = "daily"):
    """
    Combined prices + indicators for charting.
    Works for ANY ticker via three-tier fallback:
      1. DB (pre-computed)  — daily only
      2. yfinance (all NYSE/Nasdaq/Russell/ETF/Crypto/Indices)
      3. Alpha Vantage (additional coverage for OTC/international)

    interval: "daily" | "weekly" | "monthly"
      - weekly / monthly bypass DB and fetch directly from yfinance so we get
        true weekly/monthly OHLCV bars (not resampled daily data).
    limit: 0 = no cap (return all available history)
    """
    sym = symbol.upper()

    if interval == "weekly":
        prices, indicators = _yfinance_fetch(sym, limit, yf_interval="1wk")
    elif interval == "monthly":
        prices, indicators = _yfinance_fetch(sym, limit, yf_interval="1mo")
    else:
        prices, indicators = _fetch_any(sym, limit)

    if prices.empty:
        raise HTTPException(404, f"No data for {symbol}")

    return {
        "prices":     df_to_records(prices,     limit),
        "indicators": df_to_records(indicators, limit),
    }
