"""
Real-time quotes, intraday candles and company news via Finnhub.
GET /api/quote/{symbol}                   → live price snapshot (+ pre/post market)
GET /api/candles/{symbol}?resolution=5    → intraday OHLCV bars
GET /api/news/{symbol}                    → recent company news
"""

import os
from datetime import date, datetime, timedelta, timezone
from fastapi import APIRouter

router = APIRouter()
_client = None


def _finnhub():
    global _client
    if _client:
        return _client
    key = os.getenv("FINNHUB_API_KEY", "")
    if not key:
        return None
    try:
        import finnhub
        _client = finnhub.Client(api_key=key)
    except Exception:
        pass
    return _client


@router.get("/quote/{symbol}")
def get_quote(symbol: str):
    """Live price snapshot: current, open, high, low, prev_close, change_pct.
    Primary: Finnhub.  Fallback: FMP stable API.
    Extended hours (pre_market / post_market) enriched via yfinance fast_info.
    Works for ALL publicly-traded symbols.
    """
    sym = symbol.upper()
    result = None

    # ── Primary: Finnhub ──────────────────────────────────────────────────────
    client = _finnhub()
    if client:
        try:
            q = client.quote(sym)
            if q.get("c"):   # valid non-zero price
                result = {
                    "symbol":     sym,
                    "price":      q.get("c"),
                    "open":       q.get("o"),
                    "high":       q.get("h"),
                    "low":        q.get("l"),
                    "prev_close": q.get("pc"),
                    "change":     q.get("d"),
                    "change_pct": q.get("dp"),
                    "timestamp":  q.get("t"),
                }
        except Exception:
            pass

    # ── Fallback: FMP ─────────────────────────────────────────────────────────
    if result is None:
        try:
            from api.routers.fmp import get_quote as fmp_quote
            fq = fmp_quote(sym)
            if fq and fq.get("price"):
                prev = fq.get("previousClose") or 0
                price = fq.get("price") or 0
                change = fq.get("change") or (price - prev if prev else None)
                change_pct = fq.get("changePercentage") or (change / prev * 100 if prev else None)
                result = {
                    "symbol":     sym,
                    "price":      price,
                    "open":       fq.get("open"),
                    "high":       fq.get("dayHigh"),
                    "low":        fq.get("dayLow"),
                    "prev_close": fq.get("previousClose"),
                    "change":     round(change, 4) if change is not None else None,
                    "change_pct": round(change_pct, 4) if change_pct is not None else None,
                    "timestamp":  fq.get("timestamp"),
                    "source":     "fmp",
                }
        except Exception:
            pass

    if result is None:
        return {"error": "No quote data available — configure FINNHUB_API_KEY for best results"}

    # ── Extended hours: pre-market + after-hours via yfinance ─────────────────
    # Works for all symbols — yfinance fast_info is a lightweight, cached call.
    try:
        import yfinance as yf
        fi  = yf.Ticker(sym).fast_info
        pc  = result.get("prev_close") or result.get("price") or 1

        pm = getattr(fi, "pre_market_price",  None)
        ah = getattr(fi, "post_market_price", None)

        if pm and float(pm) > 0:
            pm = round(float(pm), 4)
            result["pre_market_price"]      = pm
            result["pre_market_change_pct"] = round((pm - pc) / pc * 100, 3) if pc else None

        if ah and float(ah) > 0:
            ah = round(float(ah), 4)
            result["post_market_price"]      = ah
            result["post_market_change_pct"] = round((ah - pc) / pc * 100, 3) if pc else None
    except Exception:
        pass

    return result


@router.get("/candles/{symbol}")
def get_candles(symbol: str, resolution: int = 5, days_back: int = 5):
    """Intraday OHLCV candles via yfinance.

    resolution in minutes: 1, 5, 15, 30, 60, 120 (2h), 240 (4h).
    120 and 240 are synthesised by fetching 1h bars and resampling.
    """
    import logging
    log = logging.getLogger(__name__)
    try:
        import yfinance as yf
        import pandas as pd

        # 2h (120) and 4h (240) are not native yfinance intervals — fetch 1h and resample
        resample_factor = None
        if resolution in (120, 240):
            resample_factor = resolution // 60   # 2 or 4
            base_resolution = 60
        else:
            base_resolution = resolution

        interval_map = {1: "1m", 5: "5m", 15: "15m", 30: "30m", 60: "1h"}
        interval = interval_map.get(base_resolution, "5m")

        # Use valid yfinance period strings — "Nd" is unreliable for large N.
        # yfinance caps: 1m=7d, 2-30m=60d, 60m=730d
        if base_resolution <= 1:
            period = "7d"
        elif base_resolution <= 30:
            period = "60d"
        else:
            # 1h/2h/4h: request up to 2 years of 1h bars (yfinance hard cap: 730d)
            capped = min(days_back, 730)
            import datetime
            start = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=capped)
            end   = datetime.datetime.now(datetime.timezone.utc)
            df = yf.Ticker(symbol.upper()).history(start=start, end=end, interval=interval)
            period = None   # signal we've already fetched

        if period is not None:
            df = yf.Ticker(symbol.upper()).history(period=period, interval=interval)

        if df is None or df.empty:
            return []

        # Flatten MultiIndex columns (newer yfinance versions return (column, ticker) tuples)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Strip timezone so resample + timestamp() work consistently across platforms
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Resample 1h → 2h / 4h
        if resample_factor and resample_factor > 1:
            freq = f"{resolution}min"
            df = (
                df.resample(freq)
                .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                .dropna(subset=["Open", "Close"])
            )

        result = []
        for ts, row in df.iterrows():
            try:
                o = float(row["Open"])
                h = float(row["High"])
                lo = float(row["Low"])
                c = float(row["Close"])
                v = float(row.get("Volume", 0) or 0)
                if any(pd.isna(x) for x in (o, h, lo, c)):
                    continue
                t = int(pd.Timestamp(ts).timestamp())
                result.append({"time": t, "open": o, "high": h, "low": lo, "close": c, "volume": v})
            except Exception:
                continue
        return result
    except Exception as exc:
        log.error("get_candles %s res=%s: %s", symbol, resolution, exc)
        return []


@router.get("/news/{symbol}")
def get_news(symbol: str, days: int = 7, limit: int = 20):
    """Recent company news headlines from Finnhub."""
    client = _finnhub()
    if not client:
        return []
    try:
        to_date   = date.today().isoformat()
        from_date = (date.today() - timedelta(days=days)).isoformat()
        articles  = client.company_news(symbol.upper(), _from=from_date, to=to_date)
        return [
            {
                "headline": a.get("headline"),
                "source":   a.get("source"),
                "url":      a.get("url"),
                "datetime": a.get("datetime"),
                "summary":  a.get("summary", "")[:200],
            }
            for a in (articles or [])[:limit]
        ]
    except Exception as e:
        return []


# ── Company name lookup ────────────────────────────────────────────────────────
import time as _time

_company_cache: dict[str, dict] = {}
_COMPANY_TTL = 3600  # 1 hour


@router.get("/company/{symbol}")
def get_company(symbol: str):
    """Return company name, sector and exchange for a symbol.
    Tries Finnhub profile2 first, falls back to yfinance fast_info.
    Cached 1 hour — called from the frontend search dropdown.
    """
    sym = symbol.upper()
    cached = _company_cache.get(sym)
    if cached and _time.time() - cached["_ts"] < _COMPANY_TTL:
        return {k: v for k, v in cached.items() if k != "_ts"}

    result = {"symbol": sym, "name": None, "sector": None, "exchange": None}

    # ── Primary: Finnhub profile2 ─────────────────────────────────────────────
    client = _finnhub()
    if client:
        try:
            p = client.company_profile2(symbol=sym)
            if p and p.get("name"):
                result = {
                    "symbol":   sym,
                    "name":     p.get("name"),
                    "sector":   p.get("finnhubIndustry"),
                    "exchange": p.get("exchange"),
                }
                _company_cache[sym] = {**result, "_ts": _time.time()}
                return result
        except Exception:
            pass

    # ── Fallback: yfinance ────────────────────────────────────────────────────
    try:
        import yfinance as yf
        info = yf.Ticker(sym).fast_info
        name = getattr(info, "display_name", None) or getattr(info, "long_name", None)
        if name:
            result["name"] = name
    except Exception:
        pass

    _company_cache[sym] = {**result, "_ts": _time.time()}
    return result


# ── Macro market ticker ────────────────────────────────────────────────────────
# Global markets — indices, commodities, FX, crypto
_MACRO_SYMS = [
    ("^GSPTSE",  "S&P/TSX"),
    ("^DJI",     "Dow Jones"),
    ("^GSPC",    "S&P 500"),
    ("^IXIC",    "Nasdaq"),
    ("CL=F",     "WTI Crude"),
    ("BZ=F",     "Brent"),
    ("GC=F",     "Gold"),
    ("CAD=X",    "USD/CAD"),
    ("EURUSD=X", "EUR/USD"),
    ("BTC-USD",  "Bitcoin"),
]
_macro_cache: dict = {"ts": 0.0, "data": []}
_MACRO_TTL  = 30  # seconds — longer cache to avoid Yahoo rate limits with 10 symbols


@router.get("/macro")
def get_macro():
    """Live macro snapshot — global indices, commodities, FX, crypto. Cached 30s."""
    global _macro_cache
    now = _time.time()
    if now - _macro_cache["ts"] < _MACRO_TTL:
        return {"items": _macro_cache["data"], "cached": True}

    items = []

    try:
        import yfinance as yf
        syms   = [s for s, _ in _MACRO_SYMS]
        names  = {s: n for s, n in _MACRO_SYMS}
        # yf.download is a single HTTP request — much less likely to rate-limit
        df = yf.download(
            syms,
            period="2d",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        for sym in syms:
            try:
                # Multi-ticker download: df[sym]["Close"] is a Series
                if len(syms) > 1:
                    closes = df[sym]["Close"].dropna()
                else:
                    closes = df["Close"].dropna()
                if len(closes) < 1:
                    continue
                price = float(closes.iloc[-1])
                prev  = float(closes.iloc[-2]) if len(closes) >= 2 else price
                chg   = round(price - prev, 6)
                pct   = round((chg / prev * 100) if prev else 0, 2)
                items.append({
                    "symbol": sym,
                    "name":   names[sym],
                    "price":  round(price, 4),
                    "change": round(chg, 4),
                    "pct":    pct,
                })
            except Exception:
                pass
    except Exception:
        pass

    # Only update cache when we got data — stale data is better than empty
    if items:
        _macro_cache = {"ts": now, "data": items}
    elif _macro_cache["data"]:
        # Rate-limited: bump timestamp so we don't hammer again immediately
        _macro_cache["ts"] = now - _MACRO_TTL + 10  # retry in 10 s

    return {"items": _macro_cache["data"], "cached": not bool(items)}
