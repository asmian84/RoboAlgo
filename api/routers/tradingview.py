"""
TradingView MCP — evaluation router.
Wraps tradingview-ta + tradingview-screener for technical analysis,
screening, and backtesting. No API keys required.
"""

import logging
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Single-symbol full analysis ──────────────────────────────────────────────
@router.get("/analysis/{symbol}")
def tv_analysis(symbol: str, exchange: str = "NASDAQ", interval: str = "1d"):
    """Full TradingView technical analysis for a single symbol."""
    from tradingview_ta import TA_Handler, Interval

    INTERVAL_MAP = {
        "1m": Interval.INTERVAL_1_MINUTE,
        "5m": Interval.INTERVAL_5_MINUTES,
        "15m": Interval.INTERVAL_15_MINUTES,
        "30m": Interval.INTERVAL_30_MINUTES,
        "1h": Interval.INTERVAL_1_HOUR,
        "2h": Interval.INTERVAL_2_HOURS,
        "4h": Interval.INTERVAL_4_HOURS,
        "1d": Interval.INTERVAL_1_DAY,
        "1w": Interval.INTERVAL_1_WEEK,
        "1M": Interval.INTERVAL_1_MONTH,
    }
    tv_interval = INTERVAL_MAP.get(interval, Interval.INTERVAL_1_DAY)

    try:
        handler = TA_Handler(
            symbol=symbol.upper(),
            screener="america",
            exchange=exchange.upper(),
            interval=tv_interval,
        )
        analysis = handler.get_analysis()

        # Oscillators
        osc = analysis.oscillators
        # Moving averages
        ma = analysis.moving_averages
        # Summary
        summary = analysis.summary
        # Indicators (raw values)
        indicators = analysis.indicators

        return {
            "symbol": symbol.upper(),
            "exchange": exchange.upper(),
            "interval": interval,
            "summary": {
                "recommendation": summary.get("RECOMMENDATION", "NEUTRAL"),
                "buy": summary.get("BUY", 0),
                "sell": summary.get("SELL", 0),
                "neutral": summary.get("NEUTRAL", 0),
            },
            "oscillators": {
                "recommendation": osc.get("RECOMMENDATION", "NEUTRAL"),
                "buy": osc.get("BUY", 0),
                "sell": osc.get("SELL", 0),
                "neutral": osc.get("NEUTRAL", 0),
                "details": osc.get("COMPUTE", {}),
            },
            "moving_averages": {
                "recommendation": ma.get("RECOMMENDATION", "NEUTRAL"),
                "buy": ma.get("BUY", 0),
                "sell": ma.get("SELL", 0),
                "neutral": ma.get("NEUTRAL", 0),
                "details": ma.get("COMPUTE", {}),
            },
            "indicators": {
                "rsi": indicators.get("RSI"),
                "stoch_k": indicators.get("Stoch.K"),
                "stoch_d": indicators.get("Stoch.D"),
                "cci": indicators.get("CCI20"),
                "adx": indicators.get("ADX"),
                "ao": indicators.get("AO"),
                "mom": indicators.get("Mom"),
                "macd": indicators.get("MACD.macd"),
                "macd_signal": indicators.get("MACD.signal"),
                "bbands_upper": indicators.get("BB.upper"),
                "bbands_lower": indicators.get("BB.lower"),
                "ema20": indicators.get("EMA20"),
                "ema50": indicators.get("EMA50"),
                "ema200": indicators.get("EMA200"),
                "sma20": indicators.get("SMA20"),
                "sma50": indicators.get("SMA50"),
                "sma200": indicators.get("SMA200"),
                "pivot_p": indicators.get("Pivot.M.Classic.Middle"),
                "pivot_r1": indicators.get("Pivot.M.Classic.R1"),
                "pivot_s1": indicators.get("Pivot.M.Classic.S1"),
                "atr": indicators.get("ATR"),
                "volume": indicators.get("volume"),
                "close": indicators.get("close"),
                "open": indicators.get("open"),
                "high": indicators.get("high"),
                "low": indicators.get("low"),
            },
        }
    except Exception as e:
        logger.error("TradingView analysis failed for %s: %s", symbol, e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Multi-timeframe analysis ─────────────────────────────────────────────────
@router.get("/mtf/{symbol}")
def tv_mtf_analysis(symbol: str, exchange: str = "NASDAQ"):
    """Multi-timeframe TradingView analysis (15m → Weekly)."""
    from tradingview_ta import TA_Handler, Interval

    timeframes = [
        ("15m", Interval.INTERVAL_15_MINUTES),
        ("1h",  Interval.INTERVAL_1_HOUR),
        ("4h",  Interval.INTERVAL_4_HOURS),
        ("1d",  Interval.INTERVAL_1_DAY),
        ("1w",  Interval.INTERVAL_1_WEEK),
    ]

    results = {}
    for label, tv_int in timeframes:
        try:
            handler = TA_Handler(
                symbol=symbol.upper(),
                screener="america",
                exchange=exchange.upper(),
                interval=tv_int,
            )
            a = handler.get_analysis()
            s = a.summary
            results[label] = {
                "recommendation": s.get("RECOMMENDATION", "NEUTRAL"),
                "buy": s.get("BUY", 0),
                "sell": s.get("SELL", 0),
                "neutral": s.get("NEUTRAL", 0),
            }
        except Exception:
            results[label] = {"recommendation": "ERROR", "buy": 0, "sell": 0, "neutral": 0}

    # Alignment score: how many timeframes agree on BUY or SELL
    recs = [v["recommendation"] for v in results.values()]
    buy_count = sum(1 for r in recs if "BUY" in r)
    sell_count = sum(1 for r in recs if "SELL" in r)
    alignment = "BULLISH" if buy_count >= 4 else "BEARISH" if sell_count >= 4 else "MIXED"

    return {
        "symbol": symbol.upper(),
        "timeframes": results,
        "alignment": alignment,
        "buy_count": buy_count,
        "sell_count": sell_count,
    }


# ── Top gainers / losers (Bollinger-based) ───────────────────────────────────
@router.get("/screener")
def tv_screener(
    market: str = "america",
    limit: int = Query(default=20, le=50),
):
    """Top movers from TradingView screener."""
    try:
        from tradingview_screener import Scanner
        # Get top gainers by change%
        result = (
            Scanner()
            .select("name", "close", "change", "volume", "Recommend.All", "RSI", "ADX")
            .where(Scanner.filters.market == market)
            .order_by("change", ascending=False)
            .limit(limit)
            .get()
        )
        rows = []
        for _, row in result.iterrows():
            rows.append({
                "symbol": row.get("name", ""),
                "close": row.get("close"),
                "change_pct": round(row.get("change", 0), 2),
                "volume": row.get("volume"),
                "recommendation": row.get("Recommend.All"),
                "rsi": row.get("RSI"),
                "adx": row.get("ADX"),
            })
        return {"market": market, "count": len(rows), "movers": rows}
    except Exception as e:
        logger.error("TradingView screener failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Bollinger squeeze scan ───────────────────────────────────────────────────
@router.get("/squeeze")
def tv_bollinger_squeeze(exchange: str = "NASDAQ", limit: int = Query(default=20, le=50)):
    """Find stocks with tight Bollinger Bands (squeeze candidates)."""
    try:
        from tradingview_screener import Scanner
        result = (
            Scanner()
            .select("name", "close", "BB.upper", "BB.lower", "volume", "change", "RSI")
            .where(Scanner.filters.exchange == exchange)
            .order_by("BB.lower", ascending=False)  # proxy for tight bands
            .limit(limit * 2)
            .get()
        )
        rows = []
        for _, row in result.iterrows():
            bb_upper = row.get("BB.upper")
            bb_lower = row.get("BB.lower")
            close = row.get("close")
            if bb_upper and bb_lower and close and close > 0:
                bb_width = (bb_upper - bb_lower) / close * 100
                rows.append({
                    "symbol": row.get("name", ""),
                    "close": close,
                    "bb_width_pct": round(bb_width, 2),
                    "change_pct": round(row.get("change", 0), 2),
                    "volume": row.get("volume"),
                    "rsi": row.get("RSI"),
                })
        rows.sort(key=lambda x: x["bb_width_pct"])
        return {"exchange": exchange, "count": len(rows[:limit]), "squeeze": rows[:limit]}
    except Exception as e:
        logger.error("TradingView squeeze scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Financial news (RSS) ─────────────────────────────────────────────────────
@router.get("/news")
def tv_news(limit: int = Query(default=15, le=50)):
    """Aggregate financial news from RSS feeds."""
    import feedparser
    from datetime import datetime

    feeds = [
        ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews"),
        ("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
        ("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories"),
    ]

    articles = []
    for source_name, url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:limit]:
                published = ""
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    try:
                        published = datetime(*entry.published_parsed[:6]).isoformat()
                    except Exception:
                        published = getattr(entry, "published", "")
                articles.append({
                    "source": source_name,
                    "title": getattr(entry, "title", ""),
                    "link": getattr(entry, "link", ""),
                    "published": published,
                    "summary": getattr(entry, "summary", "")[:200],
                })
        except Exception:
            continue

    # Sort by published date descending
    articles.sort(key=lambda x: x.get("published", ""), reverse=True)
    return {"count": len(articles[:limit]), "articles": articles[:limit]}
