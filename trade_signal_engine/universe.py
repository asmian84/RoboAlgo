# trade_signal_engine/universe.py
"""
Universe Provider — scans ALL ~6,400+ US-listed stocks plus OTC.

Three-tier funnel:
  Tier 1: Full market snapshot (all tickers, batch download, ~10s)
  Tier 2: Pre-filter to top 300 candidates by unusual activity
  Tier 3: Deep 18-confluence analysis (handled by Scanner)

Data sources:
  - Finnhub API: complete exchange symbol list (NYSE + NASDAQ + AMEX)
  - yfinance bulk download: price/volume/change for thousands of tickers in one call
  - DB: our tracked 233+ instruments (always included)

The goal: NEVER miss a stock going from $0.10 to $10.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Cache (refreshed every 15 min) ────────────────────────────────────────────

_TICKER_CACHE: dict[str, tuple[float, list[str]]] = {}
_TICKER_CACHE_TTL = 900  # 15 min

# ── Known liquid options tickers (always deep-scanned with options chain) ──────

OPTIONS_FOCUS = [
    "SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA", "AMD", "AMZN", "META",
    "GOOGL", "MSFT", "NFLX", "COIN", "MARA", "SOFI", "PLTR", "ARM",
    "BAC", "JPM", "XOM", "BABA", "NIO", "RIVN", "LCID", "F", "GM",
]


@dataclass
class SymbolCandidate:
    symbol: str
    source: str              # "exchange" | "tracked" | "curated"
    price: float = 0.0
    change_pct: float = 0.0
    volume: float = 0.0
    avg_volume: float = 0.0
    volume_ratio: float = 1.0
    float_shares: float = 0.0       # in millions
    short_pct: float = 0.0          # short % of float
    market_cap: float = 0.0         # in millions
    flags: list[str] = field(default_factory=list)


# ── Tier 1: Get ALL listed tickers ────────────────────────────────────────────

def _get_all_exchange_tickers() -> list[str]:
    """
    Fetch complete list of US-listed stock symbols from Finnhub.
    Returns ~6,400 tickers across NYSE, NASDAQ, AMEX.
    Falls back to a broad yfinance approach if Finnhub unavailable.
    """
    cache_key = "exchange_tickers"
    now = time.time()
    if cache_key in _TICKER_CACHE:
        ts, tickers = _TICKER_CACHE[cache_key]
        if now - ts < _TICKER_CACHE_TTL:
            return tickers

    tickers: list[str] = []

    # Method 1: Finnhub (preferred — returns ALL listed symbols)
    api_key = os.environ.get("FINNHUB_API_KEY")
    if api_key:
        try:
            import requests
            resp = requests.get(
                "https://finnhub.io/api/v1/stock/symbol",
                params={"exchange": "US", "token": api_key},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                tickers = [
                    item["symbol"] for item in data
                    if item.get("type") in ("Common Stock", "ETP", "ADR")
                    and "." not in item.get("symbol", ".")
                    and len(item.get("symbol", "")) <= 5
                ]
                logger.info("Finnhub returned %d US tickers", len(tickers))
        except Exception as e:
            logger.warning("Finnhub ticker fetch failed: %s", e)

    # Method 2: Fallback — use DB + static broad universe
    if not tickers:
        try:
            from data_engine.loader import DataLoader
            loader = DataLoader()
            tickers = loader.get_all_symbols()
            logger.info("Loaded %d tickers from DB", len(tickers))
        except Exception:
            pass

        tickers = list(set(tickers + _BROAD_UNIVERSE))

    _TICKER_CACHE[cache_key] = (now, tickers)
    return tickers


# ── Tier 2: Batch snapshot + pre-filter ───────────────────────────────────────

def get_full_universe(
    min_volume_ratio: float = 2.0,
    min_change_pct: float = 3.0,
    max_candidates: int = 300,
    include_all_tracked: bool = True,
) -> list[SymbolCandidate]:
    """
    Scan ALL US stocks, batch-filter by unusual activity, return top candidates.

    Uses yfinance bulk download to get 1-day snapshot of ALL tickers at once.
    This is how we catch penny stocks before they 100x.
    """
    all_tickers = _get_all_exchange_tickers()
    logger.info("Scanning %d tickers for unusual activity...", len(all_tickers))

    candidates: dict[str, SymbolCandidate] = {}

    batch_size = 500
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i + batch_size]
        try:
            df = yf.download(
                " ".join(batch), period="5d", interval="1d",
                progress=False, timeout=30, group_by="ticker",
                threads=True,
            )
            if df is None or df.empty:
                continue

            for sym in batch:
                try:
                    if len(batch) == 1:
                        sym_df = df
                    else:
                        if sym not in df.columns.get_level_values(0):
                            continue
                        sym_df = df[sym].dropna()

                    if sym_df.empty or len(sym_df) < 2:
                        continue

                    if isinstance(sym_df.columns, pd.MultiIndex):
                        sym_df.columns = sym_df.columns.get_level_values(0)
                    sym_df.columns = [c.lower() for c in sym_df.columns]

                    close_today = float(sym_df["close"].iloc[-1])
                    close_prev = float(sym_df["close"].iloc[-2])
                    vol_today = float(sym_df["volume"].iloc[-1]) if "volume" in sym_df else 0
                    avg_vol = float(sym_df["volume"].iloc[:-1].mean()) if "volume" in sym_df and len(sym_df) > 2 else 1

                    if close_prev <= 0 or close_today <= 0:
                        continue

                    change_pct = (close_today - close_prev) / close_prev * 100
                    vol_ratio = vol_today / avg_vol if avg_vol > 0 else 1.0

                    flags = []
                    if close_today < 5:
                        flags.append("penny")
                    if close_today < 1:
                        flags.append("sub_dollar")
                    if vol_ratio >= min_volume_ratio:
                        flags.append("volume_spike")
                    if abs(change_pct) >= min_change_pct:
                        flags.append("big_mover")
                    if vol_ratio >= 5:
                        flags.append("extreme_volume")
                    if abs(change_pct) >= 15:
                        flags.append("explosive_move")

                    if flags or include_all_tracked:
                        candidates[sym] = SymbolCandidate(
                            symbol=sym, source="exchange",
                            price=close_today, change_pct=round(change_pct, 2),
                            volume=vol_today, avg_volume=avg_vol,
                            volume_ratio=round(vol_ratio, 2),
                            flags=flags,
                        )
                except Exception:
                    continue

        except Exception as e:
            logger.warning("Batch download failed for chunk %d: %s", i, e)
            continue

    # Always include tracked DB symbols
    try:
        from data_engine.loader import DataLoader
        for sym in DataLoader().get_all_symbols():
            if sym not in candidates:
                candidates[sym] = SymbolCandidate(symbol=sym, source="tracked")
            else:
                candidates[sym].source = "tracked"
    except Exception:
        pass

    # Enrich top candidates with float + short interest
    top_flagged = sorted(
        [c for c in candidates.values() if c.flags],
        key=lambda c: (len(c.flags), abs(c.change_pct), c.volume_ratio),
        reverse=True,
    )[:100]

    _enrich_float_and_si(top_flagged)

    result = sorted(
        candidates.values(),
        key=lambda c: (
            len(c.flags),
            1 if "extreme_volume" in c.flags else 0,
            1 if "explosive_move" in c.flags else 0,
            1 if "low_float" in c.flags else 0,
            abs(c.change_pct),
            c.volume_ratio,
        ),
        reverse=True,
    )
    return result[:max_candidates]


def _enrich_float_and_si(candidates: list[SymbolCandidate]) -> None:
    """Fetch float shares and short interest for top candidates."""
    if not candidates:
        return

    symbols = [c.symbol for c in candidates[:50]]
    sym_map = {c.symbol: c for c in candidates}

    def _fetch_info(sym: str):
        try:
            info = yf.Ticker(sym).info
            c = sym_map.get(sym)
            if not c:
                return
            c.float_shares = (info.get("floatShares", 0) or 0) / 1e6
            c.short_pct = (info.get("shortPercentOfFloat", 0) or 0) * 100
            c.market_cap = (info.get("marketCap", 0) or 0) / 1e6

            if c.float_shares > 0 and c.float_shares < 20:
                c.flags.append("low_float")
            if c.short_pct > 15:
                c.flags.append("high_si")
            if c.short_pct > 30:
                c.flags.append("squeeze_candidate")
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_fetch_info, symbols))


def get_options_universe() -> list[str]:
    """Return symbols with liquid options for options-specific scanning."""
    return OPTIONS_FOCUS


# ── Broad static universe (fallback when Finnhub unavailable) ─────────────────

_BROAD_UNIVERSE = [
    # Mega/Large Cap
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "BRK-B", "TSLA",
    "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
    "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO", "ACN", "CRM",
    "AMD", "ABT", "TMO", "DHR", "LIN", "NEE", "PM", "TXN", "ADBE", "ORCL",
    "NFLX", "CMCSA", "NKE", "BMY", "UPS", "RTX", "LOW", "UNP", "INTC", "QCOM",
    "BA", "CAT", "GE", "DE", "SBUX", "PLD", "AMAT", "ISRG", "MDT", "GS",
    "BLK", "GILD", "SYK", "ADP", "MMC", "REGN", "VRTX", "BKNG", "ZTS", "CB",
    "PYPL", "LRCX", "MU", "KLAC", "ASML", "MRVL", "SNPS", "CDNS", "PANW",
    # Mid Cap Tech / Growth
    "ARM", "PLTR", "SOFI", "COIN", "SQ", "SHOP", "UBER", "LYFT", "DASH", "ABNB",
    "DDOG", "SNOW", "NET", "CRWD", "ZS", "OKTA", "MDB", "ROKU", "TTD", "PINS",
    "RBLX", "U", "HOOD", "AFRM", "UPST", "BILL", "HUBS", "TWLO", "DKNG", "PENN",
    # Biotech / Pharma
    "MRNA", "BNTX", "PFE", "BIIB", "AMGN", "REGN", "VRTX", "SGEN", "ALNY",
    # Energy / Commodities
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX", "HAL", "DVN",
    "GLD", "SLV", "USO", "UNG", "BOIL",
    # Financials
    "BAC", "JPM", "GS", "MS", "WFC", "C", "SCHW", "BX", "KKR", "APO",
    # ETFs
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "XLP",
    "TLT", "HYG", "GDX", "ARKK", "ARKG", "XBI", "SMH", "SOXX", "KWEB",
    "TQQQ", "SQQQ", "SPXS", "SPXL", "UVXY", "VXX", "VIXY",
    # Leveraged / Inverse
    "SOXL", "SOXS", "LABU", "LABD", "TECS", "TECL", "FAS", "FAZ",
    "SDOW", "UDOW", "TNA", "TZA", "NUGT", "DUST",
    # Popular Micro-caps / Penny Stocks / Meme Stocks
    "MULN", "FFIE", "SNDL", "CLOV", "WISH", "GOEV", "NKLA", "RIDE", "QS",
    "HYMC", "GSAT", "TELL", "WKHS", "OPEN", "SKLZ", "BB", "NOK", "AMC", "GME",
    "BBBY", "IONQ", "RKLB", "LUNR", "ASTS", "JOBY", "EVTL", "LILM", "ACHR",
    "NIO", "XPEV", "LI", "RIVN", "LCID", "FSR", "PSNY", "VFS", "GOEV",
    "MARA", "RIOT", "HUT", "BITF", "CLSK", "CIFR", "HIVE",
    # Cannabis
    "TLRY", "CGC", "ACB", "SNDL", "OGI", "HEXO",
    # Space / Defense
    "LMT", "RTX", "NOC", "GD", "BA", "RKLB", "ASTR", "SPCE", "BKSY",
]
