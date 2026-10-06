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
    Fetch complete list of US-listed stock symbols.
    Priority: Finnhub (fast, free, 6000+) → AlphaVantage LISTING → DB → static list.
    Returns ~6,400 tickers across NYSE, NASDAQ, AMEX.
    """
    cache_key = "exchange_tickers"
    now = time.time()
    if cache_key in _TICKER_CACHE:
        ts, tickers = _TICKER_CACHE[cache_key]
        if now - ts < _TICKER_CACHE_TTL:
            return tickers

    # Ensure .env is loaded for API keys
    try:
        import dotenv
        dotenv.load_dotenv()
    except ImportError:
        pass

    tickers: list[str] = []

    # Method 1: Finnhub (fast, free tier gives full US listing — ~7000+ tickers)
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

    # Method 2: AlphaVantage LISTING_STATUS (fallback — complete US listing)
    if len(tickers) < 1000:
        try:
            from data_engine.alphavantage_fetcher import fetch_active_listing
            listing = fetch_active_listing()
            if listing and isinstance(listing, list):
                av_tickers = [
                    item.get("symbol", "") for item in listing
                    if isinstance(item, dict)
                    and item.get("assetType") in ("Stock", "ETF")
                    and "." not in item.get("symbol", ".")
                    and len(item.get("symbol", "")) <= 5
                    and item.get("symbol", "")
                ]
                if len(av_tickers) > len(tickers):
                    tickers = av_tickers
                    logger.info("AlphaVantage listing returned %d US tickers", len(tickers))
        except Exception as e:
            logger.warning("AV listing fetch failed: %s", e)

    # Method 3: DB symbols (smallest set — only tracked symbols)
    if len(tickers) < 1000:
        try:
            from data_engine.loader import DataLoader
            loader = DataLoader()
            db_syms = loader.get_all_symbols()
            # Merge DB symbols into existing list
            existing = set(tickers)
            for s in db_syms:
                if s not in existing:
                    tickers.append(s)
                    existing.add(s)
            logger.info("Merged DB symbols: now %d tickers total", len(tickers))
        except Exception:
            pass

    # Always merge with broad static universe as safety net
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
    # Filter out warrants, units, rights, and other non-tradeable symbols
    # Prefer shorter tickers (more liquid, more well-known) — cap at 8000
    all_tickers = [
        t for t in all_tickers
        if len(t) <= 5
        and not t.endswith("W") and not t.endswith("U") and not t.endswith("R")
        and "-" not in t
    ]
    # Prioritize: short tickers first (1-4 chars = major stocks), then 5-char
    all_tickers.sort(key=lambda t: (len(t), t))
    _MAX_PRESCREEN_TICKERS = 8000
    if len(all_tickers) > _MAX_PRESCREEN_TICKERS:
        all_tickers = all_tickers[:_MAX_PRESCREEN_TICKERS]
    logger.info("Scanning %d tickers for unusual activity...", len(all_tickers))

    candidates: dict[str, SymbolCandidate] = {}

    # ── yfinance bulk download — fast batch screening ────────────────────
    # Skip AlphaVantage batch quotes (too slow with rate limiting, 20+ API calls).
    # yfinance bulk download fetches thousands of tickers in a few large batches.
    import time as _time
    _scan_start = _time.time()
    _MAX_SCAN_SECS = 90  # hard timeout for pre-screen

    batch_size = 800  # yfinance handles large batches well with threads=True
    for i in range(0, len(all_tickers), batch_size):
        # Hard timeout — don't let pre-screen run forever
        if _time.time() - _scan_start > _MAX_SCAN_SECS:
            logger.warning("Pre-screen timeout after %.0fs — scanned %d/%d tickers, %d candidates",
                           _time.time() - _scan_start, i, len(all_tickers), len(candidates))
            break

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

    logger.info("Pre-screen complete: %d candidates from %d tickers in %.0fs",
                len(candidates), len(all_tickers), _time.time() - _scan_start)

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


# ── Categorized universe ────────────────────────────────────────────────────

# Price range categories — applied as filters during scan, not static lists
PRICE_RANGE_CATEGORIES = {
    "under5":  (0.01, 5.0),
    "under10": (5.0, 10.0),
    "under20": (10.0, 20.0),
    "under50": (20.0, 50.0),
    "over50":  (50.0, 999999.0),
}

UNIVERSE_CATEGORIES: dict[str, list[str]] = {
    "bluechip": [
        # Mega/large-cap — tech
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO",
        "ADBE", "CRM", "ORCL", "AMD", "INTC", "QCOM", "TXN", "MU", "AMAT", "LRCX",
        "KLAC", "MRVL", "SNPS", "CDNS", "PANW", "CSCO", "ACN", "INTU", "NOW", "WDAY",
        "ADSK", "ANSS", "FTNT", "TEAM", "ZM", "VEEV", "DOCU", "SPLK", "CRWD", "DDOG",
        # Healthcare
        "UNH", "JNJ", "LLY", "ABBV", "MRK", "ABT", "TMO", "DHR", "BMY", "GILD",
        "ISRG", "SYK", "MDT", "REGN", "VRTX", "BIIB", "AMGN", "ZTS", "CI", "HCA",
        "EW", "IDXX", "DXCM", "ALGN", "HOLX", "IQV", "MTD", "WST", "RMD",
        # Consumer
        "WMT", "COST", "HD", "MCD", "SBUX", "NKE", "PG", "KO", "PEP", "LOW",
        "TGT", "BKNG", "ABNB", "DIS", "NFLX", "CMCSA", "AMGN", "LULU", "ROST",
        "DLTR", "DG", "YUM", "CMG", "DARDEN", "EL", "CL", "KMB",
        # Industrials
        "BA", "CAT", "GE", "DE", "UPS", "UNP", "RTX", "HON", "MMM", "ETN",
        "EMR", "ROK", "PH", "ITW", "CSX", "NSC", "FDX", "WM", "RSG", "FAST",
        "ODFL", "JBHT", "PCAR", "CTAS", "PAYX",
        # Energy
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX", "HAL", "DVN",
        "PXD", "FANG", "OXY", "HES", "WMB", "KMI", "OKE", "TRGP",
        # Financials
        "JPM", "BAC", "MS", "WFC", "C", "GS", "SCHW", "BX", "KKR", "APO",
        "V", "MA", "PYPL", "ADP", "CB", "BLK", "ICE", "CME", "MCO", "MSCI",
        "SPGI", "TRV", "AIG", "PGR", "ALL", "MET", "PRU", "AFL",
        # Real estate / utilities
        "PLD", "NEE", "LIN", "PM", "AMT", "CCI", "EQIX", "DLR", "PSA", "O",
        # Defense
        "LMT", "NOC", "GD", "HII", "LHX", "TXT", "HWM",
        # Materials
        "FCX", "NEM", "GOLD", "FMC", "APD", "ECL", "SHW", "PPG", "NUE", "STLD",
    ],
    "growth": [
        # High-growth tech
        "ARM", "PLTR", "SOFI", "COIN", "SQ", "SHOP", "UBER", "LYFT", "DASH",
        "SNOW", "NET", "ZS", "OKTA", "MDB", "ROKU", "TTD", "PINS",
        "RBLX", "U", "HOOD", "AFRM", "UPST", "BILL", "HUBS", "TWLO", "DKNG",
        "PENN", "GTLB", "ASAN", "MNDY", "BRZE", "PATH", "AI",
        "SOUN", "OPEN", "OPRA", "IONQ", "QUBT", "RGTI", "ARQQ",
        "CFLT", "ESTC", "DOCN", "SMCI", "PSTG", "CRDO", "APP", "CWAN",
        "GLBE", "TOST", "COUR", "DUOL", "CELH", "ONON", "BIRK",
        # Biotech growth
        "MRNA", "BNTX", "PFE", "ALNY", "NTLA", "BEAM", "EDIT", "CRSP",
        "FATE", "KYMR", "AKBA", "IMVT", "RCUS", "DNLI", "LEGN", "PCVX",
        "KRTX", "HALO", "MGNX", "CYTK", "RPRX", "GERN", "ACLX", "NUVB",
        "TGTX", "ACAD", "EXAS", "TWST", "BNGO", "APLS", "ARVN",
        # Crypto / fintech
        "MARA", "RIOT", "HUT", "BITF", "CLSK", "CIFR", "HIVE", "CORZ", "BTBT",
        "IREN", "WULF", "MSTR",
        # EV / clean energy
        "NIO", "XPEV", "LI", "RIVN", "LCID", "BLNK", "CHPT", "EVGO",
        "PLUG", "FCEL", "BE", "ENPH", "SEDG", "RUN", "NOVA", "SHLS",
        "ARRY", "MAXN", "FLNC", "STEM",
        # Space / defense tech
        "RKLB", "SPCE", "BKSY", "LUNR", "ASTS", "JOBY", "ACHR", "EVTL", "LILM",
        "MNTS", "VORB",
        # Consumer / retail growth
        "ETSY", "W", "CVNA", "CHWY", "RVLV", "POSH", "REAL", "WISH",
        "CARG", "CARS", "VRM",
        # Industrials / misc growth
        "STEM", "ENVX", "QS", "MVST", "SLDP", "AMPX", "DCFC",
        "HYLN", "SHPW", "GEVO", "CLNE",
        # China tech
        "BABA", "JD", "PDD", "BIDU", "NIO", "XPEV", "LI", "BILI", "IQ",
        "TME", "DIDI", "FUTU", "TIGR", "YMM", "ZTO", "MNSO",
    ],
    "etfs": [
        # Broad market
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "RSP", "MDY", "IJR",
        # Sector
        "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLB", "XLU", "XLRE",
        "XLY", "XLC", "XBI", "SMH", "SOXX", "KWEB", "GDX", "GDXJ", "COPX",
        "ARKK", "ARKG", "ARKF", "ARKW", "ARKQ", "IBIT", "BITO", "GBTC",
        "TAN", "ICLN", "QCLN", "PBW", "LIT", "REMX", "URA", "URNM",
        "HACK", "CIBR", "WCLD", "IGV", "SKYY", "CLOU", "AIQ",
        "IBB", "XBI", "LABU", "LABD",
        "ITA", "PPA", "DFEN",
        "KRE", "IAI", "KBE",
        "ITB", "XHB", "NAIL",
        "JETS", "CRUZE",
        # Fixed income / volatility
        "TLT", "IEF", "SHY", "HYG", "LQD", "UVXY", "VXX", "VIXY",
        "BND", "AGG", "GOVT", "TIPS", "VTIP",
        # Commodities
        "GLD", "SLV", "USO", "UNG", "PDBC", "DBA", "WEAT", "CORN", "SOYB",
        "PPLT", "PALL", "CPER",
        # Leveraged bull
        "TQQQ", "SPXL", "SOXL", "LABU", "TECL", "FAS", "UDOW", "TNA",
        "NUGT", "BOIL", "FNGU", "NAIL", "DPST", "CURE", "DUSL",
        "BULZ", "WEBL", "RETL", "HIBL", "WANT",
        # Leveraged bear (inverse — for hedging in bear markets)
        "SQQQ", "SPXS", "SOXS", "LABD", "TECS", "FAZ", "SDOW", "TZA",
        "DUST", "KOLD", "DRIP", "WEBS", "HIBS",
        # International
        "EEM", "VWO", "EFA", "FXI", "MCHI", "EWZ", "EWJ", "INDA",
    ],
    "penny": [
        # Meme / high-retail-interest
        "AMC", "GME", "NOK", "BB", "SNDL", "TLRY", "CGC",
        "CLOV", "WKHS", "WISH", "SOFI", "PLTR",
        # Speculative EV / mobility
        "MULN", "FFIE", "GOEV", "NKLA", "PSNY", "VFS",
        "SOLO", "AYRO", "RIDE", "WKHS", "HYLN", "ARVL", "PTRA",
        "REE", "ELMS", "SLI", "EVEX", "BRDS",
        # Speculative tech / AI / quantum
        "SKLZ", "GSAT", "TELL", "HYMC",
        "SOUN", "BBAI", "INPX", "XTIA", "BKKT",
        "RGTI", "QUBT", "IONQ", "ARQQ", "QMCO",
        "BTAI", "GFAI", "PRST", "LQDA",
        # Biotech small-cap (high catalyst risk/reward)
        "SAVA", "NNOX", "PRAX", "ANAB", "ARDX", "NKTR", "CLVS",
        "NVAX", "OCGN", "INO", "SRNE", "ATOS", "VTGN", "OBLG",
        "APRE", "CPRX", "CTMX", "STSA", "MDXH",
        "GERN", "NUVB", "TGTX", "IOVA", "PDSB", "BIOR",
        "LIDR", "VNET", "HGEN", "CVM", "BNGO", "FGEN",
        "XERS", "ZYNE", "DARE", "EOLS", "AVDL",
        # Cannabis
        "ACB", "OGI", "HEXO", "TLRY", "SNDL", "CGC",
        "GRWG", "VFF", "CRON", "MAPS", "IIPR",
        # Crypto adjacent / mining
        "BTBT", "WULF", "CORZ", "CIFR", "HIVE", "HUT", "BITF",
        "MARA", "RIOT", "CLSK", "IREN", "GREE", "SOS", "EBON",
        "ANY", "BTCS", "BKKT", "MSTR",
        # Speculative defense / space
        "ASTR", "SPCE", "BKSY", "MNTS", "VORB",
        "LUNR", "RKLB", "ASTS",
        # Chinese ADRs (volatile)
        "JKS", "XNET", "FUTU", "TIGR",
        "YMM", "DIDI", "IQ", "BILI", "TME",
        "TAL", "EDU", "GOTU", "LX", "DAO",
        # Clean energy small-cap
        "PLUG", "FCEL", "BLNK", "CHPT", "EVGO",
        "STEM", "SHLS", "ARRY", "MAXN",
        "NOVA", "FLNC", "GEVO", "CLNE", "AMTX",
        "BE", "BLDP", "DCFC",
        # Volatile retail / fintech
        "UPST", "AFRM", "HOOD", "OPEN", "REAL",
        "PAYO", "PSFE", "OLO", "CWAN", "PAGS",
        "STNE", "LMND", "ROOT", "OUST", "LAZR",
        # 3D printing / robotics
        "DDD", "SSYS", "XONE", "DM", "MKFG",
        "PRNT", "NNDM", "VELO",
        # REITs / misc small cap
        "IIPR", "SAFE", "APLE",
        "IRBT", "SFIX", "SDC", "BARK", "WRBY",
        "BMBL", "MTTR", "VIEW", "WIMI",
        # More volatile small-caps
        "RCAT", "CUEN", "GFAI", "CXAI", "CAVA",
        "PRCH", "ARIS", "SNTI", "SMWB", "TRUP",
        "ACMR", "SATS", "SWI", "ENFN", "BIRD",
    ],
}

# Flat list for backward compat
_BROAD_UNIVERSE = list(dict.fromkeys(
    sym for syms in UNIVERSE_CATEGORIES.values() for sym in syms
))


def get_symbols_for_categories(categories: list[str]) -> list[str]:
    """Return deduplicated symbols for the given category names.

    Handles both asset-type categories (bluechip, growth, etfs, penny)
    and price-range categories (under5, under10, under20, under50, over50).

    Price-range categories return ALL known symbols — the price filter
    is applied during scan when actual prices are available.

    Also merges in all DB-tracked symbols to ensure comprehensive coverage.
    """
    result: list[str] = []
    seen: set[str] = set()

    # Check if any price-range categories are requested
    has_price_filter = any(cat in PRICE_RANGE_CATEGORIES for cat in categories)

    # Asset-type categories: add their static symbol lists
    for cat in categories:
        if cat in PRICE_RANGE_CATEGORIES:
            continue  # handled below
        # Support old name "midcap" mapping to "growth"
        effective_cat = "growth" if cat == "midcap" else cat
        for sym in UNIVERSE_CATEGORIES.get(effective_cat, []):
            if sym not in seen:
                result.append(sym)
                seen.add(sym)

    # If price-range categories are selected, expand the symbol list —
    # but ONLY from asset-type categories that are also selected.
    # If NO asset-type categories are selected, expand from ALL categories.
    if has_price_filter:
        asset_cats = [c for c in categories if c not in PRICE_RANGE_CATEGORIES]
        if asset_cats:
            # Only expand from the selected asset-type categories
            for cat in asset_cats:
                effective_cat = "growth" if cat == "midcap" else cat
                for sym in UNIVERSE_CATEGORIES.get(effective_cat, []):
                    if sym not in seen:
                        result.append(sym)
                        seen.add(sym)
        else:
            # No asset-type filter — expand from all categories
            for cat_syms in UNIVERSE_CATEGORIES.values():
                for sym in cat_syms:
                    if sym not in seen:
                        result.append(sym)
                        seen.add(sym)
        # Also add DB-tracked symbols (these have price data for instant filtering)
        # Exchange tickers are NOT added here — they'd need yfinance downloads
        # just to filter, creating a traffic jam. The curated lists + DB symbols
        # provide sufficient coverage for price-range scans.

    # Merge DB-tracked symbols
    try:
        from data_engine.loader import DataLoader
        for sym in DataLoader().get_all_symbols():
            if sym not in seen and len(sym) <= 5 and "." not in sym:
                result.append(sym)
                seen.add(sym)
    except Exception:
        pass

    return result


def get_price_range_filter(categories: list[str]) -> tuple[float, float] | None:
    """Extract combined price range from category list.

    If multiple price categories are selected (e.g. under5 + under10),
    returns the union range (min of all lows, max of all highs).
    """
    lo_vals, hi_vals = [], []
    for cat in categories:
        if cat in PRICE_RANGE_CATEGORIES:
            lo, hi = PRICE_RANGE_CATEGORIES[cat]
            lo_vals.append(lo)
            hi_vals.append(hi)
    if lo_vals:
        return (min(lo_vals), max(hi_vals))
    return None


def _get_all_db_last_prices() -> dict[str, float]:
    """
    Fast single-query fetch of the last close price for EVERY instrument in the DB.
    Returns {symbol: last_close_price}.
    """
    cache_key = "db_last_prices"
    now = time.time()
    if cache_key in _TICKER_CACHE:
        ts, data = _TICKER_CACHE[cache_key]
        if now - ts < 300:  # cache for 5 min
            return data

    prices: dict[str, float] = {}
    try:
        from database.connection import get_session
        from sqlalchemy import text
        session = get_session()
        try:
            # Single query: get the most recent close for each instrument
            rows = session.execute(text("""
                SELECT i.symbol, p.close
                FROM instruments i
                JOIN price_data p ON p.instrument_id = i.id
                WHERE p.date = (
                    SELECT MAX(p2.date) FROM price_data p2 WHERE p2.instrument_id = i.id
                )
            """)).fetchall()
            for row in rows:
                sym, close = row[0], float(row[1])
                if close > 0:
                    prices[sym] = close
            logger.info("DB last prices: %d instruments", len(prices))
        finally:
            session.close()
    except Exception as e:
        logger.debug("DB last price query failed: %s", e)

    if prices:
        _TICKER_CACHE[cache_key] = (now, prices)
    return prices


def prefilter_by_price(
    symbols: list[str],
    price_range: tuple[float, float],
    max_symbols: int = 8000,
) -> list[str]:
    """
    Pre-filter symbols by last known price BEFORE scanning.

    LEAN approach — zero extra downloads:
      1. DB prices (instant, single SQL query) → filter known symbols
      2. Unknown symbols pass through to the scanner which filters
         during scoring (after its own data download — single download, not double)

    This avoids creating a data traffic jam by not duplicating yfinance calls.
    """
    lo, hi = price_range
    db_matched: list[str] = []
    db_rejected: set[str] = set()

    # ── DB bulk lookup (instant — cached single SQL query) ───────────
    db_prices = _get_all_db_last_prices()
    for sym in symbols:
        p = db_prices.get(sym)
        if p is not None:
            if lo <= p <= hi:
                db_matched.append(sym)
            else:
                db_rejected.add(sym)
        # else: unknown price — pass through to scanner

    # ── Build result: DB matches first, then unknowns ────────────────
    # Unknown symbols go to the scanner which downloads data once and
    # filters by price_range in _score_one() — no double download.
    unknown = [s for s in symbols if s not in db_prices and s not in db_rejected]

    result = db_matched + unknown

    logger.info(
        "Price pre-filter $%.0f-$%.0f: %d DB-matched + %d unknown = %d "
        "(rejected %d DB-known outside range) from %d total",
        lo, hi, len(db_matched), len(unknown), len(result),
        len(db_rejected), len(symbols),
    )
    return result[:max_symbols]
