"""
api/routers/quant_scanner.py
────────────────────────────────────────────────────────────────────────────────
Quant Scanner API — unified endpoint for the full scan → score → allocate pipeline.

Endpoints:
  GET  /quant/scan          Run full pipeline (scan + context + score + allocate)
  GET  /quant/score/{sym}   Score a single symbol
  GET  /quant/gex/{sym}     GEX / DEX / VEX for a symbol
  GET  /quant/explain/{sym} Natural language explanation of current signal
  GET  /quant/tooltips      Return all tooltip definitions
  GET  /quant/tooltips/{key} Return single tooltip
  GET  /quant/tracker/stats Return performance statistics
  POST /quant/tracker/open  Open a trade
  POST /quant/tracker/close Close a trade
"""
from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Lazy-init singletons (avoid import cost at module load) ───────────────────
_scoring_engine  = None
_allocator       = None
_tracker         = None
_catalyst_engine = None
_explainer_beg   = None
_explainer_adv   = None
_level_engine    = None
_backtest_engine = None


def _get_scorer():
    global _scoring_engine
    if _scoring_engine is None:
        from signal_engine.scoring_engine import ScoringEngine
        _scoring_engine = ScoringEngine()
    return _scoring_engine


def _get_allocator(account_size: float = 10_000):
    from trade_engine.allocator import TradeAllocator
    return TradeAllocator(account_size=account_size)


def _get_tracker():
    global _tracker
    if _tracker is None:
        from analytics_engine.trade_tracker import TradeTracker
        _tracker = TradeTracker()
    return _tracker


def _get_catalyst():
    global _catalyst_engine
    if _catalyst_engine is None:
        from context_engine.catalyst_engine import CatalystEngine
        _catalyst_engine = CatalystEngine()
    return _catalyst_engine


def _get_explainer(mode: str = "beginner"):
    from education_engine.signal_explainer import SignalExplainer
    return SignalExplainer(mode=mode)


def _get_level_engine():
    global _level_engine
    if _level_engine is None:
        from signal_engine.level_engine import LevelEngine
        _level_engine = LevelEngine()
    return _level_engine


def _get_backtest_engine():
    global _backtest_engine
    if _backtest_engine is None:
        from signal_engine.backtest_engine import BacktestEngine
        _backtest_engine = BacktestEngine()
    return _backtest_engine


# ── Cache for expensive scan results ─────────────────────────────────────────
_scan_cache: dict[str, tuple[Any, float]] = {}
_SCAN_TTL = 300.0  # 5 minutes

# ── Backtest quality cache — populated by /backtest-universe ──────────────────
# Keyed by symbol → {strategy_key: stats_dict}
# Used by pounce scanner to annotate matched strategies with historical win rate.
_bt_quality_cache: dict[str, dict[str, dict]] = {}


# ── Request/response models ───────────────────────────────────────────────────

class OpenTradeRequest(BaseModel):
    symbol:       str
    signal_type:  str = "long"
    confidence:   str = "medium"
    score:        float = 0.5
    entry_price:  float
    shares:       int
    stop_price:   float
    target_price: float
    dollar_risk:  float = 0.0
    gex_regime:   str = "NEUTRAL"
    catalyst_type: str = "none"


class CloseTradeRequest(BaseModel):
    trade_id:    str
    exit_price:  float
    exit_reason: str = "manual"


# ── Full Pipeline Endpoint ────────────────────────────────────────────────────

@router.get("/scan")
def run_full_scan(
    symbols:      str   = Query(default="AAPL,NVDA,TSLA,SPY,QQQ,AMD,META,MSFT,AMZN,GOOGL"),
    account_size: float = Query(default=10_000.0),
    min_score:    float = Query(default=0.40),   # lowered: GEX/VIX unavailable in batch
    max_trades:   int   = Query(default=3),
    mode:         str   = Query(default="beginner"),
    force:        bool  = Query(default=False),
):
    """
    Full pipeline: scan → context → score → allocate → explain.

    Returns:
      candidates:   All scored candidates above min_score
      allocations:  Top N sized trade plans
      explanations: Natural language explanations per allocation
      summary:      Portfolio-level stats
    """
    cache_key = f"scan_{symbols}_{min_score}"
    now = time.time()
    if not force and cache_key in _scan_cache:
        result, cached_at = _scan_cache[cache_key]
        if now - cached_at < _SCAN_TTL:
            result["cached"] = True
            return result

    ticker_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    scorer      = _get_scorer()
    allocator   = _get_allocator(account_size)
    allocator.max_trades = max_trades
    allocator.min_score  = min_score

    # ── Step 1: Fetch market data + VIX ──────────────────────────────────
    from data_engine.market_data_service import get_quotes_batch, get_quote
    quotes = get_quotes_batch(ticker_list)

    # Fetch VIX once — used by all symbol scores for volatility_state factor
    vix_quote = get_quote("^VIX")
    vix_value = vix_quote.get("price") if vix_quote else None

    # ── Step 2: Scan for movers ───────────────────────────────────────────
    from scanner_engine.top_percent_gainers import TopPercentGainerScanner
    from scanner_engine.high_volume_movers import HighVolumeMoverScanner
    gainers_scanner = TopPercentGainerScanner(top_n=len(ticker_list), min_change=0.5)
    volume_scanner  = HighVolumeMoverScanner(min_vol_ratio=1.5, min_move_pct=0.5)

    gainers = {c["symbol"]: c for c in gainers_scanner.run(ticker_list)}
    movers  = {c["symbol"]: c for c in volume_scanner.run(ticker_list)}

    # ── Step 3: Score each candidate ─────────────────────────────────────
    scored_signals = []
    for sym in ticker_list:
        quote   = quotes.get(sym, {})
        gainer  = gainers.get(sym, {})
        mover   = movers.get(sym, {})
        price   = quote.get("price", 0)
        chg_pct = quote.get("change_percent", gainer.get("change_percent", 0))

        # Volume ratio: prefer scanner result (more precise), fall back to
        # quote volume / avg_volume so all symbols get a real reading
        vol_from_quote = (
            quote.get("volume", 1) / max(quote.get("avg_volume", 1), 1)
            if quote.get("avg_volume", 0) > 0 else 1.0
        )
        vol_r = max(
            gainer.get("volume_ratio", vol_from_quote),
            mover.get("volume_ratio",  vol_from_quote),
        )

        sig = scorer.score(
            symbol      = sym,
            price       = price,
            change_pct  = chg_pct,
            vol_ratio   = vol_r,
            direction   = "long" if chg_pct >= 0 else "short",
            vix         = vix_value,   # ← live VIX for volatility_state factor
        )
        sig_dict = sig.to_dict()
        sig_dict["price"]          = price
        sig_dict["change_percent"] = chg_pct
        sig_dict["volume_ratio"]   = round(vol_r, 2)
        scored_signals.append(sig_dict)

    # Sort by score
    scored_signals.sort(key=lambda x: x["score"], reverse=True)
    candidates = [s for s in scored_signals if s["score"] >= min_score]

    # ── Step 4: Allocate trades ───────────────────────────────────────────
    # ATR proxy = (high - low) from today's daily bar; allocator uses it for
    # stop placement. If high/low unavailable, omit so allocator falls back to
    # price × 2% default.
    def _atr_proxy(q: dict) -> float:
        h = q.get("high", 0)
        lo = q.get("low", 0)
        if h > 0 and lo > 0 and h > lo:
            return h - lo
        return 0.0   # allocator will use price × 2% fallback

    market_data = {
        sym: {
            "price": quotes.get(sym, {}).get("price", 0),
            **({"atr": _atr_proxy(quotes[sym])} if sym in quotes and _atr_proxy(quotes[sym]) > 0 else {}),
        }
        for sym in ticker_list
    }
    allocations  = allocator.allocate(candidates, market_data)
    alloc_dicts  = [a.to_dict() for a in allocations]
    port_summary = allocator.summary(allocations)

    # ── Step 5: Explain allocations ───────────────────────────────────────
    explainer    = _get_explainer(mode)
    explanations = [explainer.explain(a) for a in alloc_dicts]

    result = {
        "cached":       False,
        "timestamp":    now,
        "universe":     ticker_list,
        "candidates":   candidates,
        "allocations":  alloc_dicts,
        "explanations": explanations,
        "summary":      port_summary,
    }
    _scan_cache[cache_key] = (result, now)
    return result


# ── Single Symbol Score ───────────────────────────────────────────────────────

@router.get("/score/{symbol}")
def score_symbol(
    symbol: str,
    mode:   str = Query(default="beginner"),
):
    """Score and explain a single symbol."""
    sym   = symbol.upper()
    from data_engine.market_data_service import get_quote
    quote = get_quote(sym) or {}

    scorer    = _get_scorer()
    chg_pct   = quote.get("change_percent", 0)
    avg_vol   = quote.get("avg_volume", 0)
    vol_ratio = (quote.get("volume", 1) / max(avg_vol, 1)) if avg_vol > 0 else 1.0

    from data_engine.market_data_service import get_quote as _gq
    vix_q   = _gq("^VIX")
    vix_val = vix_q.get("price") if vix_q else None

    sig = scorer.score(
        symbol      = sym,
        price       = quote.get("price", 0),
        change_pct  = chg_pct,
        vol_ratio   = vol_ratio,
        direction   = "long" if chg_pct >= 0 else "short",
        vix         = vix_val,
    )
    sig_dict  = sig.to_dict()
    explainer = _get_explainer(mode)
    explanation = explainer.explain(sig_dict)
    return {
        "symbol":      sym,
        "quote":       quote,
        "signal":      sig_dict,
        "explanation": explanation,
    }


# ── GEX / DEX / VEX ──────────────────────────────────────────────────────────

@router.get("/gex/{symbol}")
def get_gex_profile(symbol: str, near_expiry: bool = Query(default=False)):
    """Full dealer positioning: GEX + DEX + VEX combined."""
    sym = symbol.upper()
    try:
        from data_engine.yahoo_options_fetcher import fetch_options_chain
        from options_engine.gamma_exposure import GammaExposureEngine
        from options_engine.dex_engine import DEXEngine
        from options_engine.vanna_exposure import VannaExposureEngine
        from options_engine.dealer_positioning import DealerPositioningEngine

        chain = fetch_options_chain(sym)
        if chain is None:
            raise HTTPException(status_code=404, detail=f"Options data unavailable for {sym}")

        gex_eng = GammaExposureEngine()
        dex_eng = DEXEngine(near_expiry_only=near_expiry)
        vex_eng = VannaExposureEngine()
        dp_eng  = DealerPositioningEngine()

        gex_profile = gex_eng.compute(chain)
        dex_profile = dex_eng.compute(chain)
        vex_profile = vex_eng.compute(chain)

        # Convert profiles to dicts (they have to_dict() method or dataclass fields)
        def _to_dict(obj):
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            if hasattr(obj, "__dict__"):
                return {k: v for k, v in obj.__dict__.items()
                        if not k.startswith("_")}
            return {}

        gex_dict = _to_dict(gex_profile)
        dex_dict = _to_dict(dex_profile)
        vex_dict = _to_dict(vex_profile)

        positioning = dp_eng.compute(gex_profile)

        # Map numeric dealer_bias int to human-readable label
        _bias_int = dex_dict.get("dealer_bias", 0)
        _bias_label = {1: "LONG", -1: "SHORT"}.get(_bias_int, "NEUTRAL")

        return {
            "symbol":           sym,
            "spot_price":       chain.spot,
            "gex":              gex_dict,
            "dex":              dex_dict,
            "vex":              vex_dict,
            "gamma_flip":       gex_dict.get("zero_gamma"),
            "gex_regime":       gex_dict.get("gamma_regime"),       # POSITIVE | NEGATIVE | NEUTRAL
            "positioning_bias": gex_dict.get("positioning_bias"),   # BULLISH | BEARISH | NEUTRAL
            "dex_regime":       dex_dict.get("dex_regime"),         # LONG_DELTA | SHORT_DELTA | NEUTRAL
            "dealer_bias":      _bias_label,                        # LONG | SHORT | NEUTRAL
            "positioning":      _to_dict(positioning),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("GEX compute failed for %s: %s", sym, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Education / Explanation ───────────────────────────────────────────────────

@router.get("/explain/{symbol}")
def explain_signal(
    symbol: str,
    mode:   str = Query(default="beginner"),
):
    """Natural language explanation of the current signal for a symbol."""
    resp = score_symbol(symbol, mode=mode)
    return resp["explanation"]


@router.get("/tooltips")
def get_all_tooltips(category: str = Query(default="")):
    """Return all tooltip definitions, optionally filtered by category."""
    from education_engine.tooltip_system import get_all_tooltips
    return get_all_tooltips(category or None)


@router.get("/tooltips/{key}")
def get_tooltip(key: str):
    """Return a single tooltip by key."""
    from education_engine.tooltip_system import get_tooltip as _get
    tip = _get(key)
    if tip is None:
        raise HTTPException(status_code=404, detail=f"Tooltip '{key}' not found")
    return tip


# ── Trade Tracker ─────────────────────────────────────────────────────────────

@router.get("/tracker/stats")
def get_tracker_stats():
    """Return full performance statistics."""
    tracker = _get_tracker()
    return {
        "overall":       tracker.compute_stats(),
        "by_regime":     tracker.stats_by_regime(),
        "by_confidence": tracker.stats_by_confidence(),
        "by_symbol":     tracker.stats_by_symbol(),
        "open_trades":   tracker.open_trades,
    }


@router.get("/tracker/trades")
def get_all_trades():
    """Return all trades (open + closed)."""
    return _get_tracker().get_all()


@router.post("/tracker/open")
def open_trade(req: OpenTradeRequest):
    """Record a new trade."""
    tracker  = _get_tracker()
    trade_id = tracker.open_trade(
        symbol       = req.symbol,
        signal_type  = req.signal_type,
        confidence   = req.confidence,
        score        = req.score,
        entry_price  = req.entry_price,
        shares       = req.shares,
        stop_price   = req.stop_price,
        target_price = req.target_price,
        dollar_risk  = req.dollar_risk,
        gex_regime   = req.gex_regime,
        catalyst_type = req.catalyst_type,
    )
    return {"trade_id": trade_id, "status": "opened"}


@router.post("/tracker/close")
def close_trade(req: CloseTradeRequest):
    """Close a trade and record P&L."""
    tracker = _get_tracker()
    trade   = tracker.close_trade(req.trade_id, req.exit_price, req.exit_reason)
    if trade is None:
        raise HTTPException(status_code=404, detail=f"Trade {req.trade_id} not found")
    return trade


# ── Pounce Scanner ────────────────────────────────────────────────────────────
# Pre-scans a full universe in parallel. Returns only setups with confluence
# gems within striking distance (support-side only → long trades).

# Inverse ETFs included so the user buys them long for bearish regime plays.
# ── Universe by price band ────────────────────────────────────────────────────
# Sorted roughly low-to-high price. Actual bands are applied live after
# fetching current prices (prices change — we don't hardcode them).
_BULL_UNIVERSE = [
    # ── Mega-cap tech ──────────────────────────────────────────────────────────
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL",
    "AMD", "QCOM", "TXN", "INTC", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "ON",
    "CSCO", "ADBE", "CRM", "NOW", "INTU", "SNPS", "CDNS", "FTNT",
    # ── High-growth / momentum ─────────────────────────────────────────────────
    "NFLX", "PLTR", "COIN", "SMCI", "ARM", "MSTR", "CRWD", "PANW", "UBER",
    "SHOP", "DDOG", "ZS", "NET", "SNOW", "OKTA", "MDB", "GTLB", "HUBS",
    "RBLX", "HOOD", "AFRM", "UPST", "SOFI", "PYPL", "XYZ",
    "PINS", "LYFT", "DASH", "RDDT",
    # ── Semiconductors / hardware ──────────────────────────────────────────────
    "TSM", "ASML", "MCHP", "SWKS", "QRVO", "ENTG", "MPWR", "ONTO", "COHU",
    # ── Finance / banks ────────────────────────────────────────────────────────
    "JPM", "GS", "MS", "BAC", "C", "WFC", "AXP", "V", "MA", "COF",
    "ALLY", "BX", "KKR", "APO", "SCHW", "IBKR",
    # ── Healthcare / biotech ───────────────────────────────────────────────────
    "LLY", "ABBV", "AMGN", "GILD", "REGN", "VRTX", "MRNA", "BIIB",
    "ISRG", "DXCM", "TMO", "ABT", "MDT", "SYK", "BSX", "ZBH",
    "NVAX", "CRSP", "EDIT", "BEAM", "NTLA",
    # ── Consumer / retail ─────────────────────────────────────────────────────
    "COST", "WMT", "TGT", "HD", "LOW", "NKE", "SBUX", "MCD", "CMG",
    "LULU", "ROST", "TJX", "ETSY", "CHWY",
    # ── Energy ────────────────────────────────────────────────────────────────
    "XOM", "CVX", "COP", "SLB", "HAL", "OXY", "MPC", "VLO", "DVN",
    "FANG", "AR", "EQT",
    # ── Industrials / defense ─────────────────────────────────────────────────
    "BA", "CAT", "DE", "HON", "RTX", "LMT", "GD", "NOC",
    "GE", "ETN", "EMR", "PH", "ROK",
    # ── Commodities / materials ───────────────────────────────────────────────
    "FCX", "NEM", "GOLD", "AA", "CLF", "MP", "VALE",
    # ── EV / clean energy ─────────────────────────────────────────────────────
    "NIO", "XPEV", "LI", "RIVN", "LCID", "CHPT", "BLNK", "EVGO",
    "PLUG", "FCEL",
    # ── Mid/small growth ──────────────────────────────────────────────────────
    "BABA", "JD", "PDD", "SE", "MELI", "NU",
    "U", "DKNG", "NCLH", "CCL", "RCL",
    "AAL", "DAL", "UAL",
    "CVS",
    "F", "GM", "STLA", "TM",
    # ── Low-float / micro ─────────────────────────────────────────────────────
    "SIRI", "NOK", "SNAP", "BYND", "SPCE", "NKLA",
    "OPEN", "UWMC", "RKT",
    # ── ETFs (index, sector, vol) ──────────────────────────────────────────────
    "SPY", "QQQ", "IWM", "DIA", "MDY",
    "GLD", "SLV", "USO", "UNG",
    "XLF", "XLE", "XLK", "XBI", "GDX",
    "XLV", "XLI", "XLC", "XLU", "XLRE",
    "ARKK", "ARKF", "ARKG",
    "HYG", "TLT", "EMB",
    "XRT", "IBB", "SOXX",
    "TQQQ", "SOXL",                     # leveraged longs — high momentum
]
_BEAR_UNIVERSE = [
    "SQQQ", "SPXS", "TZA", "SOXS", "UVXY", "SDOW", "ZSL", "SRTY", "VIXY",
]
_INVERSE_SET = set(_BEAR_UNIVERSE)
_OPTIONS_MIN_PRICE = 50.0   # only recommend options for $50+ stocks

# ── Full US market universe (dynamic, fetched from FinnHub + cached 24h) ──────
_full_universe_cache: dict = {"tickers": [], "fetched_at": 0.0}
_FULL_UNIVERSE_TTL = 86_400.0  # 24 hours


def _get_full_us_universe() -> list[str]:
    """
    Fetch all common-stock tickers listed on US exchanges via FinnHub.
    Falls back to the curated _BULL_UNIVERSE if FinnHub is unavailable.
    Result is cached for 24 hours.

    The returned list is used ONLY as a candidate pool for the fast screener —
    the actual deep scan runs on a much smaller filtered subset.
    """
    import time as _t
    import requests as _req

    now = _t.time()
    if _full_universe_cache["tickers"] and now - _full_universe_cache["fetched_at"] < _FULL_UNIVERSE_TTL:
        return _full_universe_cache["tickers"]

    try:
        from api.routers.settings import _load as _load_settings
        key = _load_settings().get("finnhub", {}).get("api_key", "")
        if not key:
            raise ValueError("No FinnHub key")

        r = _req.get(
            "https://finnhub.io/api/v1/stock/symbol",
            params={"exchange": "US", "token": key},
            timeout=15,     # short timeout — fallback is fine
        )
        r.raise_for_status()
        raw = r.json()

        # Keep only common stocks with clean single-segment tickers (≤5 chars)
        tickers = [
            s["symbol"] for s in raw
            if s.get("type") in ("Common Stock", "ETP")
            and "." not in s["symbol"]
            and len(s["symbol"]) <= 5
        ]
        tickers = sorted(set(tickers))
        logger.info("Full US universe loaded: %d tickers from FinnHub", len(tickers))
        _full_universe_cache["tickers"] = tickers
        _full_universe_cache["fetched_at"] = now
        return tickers

    except Exception as exc:
        logger.warning("FinnHub universe fetch failed (%s) — using curated list", exc)
        return list(_BULL_UNIVERSE)


def _finnhub_batch_quote(tickers: list[str], key: str) -> dict[str, dict]:
    """
    Fetch FinnHub batch quotes for up to len(tickers) symbols.
    Returns {symbol: {c, v, dp, ...}} dict, silently skipping failed batches.
    """
    import requests as _req
    result: dict[str, dict] = {}
    for i in range(0, len(tickers), 100):
        batch = tickers[i : i + 100]
        try:
            r = _req.get(
                "https://finnhub.io/api/v1/quote",
                params={"symbol": ",".join(batch), "token": key},
                timeout=5,
            )
            if not r.ok:
                continue
            data = r.json()
            if isinstance(data, dict) and "c" not in data:
                result.update({s: q for s, q in data.items() if isinstance(q, dict)})
        except Exception:
            continue
    return result


def _fast_screen_finnhub(
    extra_tickers: list[str],
    max_extras:    int = 150,
) -> list[str]:
    """
    Stage-1 filter for extra (non-core) tickers from the FinnHub universe.
    Uses FinnHub batch quote endpoint (100 symbols per call) to find tickers
    that are actually moving and liquid today.

    Returns up to max_extras additional symbols to add to the core list.
    This is fast (FinnHub can handle 100 at a time, well within our rate limit).
    """
    try:
        from api.routers.settings import _load as _load_settings
        key = _load_settings().get("finnhub", {}).get("api_key", "")
        if not key:
            return []
    except Exception:
        return []

    passed: list[str] = []
    chunk_size = 100

    # Only screen a reasonable slice — first N candidates alphabetically
    sample = extra_tickers[:3_000]  # screen up to 3000 extra tickers

    quotes = _finnhub_batch_quote(sample, key)
    for sym, q in quotes.items():
        price  = q.get("c", 0) or 0
        vol    = q.get("v", 0) or 0      # current volume
        change = abs(q.get("dp", 0) or 0)  # % change
        if price >= 0.50 and vol >= 500_000 and change >= 0.5:
            passed.append(sym)
            if len(passed) >= max_extras:
                break

    logger.info("FinnHub fast screen: %d extra tickers found (vol≥500k)", len(passed))
    return passed


_pounce_cache: dict[str, tuple[dict, float]] = {}
_POUNCE_TTL = 600.0   # 10 minutes — parallel scan is expensive
_pounce_cache.clear()  # bust cache on server restart so new thresholds apply


def _options_rec(sym: str, spot: float, target_price: float) -> dict | None:
    """
    Find the best call option to buy for a directional long play.

    Criteria:
      • DTE 21–45 (theta-safe, enough time for the move)
      • Delta 0.28–0.55 (directional premium without overpaying)
      • OI > 100 (liquid)
      • IV < 100% (not insanely expensive)
      • Scored by gamma (responsiveness) + OI (liquidity) + delta proximity to 0.40
    """
    import datetime
    try:
        from data_engine.yahoo_options_fetcher import fetch_options_chain
        chain = fetch_options_chain(sym)
        if not chain or not chain.contracts:
            return None

        today = datetime.date.today()
        best       = None
        best_score = -1.0

        for c in chain.contracts:
            if c.option_type != "call":
                continue
            try:
                exp = datetime.date.fromisoformat(c.expiration)
                dte = (exp - today).days
            except Exception:
                continue
            if not (21 <= dte <= 45):
                continue
            if not (0.28 <= c.delta <= 0.55):
                continue
            if c.open_interest < 100 or c.last_price <= 0:
                continue
            if c.implied_vol > 1.0:
                continue
            score = (
                c.gamma * 1000
                + min(c.open_interest / 1_000, 5.0)
                - abs(c.delta - 0.40) * 2.0
            )
            if score > best_score:
                best_score = score
                best = c
                best_dte  = dte

        if best is None:
            return None

        move_needed = round(target_price - spot, 2)
        move_pct    = round((target_price / spot - 1) * 100, 2) if spot > 0 else 0.0

        return {
            "strike":       best.strike,
            "expiration":   best.expiration,
            "dte":          best_dte,
            "delta":        round(best.delta, 2),
            "gamma":        round(best.gamma, 4),
            "iv_pct":       round(best.implied_vol * 100, 1),
            "contract_price": round(best.last_price, 2),
            "cost_per_contract": round(best.last_price * 100, 2),
            "oi":           best.open_interest,
            "target_stock": target_price,
            "move_needed":  move_needed,
            "move_pct":     move_pct,
        }
    except Exception as exc:
        logger.debug("Options rec failed for %s: %s", sym, exc)
        return None


def _mtf_alignment(sym: str) -> dict:
    """
    Multi-timeframe directional alignment — inspired by the Timeframe Clock Pine Script.

    Checks candle direction (open vs close) on 4 timeframes: 15m, 1H, 4H, Daily.
    Returns:
      {
        "15m":   "bull" | "bear" | "doji",
        "1H":    "bull" | "bear" | "doji",
        "4H":    "bull" | "bear" | "doji",
        "1D":    "bull" | "bear" | "doji",
        "aligned_bull": bool,   # all 4 bullish
        "aligned_bear": bool,   # all 4 bearish
        "score": int,           # -4 to +4 (sum of +1 bull / -1 bear / 0 doji per TF)
      }

    Alignment scoring:
      +4 = all timeframes bullish → strongest buy signal, confirm POUNCE
      -4 = all bearish → do NOT pounce on bull setups
       0 = mixed → neutral, skip or watch
    """
    def _candle_dir(o: float, c: float) -> str:
        if c > o * 1.001:
            return "bull"
        if c < o * 0.999:
            return "bear"
        return "doji"

    def _dir_from_df(df: Any, open_col: str = "open", close_col: str = "close") -> str:
        if df is None or (hasattr(df, "empty") and df.empty):
            return "doji"
        try:
            last = df.iloc[-1]
            return _candle_dir(float(last[open_col]), float(last[close_col]))
        except Exception:
            return "doji"

    def _resample_dir(df: Any, rule: str, open_col: str = "open", close_col: str = "close") -> str:
        if df is None or (hasattr(df, "empty") and df.empty) or len(df) < 2:
            return "doji"
        try:
            rs = df.resample(rule).agg({open_col: "first", close_col: "last"}).dropna()
            if rs.empty:
                return "doji"
            return _candle_dir(float(rs.iloc[-1][open_col]), float(rs.iloc[-1][close_col]))
        except Exception:
            return "doji"

    try:
        from data_engine.alphavantage_fetcher import is_premium, fetch_intraday_ohlcv, fetch_daily_ohlcv
        use_av = is_premium()

        if use_av:
            # ── AV Premium path: 2 AV calls → 15min + 60min (resampled to 4H) ──
            intra_15 = fetch_intraday_ohlcv(sym, interval="15min", bars=100)
            daily_av  = fetch_daily_ohlcv(sym, bars=5)

            dir_15m = _dir_from_df(intra_15)
            dir_1h  = _resample_dir(intra_15, "1h")
            dir_4h  = _resample_dir(intra_15, "4h")
            dir_1d  = _dir_from_df(daily_av)
            source  = "av"
        else:
            # ── yfinance fallback: free, no key, 2 calls (15m + 1d) ───────────
            import yfinance as yf
            ticker = yf.Ticker(sym)
            intra  = ticker.history(period="5d", interval="15m")
            daily  = ticker.history(period="3mo", interval="1d")

            dir_15m = _dir_from_df(intra,  open_col="Open", close_col="Close")
            dir_1h  = _resample_dir(intra, "1h", open_col="Open", close_col="Close")
            dir_4h  = _resample_dir(intra, "4h", open_col="Open", close_col="Close")
            dir_1d  = _dir_from_df(daily,  open_col="Open", close_col="Close")
            source  = "yf"

        result: dict[str, Any] = {
            "15m": dir_15m, "1H": dir_1h, "4H": dir_4h, "1D": dir_1d,
        }
        score = sum(1 if v == "bull" else (-1 if v == "bear" else 0) for v in result.values())
        result["aligned_bull"] = score == 4
        result["aligned_bear"] = score == -4
        result["score"]  = score
        result["source"] = source
        return result

    except Exception as exc:
        logger.debug("MTF alignment failed for %s: %s", sym, exc)
        return {"15m": "doji", "1H": "doji", "4H": "doji", "1D": "doji",
                "aligned_bull": False, "aligned_bear": False, "score": 0, "source": "none"}


def _detect_strategy(sym: str) -> list[str]:
    """
    Quick check: which backtested strategies currently trigger on this symbol?
    Returns list of matched strategy names (e.g., ["Pullback to MA", "Squeeze Breakout"]).
    """
    try:
        import yfinance as yf
        import numpy as np
        import pandas as pd
        from signal_engine.backtest_engine import _atr, _pullback_entries, _squeeze_entries, _oversold_entries

        hist = yf.Ticker(sym).history(period="6mo", interval="1d")
        if hist.empty or len(hist) < 60:
            return []
        h = hist["High"].to_numpy()
        l = hist["Low"].to_numpy()
        c = hist["Close"].to_numpy()
        atr_arr = _atr(h, l, c)

        matched = []
        last_idx = len(hist) - 1
        # Check if each strategy triggered in the last 3 bars (fresh signal)
        for name, fn in [
            ("Pullback to MA", _pullback_entries),
            ("Squeeze Breakout", _squeeze_entries),
            ("Oversold Bounce", _oversold_entries),
        ]:
            entries = fn(hist, atr_arr)
            if entries and entries[-1] >= last_idx - 3:
                matched.append(name)
        return matched
    except Exception:
        return []


def _detect_patterns_from_bars(bars: list[dict], atr: float, levels: list[dict]) -> list[dict]:
    """
    Fast, zero-network pattern recognition from the 60-day OHLCV bars already
    in memory.  Returns up to 4 patterns sorted by confidence descending.
    Each pattern: {name, confidence, desc}
    """
    if len(bars) < 20:
        return []

    closes = [b["close"]  for b in bars]
    highs  = [b["high"]   for b in bars]
    lows   = [b["low"]    for b in bars]
    vols   = [b["volume"] for b in bars]

    patterns: list[dict] = []

    # ── 1. Higher Lows — buyers stepping in at higher prices each pullback ──
    swing_lows = [
        lows[i] for i in range(2, len(lows) - 2)
        if lows[i] < lows[i-1] and lows[i] < lows[i-2]
        and lows[i] < lows[i+1] and lows[i] < lows[i+2]
    ]
    if len(swing_lows) >= 3 and swing_lows[-1] > swing_lows[-2] > swing_lows[-3]:
        patterns.append({
            "name": "Higher Lows",
            "confidence": 0.87,
            "desc": "Ascending swing lows — demand building, buyers stepping in higher each dip",
        })

    # ── 2. Bull Flag — sharp advance then tight consolidation ────────────────
    if len(closes) >= 40:
        move_20 = (closes[-20] / closes[-40] - 1) if closes[-40] > 0 else 0
        range_5 = max(highs[-5:]) - min(lows[-5:])
        if move_20 > 0.07 and range_5 < atr * 1.3:
            patterns.append({
                "name": "Bull Flag",
                "confidence": 0.84,
                "desc": f"+{move_20*100:.0f}% advance, tight {atr:.2f} ATR consolidation — flag pole reset",
            })

    # ── 3. Volatility Squeeze — Bollinger contraction ──────────────────────
    if len(closes) >= 20:
        c20   = closes[-20:]
        mean  = sum(c20) / 20
        std20 = (sum((x - mean) ** 2 for x in c20) / 20) ** 0.5
        c5    = closes[-5:]
        mean5 = sum(c5) / 5
        std5  = (sum((x - mean5) ** 2 for x in c5) / 5) ** 0.5
        if std20 > 0 and std5 / std20 < 0.45:
            patterns.append({
                "name": "Volatility Squeeze",
                "confidence": 0.80,
                "desc": "Range contracting sharply — energy coiling, breakout likely",
            })

    # ── 4. Support Coil — price sitting just above key level ───────────────
    support_levels = [l for l in levels if l.get("type") == "support"]
    if support_levels:
        nearest = min(support_levels, key=lambda l: abs(l["price"] - closes[-1]))
        dist_pct = (closes[-1] - nearest["price"]) / nearest["price"] * 100
        if 0 < dist_pct < 3.0:
            patterns.append({
                "name": "Support Coil",
                "confidence": 0.91,
                "desc": f"Coiling {dist_pct:.1f}% above ${nearest['price']:.2f} support — compression before pop",
            })

    # ── 5. Volume Expansion — institutional accumulation signal ───────────
    if len(vols) >= 20:
        avg_vol    = sum(vols[-20:-3]) / 17
        recent_vol = sum(vols[-3:]) / 3
        if avg_vol > 0 and recent_vol > avg_vol * 1.6:
            patterns.append({
                "name": "Volume Surge",
                "confidence": 0.76,
                "desc": f"Recent volume {recent_vol/avg_vol:.1f}x 20-day avg — smart money accumulating",
            })

    # ── 6. Inside Day Compression ─────────────────────────────────────────
    ic = 0
    i  = len(bars) - 1
    while i > 0 and highs[i] < highs[i-1] and lows[i] > lows[i-1]:
        ic += 1
        i  -= 1
    if ic >= 2:
        patterns.append({
            "name": f"{ic}-Bar Inside Compression",
            "confidence": 0.73,
            "desc": f"{ic} consecutive inside bars — price coiling, directional break incoming",
        })

    patterns.sort(key=lambda p: -p["confidence"])
    return patterns[:4]


def _scan_one(sym: str, engine) -> dict | None:
    """Compute levels for one symbol; return a summary dict or None on failure."""
    try:
        result = engine.compute(sym)
        if "error" in result:
            return None

        current    = result["current_price"]
        atr        = result["atr"]
        all_gems   = result["gems"]
        pounce_lvls= result["pounce"]

        # Long-only filter: only support-side gems (price approaching support = buy)
        support_gems   = [g for g in all_gems    if g["type"] == "support"]
        support_pounce = [g for g in pounce_lvls if g["type"] == "support"]

        # Hard filters — drop the ticker entirely if it fails these:
        #  1. Downtrending stocks are NOT long setups. Skip bear tickers.
        #     (Inverse ETFs are kept — they ARE the bear play.)
        if result["trend"] == "down" and sym not in _INVERSE_SET:
            return None

        #  2. Must have at least one support gem within 3 ATR to be worth showing
        near_gems = [g for g in support_gems if g["atr_distance"] <= 3.0]
        if not near_gems and not support_pounce:
            return None

        # Best (closest) gem overall
        best = min(support_gems, key=lambda g: g["atr_distance"]) if support_gems else None

        # NOTE: Strategy detection (_detect_strategy) and MTF alignment (_mtf_alignment)
        # are NOT run during the mass scan.  Each call makes 1-2 network round-trips
        # (yfinance 6-month download + 2 AlphaVantage API calls per symbol).  With 20+
        # parallel workers scanning 223+ tickers the combined latency exceeds the 30s
        # request timeout.  These enrichment signals are fetched on-demand via
        # GET /quant/enrich/{symbol} when the user expands a specific card.
        strategies_matched: list[str] = []
        mtf: dict = {
            "15m": "doji", "1H": "doji", "4H": "doji", "1D": "doji",
            "aligned_bull": False, "aligned_bear": False, "score": 0, "source": "deferred",
        }

        # Look up historical quality for matched strategies (from universe backtest cache)
        from signal_engine.backtest_engine import STRATEGIES as _BT_STRATS
        strategy_quality: dict[str, dict] = {}
        if _bt_quality_cache and strategies_matched:
            sym_quality = _bt_quality_cache.get(sym, {})
            for s_name in strategies_matched:
                for key, cfg in _BT_STRATS.items():
                    if cfg["name"] == s_name and key in sym_quality:
                        q = sym_quality[key]
                        strategy_quality[s_name] = {
                            "win_rate":      q.get("win_rate", 0),
                            "profit_factor": q.get("profit_factor", 0),
                            "total_trades":  q.get("total", 0),
                        }
                        break

        # ── Quality gates ─────────────────────────────────────────────────────
        # READY requires: gem within 0.30 ATR + uptrend + R:R ≥ 1.5
        # A gem in a downtrending stock or with garbage R:R is NOT a buy setup.
        best_rr   = (best["trade_idea"]["rr"] if best and best.get("trade_idea") else 0)
        is_uptrend = result["trend"] in ("up", "neutral")   # "down" disqualifies
        quality_ok = is_uptrend and best_rr >= 1.5

        # Status tier
        if support_pounce and quality_ok:
            status = "READY"
        elif support_pounce or any(g["atr_distance"] <= 1.5 for g in support_gems):
            # Has a gem close by but didn't pass quality bar → NEAR
            status = "NEAR"
        else:
            status = "WATCH"

        # ── Fast pattern recognition (zero network calls) ─────────────────────
        patterns = _detect_patterns_from_bars(
            result.get("spark_bars", []),
            atr,
            result.get("levels", []),
        )

        # NOTE: GEX / dealer positioning is NOT fetched during the mass scan.
        # Fetching Yahoo options chains for 30+ symbols concurrently is too slow
        # and causes the thread pool to block.  GEX data is fetched on-demand
        # via GET /quant/gex/{symbol} when the user expands a specific card.
        # The `gex` field is always null here; the frontend shows an "Expand" link.
        gex_data: dict | None = None

        # Options recommendation — only for liquid $50+ stocks (not inverse ETFs)
        options_rec = None
        if current >= _OPTIONS_MIN_PRICE and sym not in _INVERSE_SET and best:
            target = best["trade_idea"]["target"] if best.get("trade_idea") else current * 1.05
            options_rec = _options_rec(sym, current, target)

        # ── Confluence score — aggregates all non-GEX factors ────────────────
        confluence_score = 0
        if best:
            confluence_score += min(best["strength"], 5)   # level strength 1–5
        if mtf["score"] > 0:
            confluence_score += mtf["score"]               # +1 per bullish TF (max +4)
        if strategies_matched:
            confluence_score += len(strategies_matched)    # +1 per triggered strategy
        if len(set(best.get("horizons", []))) >= 2 if best else False:
            confluence_score += 1                          # multi-horizon gem bonus

        return {
            "symbol":              sym,
            "price":               current,
            "atr":                 atr,
            "atr_pct":             result["atr_pct"],
            "trend":               result["trend"],
            "is_inverse_etf":      sym in _INVERSE_SET,
            "play_type":           "BEAR PLAY" if sym in _INVERSE_SET else "BULL",
            "status":              status,
            "confluence_score":    confluence_score,
            "gem_count":           len(support_gems),
            "ready_count":         len(support_pounce),
            "top_gem":             best,
            "ready_gems":          support_pounce,
            "gex":                 gex_data,               # dealer positioning inline
            "options_rec":         options_rec,
            "strategies_matched":  strategies_matched,
            "strategy_quality":    strategy_quality,
            "mtf":                 mtf,
            "spark_bars":          result.get("spark_bars", []),  # 60-bar OHLCV — free
            "patterns":            patterns,                       # fast pattern recognition
        }
    except Exception as exc:
        logger.debug("Pounce scan failed for %s: %s", sym, exc)
        return None


@router.get("/pounce")
def run_pounce_scan(
    include_bear: bool = Query(default=True),
    force:        bool = Query(default=False),
):
    """
    Pre-scan the full universe for long-only pounce setups.

    Runs LevelEngine in parallel across ~40 tickers. Returns only tickers
    that have support-side confluence gems within 2 ATR of current price.
    Results are cached for 10 minutes.

    Status:
      POUNCE — gem within 0.30 ATR, fire when ready
      NEAR   — gem within 1.0 ATR, get ready
      WATCH  — gem within 2.0 ATR, put on radar
    """
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache_key = f"pounce_{include_bear}"
    now = _time.time()
    if not force and cache_key in _pounce_cache:
        data, cached_at = _pounce_cache[cache_key]
        if now - cached_at < _POUNCE_TTL:
            return {**data, "cached": True}

    # ── Stage 1: build scan universe ─────────────────────────────────────────
    # Core: always scan the full curated list (223 symbols).
    # Extended: fetch today's active movers from FinnHub for every ticker
    #           NOT already in the core list. Done via fast batch quote calls
    #           (no yfinance batch download — that's too slow for 8k tickers).
    core = list(dict.fromkeys(_BULL_UNIVERSE))   # deduplicated, ordered
    core_set = set(core)

    full_tickers = _get_full_us_universe()        # 8k tickers, cached 24h

    # Sample the dynamic candidates so we make at most ~20 batch API calls
    # (2 000 tickers ÷ 100 per call = 20 calls).  We shuffle so we get
    # different tickers each run, giving broad market coverage over time.
    import random as _random
    _rng_extras = [t for t in full_tickers if t not in core_set]
    _random.shuffle(_rng_extras)
    extra_candidates = _rng_extras[:2000]

    extra = _fast_screen_finnhub(extra_candidates, max_extras=150)

    universe_raw = core + [t for t in extra if t not in core_set]
    if include_bear:
        for sym in _BEAR_UNIVERSE:
            if sym not in universe_raw:
                universe_raw.append(sym)

    # ── Volume pre-screen: drop illiquid tickers (< 500k shares/day) ─────────
    # Run a single FinnHub batch quote pass over the full core list so that
    # stocks like BYND, NKLA, SPCE etc. don't waste LevelEngine threads.
    # ETFs (SPY, QQQ…) are always kept regardless of volume.
    _ETF_KEEP = {
        "SPY","QQQ","IWM","DIA","MDY","GLD","SLV","USO","UNG",
        "XLF","XLE","XLK","XBI","GDX","XLV","XLI","XLC","XLU","XLRE",
        "ARKK","ARKF","ARKG","HYG","TLT","EMB","XRT","IBB","SOXX",
        "TQQQ","SQQQ","UVXY","VXX","SPXU","UPRO","LABU","LABD",
        "YINN","YANG","TNA","TZA","FNGU","FNGD","TECL","TECS",
    }
    try:
        from api.routers.settings import _load as _load_settings
        _fh_key = _load_settings().get("finnhub", {}).get("api_key", "")
        if _fh_key:
            # Only screen the non-ETF tickers (ETFs always liquid)
            _to_screen   = [s for s in universe_raw if s not in _ETF_KEEP]
            _always_keep = [s for s in universe_raw if s in _ETF_KEEP]
            _quotes = _finnhub_batch_quote(_to_screen, _fh_key)
            _liquid = []
            for s in _to_screen:
                q = _quotes.get(s)
                if q is None:
                    _liquid.append(s)   # no FinnHub data → keep (big-cap or new)
                    continue
                vol = q.get("v", 0) or 0
                if vol >= 500_000:
                    _liquid.append(s)
                else:
                    logger.debug("Volume filter removed %s (vol=%d)", s, vol)
            universe = _always_keep + _liquid
            logger.info(
                "Volume filter: %d → %d symbols (removed %d illiquid)",
                len(universe_raw), len(universe), len(universe_raw) - len(universe),
            )
        else:
            universe = universe_raw
    except Exception as _ve:
        logger.warning("Volume pre-screen failed (%s) — using full universe", _ve)
        universe = universe_raw

    logger.info("Pounce deep-scan universe: %d symbols (%d core + %d dynamic)",
                len(universe), len(core), len(extra))

    # ── Stage 2: level engine deep scan ──────────────────────────────────────
    engine  = _get_level_engine()
    results = []

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(_scan_one, sym, engine): sym for sym in universe}
        for fut in as_completed(futures, timeout=180):
            try:
                res = fut.result()
                if res is not None:
                    results.append(res)
            except Exception as exc:
                logger.debug("Pounce scan future failed: %s", exc)

    # Sort: READY first → NEAR → WATCH; then by confluence_score desc, atr_distance asc
    order = {"READY": 0, "NEAR": 1, "WATCH": 2}
    results.sort(key=lambda r: (
        order.get(r["status"], 3),
        -r.get("confluence_score", 0),
        r["top_gem"]["atr_distance"] if r["top_gem"] else 99,
    ))

    data = {
        "cached":    False,
        "timestamp": now,
        "scanned":   len(universe),
        "surfaced":  len(results),
        "ready":     sum(1 for r in results if r["status"] == "READY"),
        "near":      sum(1 for r in results if r["status"] == "NEAR"),
        "watch":     sum(1 for r in results if r["status"] == "WATCH"),
        "results":   results,
    }
    _pounce_cache[cache_key] = (data, now)
    return data


# ── Level Mapping ─────────────────────────────────────────────────────────────

@router.get("/levels/{symbol}")
def get_levels(symbol: str):
    """
    Multi-horizon price level map for any ticker.

    Returns all significant levels tagged by horizon (scalp/day/trend),
    confluence "gems" (high-strength or multi-horizon clusters), and
    "pounce" levels where price is currently within 0.30 ATR.
    """
    try:
        result = _get_level_engine().compute(symbol.upper())
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Level engine failed for %s: %s", symbol, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── On-demand enrichment (MTF + strategy + GEX) ───────────────────────────────

@router.get("/enrich/{symbol}")
def enrich_symbol(symbol: str):
    """
    On-demand deep enrichment for a single symbol.

    Returns MTF alignment, matched strategies + quality stats, and GEX
    dealer-positioning data (if options are available).  Called per-card
    when the user wants the full confluence picture for a specific ticker.
    """
    sym = symbol.upper()
    try:
        # Multi-timeframe directional alignment
        mtf = _mtf_alignment(sym)

        # Backtested strategy detection (last 3 bars)
        strategies_matched = _detect_strategy(sym)

        # Strategy quality from backtest cache (if available)
        from signal_engine.backtest_engine import STRATEGIES as _BT_STRATS
        strategy_quality: dict[str, dict] = {}
        if _bt_quality_cache and strategies_matched:
            sym_quality = _bt_quality_cache.get(sym, {})
            for s_name in strategies_matched:
                for key, cfg in _BT_STRATS.items():
                    if cfg["name"] == s_name and key in sym_quality:
                        q = sym_quality[key]
                        strategy_quality[s_name] = {
                            "win_rate":      q.get("win_rate", 0),
                            "profit_factor": q.get("profit_factor", 0),
                            "total_trades":  q.get("total", 0),
                        }
                        break

        # GEX / dealer positioning (graceful fail — not all tickers have options)
        gex_data: dict | None = None
        try:
            from data_engine.yahoo_options_fetcher import fetch_options_chain
            from options_engine.gamma_exposure import GammaExposureEngine
            chain  = fetch_options_chain(sym)
            gex    = GammaExposureEngine().compute(chain)
            gex_data = {
                "zero_gamma":  getattr(gex, "zero_gamma",  None),
                "call_wall":   getattr(gex, "call_wall",   None),
                "put_wall":    getattr(gex, "put_wall",    None),
                "max_pain":    getattr(gex, "max_pain",    None),
                "net_gex":     getattr(gex, "net_gex",     None),
                "dealer_position": getattr(gex, "dealer_position", None),
            }
        except Exception:
            pass  # non-optionable ticker — omit GEX silently

        return {
            "symbol":            sym,
            "mtf":               mtf,
            "strategies_matched": strategies_matched,
            "strategy_quality":  strategy_quality,
            "gex":               gex_data,
        }
    except Exception as e:
        logger.error("Enrich failed for %s: %s", sym, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Sparkline ─────────────────────────────────────────────────────────────────
# Fast 1-hour-cached endpoint that returns daily OHLCV bars.
# Used by the hover tooltip (60-bar area sparkline) and the SymbolDrawer chart.

_sparkline_cache: dict[str, tuple[list, float]] = {}
_SPARKLINE_TTL = 3600.0  # 1 hour — historical bars don't need intraday refresh


@router.get("/sparkline/{symbol}")
def get_sparkline(symbol: str, days: int = Query(default=60)):
    """Return last N daily OHLCV bars.  Cached 1 hour per ticker."""
    sym = symbol.upper()
    now = time.time()
    cached = _sparkline_cache.get(sym)
    if cached:
        bars, cached_at = cached
        if now - cached_at < _SPARKLINE_TTL:
            return {"symbol": sym, "bars": bars, "cached": True}
    try:
        import yfinance as yf
        df = yf.download(sym, period="3mo", interval="1d", progress=False,
                         auto_adjust=True, threads=False)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {sym}")
        df = df.tail(max(days, 1))
        bars: list[dict] = []
        for dt, row in df.iterrows():
            bars.append({
                "date":   str(dt)[:10],
                "open":   round(float(row["Open"]),  2),
                "high":   round(float(row["High"]),  2),
                "low":    round(float(row["Low"]),   2),
                "close":  round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            })
        _sparkline_cache[sym] = (bars, now)
        return {"symbol": sym, "bars": bars, "cached": False}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Sparkline failed for %s: %s", sym, e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Backtesting ───────────────────────────────────────────────────────────────

@router.get("/backtest/{symbol}")
def run_backtest(
    symbol: str,
    period: str = Query(default="2y"),
):
    """
    Backtest all strategies on a single symbol over the given period.
    Returns per-strategy stats + equity curve + trade log.
    """
    try:
        result = _get_backtest_engine().run(symbol.upper(), period=period)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Backtest failed for %s: %s", symbol, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


_bt_universe_cache: dict[str, tuple[dict, float]] = {}
_BT_CACHE_TTL = 3600.0  # 1 hour — backtests don't change fast


@router.get("/backtest-universe")
def run_backtest_universe(
    period: str  = Query(default="2y"),
    force:  bool = Query(default=False),
):
    """
    Backtest all strategies across the full bull universe.
    Returns per-strategy aggregate stats + overall equity curve.
    Cached for 1 hour.
    """
    import time as _time
    cache_key = f"bt_{period}"
    now = _time.time()
    if not force and cache_key in _bt_universe_cache:
        data, cached_at = _bt_universe_cache[cache_key]
        if now - cached_at < _BT_CACHE_TTL:
            return {**data, "cached": True}

    try:
        result = _get_backtest_engine().run_universe(
            _BULL_UNIVERSE, period=period,
        )
        result["cached"] = False
        _bt_universe_cache[cache_key] = (result, now)

        # Populate quality cache so pounce scanner can show historical win rates
        global _bt_quality_cache
        for sym_key, sym_data in result.get("per_symbol", {}).items():
            if "per_strategy" in sym_data:
                _bt_quality_cache[sym_key] = sym_data["per_strategy"]
        logger.info("Backtest quality cache populated: %d symbols", len(_bt_quality_cache))

        return result
    except Exception as e:
        logger.error("Universe backtest failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
