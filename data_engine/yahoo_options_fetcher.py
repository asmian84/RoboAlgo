"""
RoboAlgo — Yahoo Finance Options Fetcher
Fetches full options chains from yfinance and pre-computes raw GEX inputs.

Responsibilities:
  • Pull all expirations for a symbol via yfinance.
  • Normalise calls/puts into a unified OptionsChain dataclass.
  • Pre-aggregate per-strike open interest × gamma ready for GEX engine.
  • Cache with configurable TTL (default 15 min — delayed data).
  • Expose batch fetching with a thread-pool for scanner throughput.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CHAIN_TTL   = 900    # 15 min  (yfinance is delayed ~15 min)
_MAX_WORKERS = 6
_CHAIN_CACHE: dict[str, tuple["OptionsChain", float]] = {}

CONTRACT_SIZE = 100   # standard US equity option


# ─────────────────────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptionContract:
    strike:           float
    expiration:       str       # "YYYY-MM-DD"
    option_type:      str       # "call" | "put"
    last_price:       float
    bid:              float
    ask:              float
    volume:           int
    open_interest:    int
    implied_vol:      float     # annualised IV (decimal)
    delta:            float
    gamma:            float
    theta:            float
    vega:             float
    in_the_money:     bool


@dataclass
class StrikeGEXRow:
    """Pre-aggregated per-strike GEX inputs consumed by GammaExposureEngine."""
    strike:                 float
    call_oi:                int
    put_oi:                 int
    call_gamma:             float     # avg gamma across calls at this strike
    put_gamma:              float
    call_gex:               float     # gamma × oi × contract_size × spot²
    put_gex:                float
    net_gex:                float     # call_gex − put_gex (dealer perspective)


@dataclass
class OptionsChain:
    symbol:          str
    spot:            float
    fetch_time:      float                          # unix timestamp
    contracts:       list[OptionContract] = field(default_factory=list)
    expirations:     list[str]            = field(default_factory=list)
    strike_gex:      list[StrikeGEXRow]   = field(default_factory=list)

    # Derived analytics (populated by _compute_analytics)
    call_wall:       float = 0.0
    put_wall:        float = 0.0
    zero_gamma:      float = 0.0
    gamma_regime:    str   = "NEUTRAL"   # POSITIVE | NEGATIVE | NEUTRAL
    total_call_oi:   int   = 0
    total_put_oi:    int   = 0
    pc_ratio:        float = 0.0
    max_pain:        float = 0.0
    top_gex_strikes: list[dict] = field(default_factory=list)
    # 0DTE isolation (same-day expiry)
    strike_gex_0dte:     list[StrikeGEXRow] = field(default_factory=list)
    gex_0dte_call_wall:  float = 0.0
    gex_0dte_put_wall:   float = 0.0
    gex_0dte_pins:       list[dict] = field(default_factory=list)  # proximity-weighted
    # Per-expiry term structure
    gex_by_expiry:   dict[str, list[StrikeGEXRow]] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
#  Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v or 0))
    except (TypeError, ValueError):
        return default


def _parse_chain_df(df: pd.DataFrame, expiry: str, opt_type: str) -> list[OptionContract]:
    contracts: list[OptionContract] = []
    for _, row in df.iterrows():
        contracts.append(OptionContract(
            strike        = _safe_float(row.get("strike")),
            expiration    = expiry,
            option_type   = opt_type,
            last_price    = _safe_float(row.get("lastPrice")),
            bid           = _safe_float(row.get("bid")),
            ask           = _safe_float(row.get("ask")),
            volume        = _safe_int(row.get("volume")),
            open_interest = _safe_int(row.get("openInterest")),
            implied_vol   = _safe_float(row.get("impliedVolatility")),
            delta         = _safe_float(row.get("delta")),
            gamma         = _safe_float(row.get("gamma")),
            theta         = _safe_float(row.get("theta")),
            vega          = _safe_float(row.get("vega")),
            in_the_money  = bool(row.get("inTheMoney", False)),
        ))
    return contracts


# ─────────────────────────────────────────────────────────────────────────────
#  GEX aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _bsm_gamma_fallback(c: "OptionContract", spot: float, r: float = 0.045) -> float:
    """
    Compute BSM gamma when yfinance gamma is zero/missing (common for OTM strikes).
    Uses mid-price to back out IV, then calculates gamma analytically.
    Falls back to implied_vol from contract IV if mid-price is zero.
    """
    from options_engine.greeks_calculator import bs_gamma, implied_vol_approx
    import datetime

    try:
        # Time to expiry
        exp = datetime.date.fromisoformat(c.expiration)
        T   = max((exp - datetime.date.today()).days / 365.0, 1 / 365.0)

        # Determine IV: prefer yfinance IV, fall back to mid-price solver
        iv = c.implied_vol
        if iv <= 0.001:
            mid = (c.bid + c.ask) / 2.0 if c.ask > 0 else c.last_price
            if mid > 0.001:
                iv = implied_vol_approx(mid, spot, c.strike, T, r, c.option_type)
        if iv <= 0.001:
            return 0.0

        return bs_gamma(spot, c.strike, T, r, iv)
    except Exception:
        return 0.0


def _build_strike_gex(contracts: list[OptionContract], spot: float) -> list[StrikeGEXRow]:
    """
    Aggregate per-strike GEX rows.

    GEX formula (dealer perspective — dealers are short options):
        call_gex = +gamma × call_OI × CONTRACT_SIZE × spot²
        put_gex  = −gamma × put_OI  × CONTRACT_SIZE × spot²
        net_gex  = call_gex + put_gex

    Positive net_gex → dealers long gamma → they sell rallies / buy dips
                       → dampened moves (mean-reverting)
    Negative net_gex → dealers short gamma → they buy rallies / sell dips
                       → accelerated moves (trending / explosive)

    IV fallback: when yfinance gamma = 0 (common for OTM strikes), gamma is
    recomputed from mid-price using Black-Scholes via `_bsm_gamma_fallback()`.
    """
    from collections import defaultdict

    call_oi:     dict[float, int]   = defaultdict(int)
    put_oi:      dict[float, int]   = defaultdict(int)
    call_gamma:  dict[float, list]  = defaultdict(list)
    put_gamma:   dict[float, list]  = defaultdict(list)

    for c in contracts:
        # IV fallback: use BSM when yfinance gamma is absent/zero
        gamma = c.gamma
        if gamma <= 0.0 and c.open_interest > 0:
            gamma = _bsm_gamma_fallback(c, spot)

        if c.option_type == "call":
            call_oi[c.strike]    += c.open_interest
            call_gamma[c.strike].append(gamma)
        else:
            put_oi[c.strike]     += c.open_interest
            put_gamma[c.strike].append(gamma)

    all_strikes = sorted(set(call_oi) | set(put_oi))
    rows: list[StrikeGEXRow] = []
    spot_sq = spot * spot

    for strike in all_strikes:
        coi = call_oi.get(strike, 0)
        poi = put_oi.get(strike, 0)
        cg  = float(np.mean(call_gamma[strike])) if call_gamma[strike] else 0.0
        pg  = float(np.mean(put_gamma[strike]))  if put_gamma[strike]  else 0.0

        c_gex = cg  * coi * CONTRACT_SIZE * spot_sq
        p_gex = -pg * poi * CONTRACT_SIZE * spot_sq
        net   = c_gex + p_gex

        rows.append(StrikeGEXRow(
            strike     = strike,
            call_oi    = coi,
            put_oi     = poi,
            call_gamma = cg,
            put_gamma  = pg,
            call_gex   = c_gex,
            put_gex    = p_gex,
            net_gex    = net,
        ))

    return rows


def _compute_analytics(chain: OptionsChain) -> None:
    """Populate derived fields on an OptionsChain in-place."""
    if not chain.strike_gex:
        return

    strikes   = np.array([r.strike   for r in chain.strike_gex])
    net_gex   = np.array([r.net_gex  for r in chain.strike_gex])
    call_gex  = np.array([r.call_gex for r in chain.strike_gex])
    put_gex   = np.array([abs(r.put_gex) for r in chain.strike_gex])
    call_oi   = np.array([r.call_oi  for r in chain.strike_gex])
    put_oi    = np.array([r.put_oi   for r in chain.strike_gex])

    # Call wall = strike with highest total call OI (resistance)
    chain.call_wall = float(strikes[np.argmax(call_oi)]) if len(call_oi) else 0.0

    # Put wall  = strike with highest total put OI (support)
    chain.put_wall  = float(strikes[np.argmax(put_oi)])  if len(put_oi)  else 0.0

    # Zero-gamma level = strike where cumulative net GEX crosses zero
    # (sort by strike, cumsum, find first sign change)
    order     = np.argsort(strikes)
    s_sorted  = strikes[order]
    g_sorted  = net_gex[order]
    cum_gex   = np.cumsum(g_sorted)
    zg_idx    = np.where(np.diff(np.sign(cum_gex)))[0]
    if len(zg_idx):
        # linear interpolation between the two bracketing strikes
        i = zg_idx[0]
        g1, g2 = cum_gex[i], cum_gex[i + 1]
        s1, s2 = s_sorted[i], s_sorted[i + 1]
        chain.zero_gamma = float(s1 + (s2 - s1) * (-g1) / (g2 - g1)) if g2 != g1 else float(s1)
    else:
        chain.zero_gamma = float(s_sorted[np.argmin(np.abs(cum_gex))]) if len(s_sorted) else 0.0

    # Gamma regime
    total_net = float(np.sum(net_gex))
    if total_net > 0:
        chain.gamma_regime = "POSITIVE"    # dealers long gamma → dampening
    elif total_net < 0:
        chain.gamma_regime = "NEGATIVE"    # dealers short gamma → amplifying
    else:
        chain.gamma_regime = "NEUTRAL"

    chain.total_call_oi = int(np.sum(call_oi))
    chain.total_put_oi  = int(np.sum(put_oi))
    chain.pc_ratio      = round(chain.total_put_oi / max(chain.total_call_oi, 1), 3)

    # Max pain
    all_strikes_list = sorted(set(strikes.tolist()))
    min_pain   = float("inf")
    max_pain_s = 0.0
    for ts in all_strikes_list:
        pain = sum(
            max(0.0, ts - r.strike) * r.call_oi + max(0.0, r.strike - ts) * r.put_oi
            for r in chain.strike_gex
        )
        if pain < min_pain:
            min_pain   = pain
            max_pain_s = ts
    chain.max_pain = max_pain_s

    # Top 5 GEX strikes by absolute net GEX
    gex_tuples = sorted(
        zip(strikes.tolist(), net_gex.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:5]
    chain.top_gex_strikes = [
        {"strike": s, "net_gex": round(g, 2)} for s, g in gex_tuples
    ]

    # ── 0DTE isolation ────────────────────────────────────────────────────
    import datetime
    today_str = str(datetime.date.today())
    dte0_contracts = [c for c in chain.contracts if c.expiration == today_str]
    if dte0_contracts:
        chain.strike_gex_0dte = _build_strike_gex(dte0_contracts, chain.spot)
        if chain.strike_gex_0dte:
            c0_oi = np.array([r.call_oi for r in chain.strike_gex_0dte])
            p0_oi = np.array([r.put_oi  for r in chain.strike_gex_0dte])
            s0    = np.array([r.strike  for r in chain.strike_gex_0dte])
            if c0_oi.sum() > 0:
                chain.gex_0dte_call_wall = float(s0[np.argmax(c0_oi)])
            if p0_oi.sum() > 0:
                chain.gex_0dte_put_wall  = float(s0[np.argmax(p0_oi)])
            # Proximity-weighted pin ranking (1/distance from spot)
            spot = chain.spot
            pins = []
            for r in chain.strike_gex_0dte:
                dist   = abs(r.strike - spot)
                weight = 1.0 / max(dist, 0.01)
                pins.append({
                    "strike":        r.strike,
                    "net_gex":       round(r.net_gex / 1e6, 2),
                    "proximity_wt":  round(weight, 4),
                    "pin_score":     round(abs(r.net_gex) * weight / 1e6, 4),
                    "dist_pct":      round(dist / max(spot, 1e-6) * 100, 2),
                })
            chain.gex_0dte_pins = sorted(pins, key=lambda x: x["pin_score"], reverse=True)[:5]

    # ── Per-expiry term structure ─────────────────────────────────────────
    from collections import defaultdict as _dd
    by_exp: dict[str, list] = _dd(list)
    for c in chain.contracts:
        by_exp[c.expiration].append(c)
    for exp, exp_contracts in sorted(by_exp.items()):
        chain.gex_by_expiry[exp] = _build_strike_gex(exp_contracts, chain.spot)


# ─────────────────────────────────────────────────────────────────────────────
#  Public fetching API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_options_chain(
    symbol: str,
    force_refresh: bool = False,
    max_expirations: int = 4,
) -> OptionsChain | None:
    """
    Fetch and parse the full options chain for *symbol* via yfinance.
    Returns an OptionsChain with pre-computed GEX analytics, or None on error.
    TTL-cached per symbol.
    """
    sym = symbol.upper()
    now = time.time()
    cached = _CHAIN_CACHE.get(sym)
    if not force_refresh and cached and (now - cached[1]) < _CHAIN_TTL:
        return cached[0]

    try:
        import yfinance as yf
        tk  = yf.Ticker(sym)
        inf = tk.fast_info

        spot = float(getattr(inf, "last_price", 0) or getattr(inf, "regularMarketPrice", 0) or 0)
        if spot <= 0:
            # fall back to history
            hist = tk.history(period="1d", interval="1d")
            spot = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
        if spot <= 0:
            logger.debug("No spot price for %s", sym)
            return None

        expirations = list(tk.options or [])[:max_expirations]
        if not expirations:
            return None

        all_contracts: list[OptionContract] = []
        for expiry in expirations:
            try:
                opt = tk.option_chain(expiry)
                all_contracts.extend(_parse_chain_df(opt.calls, expiry, "call"))
                all_contracts.extend(_parse_chain_df(opt.puts,  expiry, "put"))
            except Exception as exc:
                logger.debug("option_chain(%s, %s): %s", sym, expiry, exc)

        if not all_contracts:
            return None

        strike_gex = _build_strike_gex(all_contracts, spot)

        chain = OptionsChain(
            symbol      = sym,
            spot        = spot,
            fetch_time  = now,
            contracts   = all_contracts,
            expirations = expirations,
            strike_gex  = strike_gex,
        )
        _compute_analytics(chain)

        _CHAIN_CACHE[sym] = (chain, now)
        return chain

    except Exception as exc:
        logger.warning("fetch_options_chain(%s): %s", sym, exc)
        return None


def fetch_bulk_chains(
    symbols: list[str],
    max_workers: int = _MAX_WORKERS,
    max_expirations: int = 4,
) -> dict[str, OptionsChain]:
    """
    Parallel options-chain fetch for many symbols.
    Returns {symbol: OptionsChain} for successful fetches only.
    """
    results: dict[str, OptionsChain] = {}

    def _worker(sym: str) -> tuple[str, OptionsChain | None]:
        return sym, fetch_options_chain(sym, max_expirations=max_expirations)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_worker, s): s for s in symbols}
        for fut in as_completed(futs):
            sym, chain = fut.result()
            if chain is not None:
                results[sym] = chain

    logger.info("fetch_bulk_chains: %d/%d successful", len(results), len(symbols))
    return results


def chain_to_dict(chain: OptionsChain) -> dict:
    """Serialise OptionsChain to a JSON-safe dict for the API layer."""
    return {
        "symbol":          chain.symbol,
        "spot":            chain.spot,
        "call_wall":       chain.call_wall,
        "put_wall":        chain.put_wall,
        "zero_gamma":      chain.zero_gamma,
        "gamma_regime":    chain.gamma_regime,
        "total_call_oi":   chain.total_call_oi,
        "total_put_oi":    chain.total_put_oi,
        "pc_ratio":        chain.pc_ratio,
        "max_pain":        chain.max_pain,
        "top_gex_strikes": chain.top_gex_strikes,
        "expirations":     chain.expirations,
        "strike_gex": [
            {
                "strike":     r.strike,
                "call_oi":    r.call_oi,
                "put_oi":     r.put_oi,
                "call_gex":   round(r.call_gex, 2),
                "put_gex":    round(r.put_gex, 2),
                "net_gex":    round(r.net_gex, 2),
            }
            for r in chain.strike_gex
        ],
    }
