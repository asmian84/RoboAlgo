"""
Real Gamma Squeeze Scanner
==========================
Two-pass approach:
  Pass 1: Price action pre-filter using OHLCV data
          → Only stocks near 52-week lows with oversold RSI proceed

  Pass 2: Fetch live options chains via yfinance for each candidate.
          Calculate dealer delta/gamma exposure using Black-Scholes:
            net_delta_hedge = Σ(call_delta × call_OI × 100)
                            + Σ(put_delta × put_OI × 100)
            net_gex         = Σ(gamma × call_OI × 100 × S²/100)
                            − Σ(gamma × put_OI × 100 × S²/100)

  Negative GEX at 52-week lows = dealers short gamma → explosive squeeze on reversal.
  Based on: https://github.com/saakethkoka/GammaSqueezeIdentifier
"""

from fastapi import APIRouter, Query
import logging
import asyncio
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

router = APIRouter(tags=["sniper_gamma"])
logger = logging.getLogger(__name__)

# Global cache
_last_scan: list[dict] = []
_last_scan_time: float = 0
_scan_in_progress: bool = False
_SCAN_TTL = 20 * 60  # 20 min — options data is slower

# ── Black-Scholes helpers ─────────────────────────────────────────────────────

def _bs_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

def _bs_delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    from scipy.stats import norm
    if T < 1e-5 or sigma < 1e-5 or S <= 0 or K <= 0:
        return (1.0 if S > K else 0.0) if is_call else (-1.0 if S < K else 0.0)
    try:
        d1 = _bs_d1(S, K, T, r, sigma)
        return float(norm.cdf(d1)) if is_call else float(norm.cdf(d1) - 1.0)
    except Exception:
        return 0.5 if is_call else -0.5

def _bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    from scipy.stats import norm
    if T < 1e-5 or sigma < 1e-5 or S <= 0 or K <= 0:
        return 0.0
    try:
        d1 = _bs_d1(S, K, T, r, sigma)
        return float(norm.pdf(d1)) / (S * sigma * math.sqrt(T))
    except Exception:
        return 0.0

def _hist_vol(close_arr) -> float:
    """21-day annualised historical volatility"""
    import numpy as np
    if len(close_arr) < 5:
        return 0.35
    rets = np.diff(np.log(np.maximum(close_arr[-22:], 1e-6)))
    hv = float(np.std(rets) * math.sqrt(252)) if len(rets) >= 3 else 0.35
    return max(0.05, min(hv, 4.0))   # clamp 5%–400%

# ── Per-stock options GEX calculator ─────────────────────────────────────────

def _fetch_gex(sym: str, spot: float, hist_vol_val: float, risk_free: float = 0.05) -> Optional[dict]:
    """
    Fetch options chain for `sym` and compute:
      net_delta_hedge  – total shares dealers hold to stay delta-neutral
      net_gex          – net gamma exposure (negative = short gamma = explosive)
      put_call_ratio   – overall P/C OI ratio
      near_atm_pcr     – P/C in ±15% of spot (most impactful)
    """
    import yfinance as yf
    from datetime import date, datetime

    try:
        t = yf.Ticker(sym)
        exps = t.options
        if not exps:
            return None

        today = date.today()
        net_dh = 0.0          # net dealer delta hedge (shares)
        net_gex = 0.0         # net gamma exposure ($gamma × OI)
        total_call_oi = 0
        total_put_oi  = 0
        atm_call_oi   = 0
        atm_put_oi    = 0

        atm_lo = spot * 0.85
        atm_hi = spot * 1.20

        count = 0
        for exp_str in exps:
            try:
                dte = (datetime.strptime(exp_str, '%Y-%m-%d').date() - today).days
            except ValueError:
                continue
            if dte < 0:
                continue
            if dte > 60:      # only near-term (< 60 DTE) drives gamma
                break
            if count >= 6:
                break
            count += 1
            T = max(dte / 365.0, 1e-4)

            chain = t.option_chain(exp_str)

            for row in chain.calls.itertuples(index=False):
                K   = float(row.strike)
                oi  = int(row.openInterest) if (row.openInterest == row.openInterest and row.openInterest) else 0
                if oi == 0:
                    continue
                iv = float(row.impliedVolatility) if (
                    row.impliedVolatility == row.impliedVolatility and
                    row.impliedVolatility > 0.01
                ) else hist_vol_val
                iv = min(iv, 5.0)   # cap crazy IV values

                d  = _bs_delta(spot, K, T, risk_free, iv, True)
                g  = _bs_gamma(spot, K, T, risk_free, iv)
                net_dh  += d * oi * 100
                net_gex += g * oi * 100 * spot * spot / 100
                total_call_oi += oi
                if atm_lo <= K <= atm_hi:
                    atm_call_oi += oi

            for row in chain.puts.itertuples(index=False):
                K   = float(row.strike)
                oi  = int(row.openInterest) if (row.openInterest == row.openInterest and row.openInterest) else 0
                if oi == 0:
                    continue
                iv = float(row.impliedVolatility) if (
                    row.impliedVolatility == row.impliedVolatility and
                    row.impliedVolatility > 0.01
                ) else hist_vol_val
                iv = min(iv, 5.0)

                d  = _bs_delta(spot, K, T, risk_free, iv, False)   # negative
                g  = _bs_gamma(spot, K, T, risk_free, iv)
                net_dh  += d * oi * 100                             # puts pull hedge negative
                net_gex -= g * oi * 100 * spot * spot / 100         # dealers short puts → negative GEX
                total_put_oi += oi
                if atm_lo <= K <= atm_hi:
                    atm_put_oi += oi

        if total_call_oi + total_put_oi == 0:
            return None

        pcr      = total_put_oi / total_call_oi if total_call_oi > 0 else (10.0 if total_put_oi > 0 else 0.0)
        near_pcr = atm_put_oi   / atm_call_oi   if atm_call_oi   > 0 else (10.0 if atm_put_oi  > 0 else 0.0)

        return {
            'net_delta_hedge': net_dh,
            'net_gex':         net_gex,
            'put_call_ratio':  round(pcr, 3),
            'near_atm_pcr':    round(near_pcr, 3),
            'total_call_oi':   total_call_oi,
            'total_put_oi':    total_put_oi,
            'atm_call_oi':     atm_call_oi,
            'atm_put_oi':      atm_put_oi,
        }

    except Exception as e:
        logger.debug("GEX fetch failed %s: %s", sym, e)
        return None

# ── Full scan ─────────────────────────────────────────────────────────────────

def _run_gamma_squeeze_scan(top_n: int = 200, min_volume: float = 50_000) -> list[dict]:
    from data_engine.alphavantage_fetcher import load_ohlcv_from_db
    from config.settings import get_all_instruments
    import numpy as np

    syms = get_all_instruments()
    logger.info("Gamma squeeze scan: loading OHLCV for %d symbols …", len(syms))

    price_data = load_ohlcv_from_db(syms, lookback_days=400)

    # ── Pass 1: Price action pre-filter ───────────────────────────────────────
    candidates = []

    for sym, df in price_data.items():
        if df is None or len(df) < 30:
            continue
        try:
            close  = df['close'].values
            volume = df['volume'].values
            low    = df['low'].values

            if volume[-1] < min_volume:
                continue
            spot = float(close[-1])
            if spot <= 0:
                continue

            # Wilder RSI-14
            delta = np.diff(close)
            gain  = np.where(delta > 0, delta, 0.0)
            loss  = np.where(delta < 0, -delta, 0.0)
            if len(gain) >= 14:
                ag = np.mean(gain[:14]); al = np.mean(loss[:14])
                for g, l in zip(gain[14:], loss[14:]):
                    ag = (ag * 13 + g) / 14
                    al = (al * 13 + l) / 14
                rsi = 100 - (100 / (1 + ag / al)) if al > 0 else 100.0
            else:
                rsi = 50.0

            # Distance from historical lows
            low_all = float(np.min(low))
            dist_all = (spot - low_all) / low_all * 100 if low_all > 0 else 999

            # Pre-filter: must be within 50% of historical low + some oversold signal
            if dist_all > 50:
                continue
            if rsi > 55:
                continue

            hv = _hist_vol(close)

            candidates.append({
                'sym':      sym,
                'spot':     spot,
                'rsi':      rsi,
                'dist_all': dist_all,
                'hv':       hv,
                'vol':      float(volume[-1]),
            })
        except Exception:
            continue

    logger.info("Gamma scan: %d candidates after price-action filter, fetching options …", len(candidates))

    # ── Pass 2: Options chain + GEX ──────────────────────────────────────────
    results = []

    def _process_one(c: dict) -> Optional[dict]:
        gex_data = _fetch_gex(c['sym'], c['spot'], c['hv'])
        if gex_data is None:
            return None   # no options → skip (technical scanner handles these)

        spot      = c['spot']
        net_dh    = gex_data['net_delta_hedge']
        net_gex   = gex_data['net_gex']
        pcr       = gex_data['put_call_ratio']
        near_pcr  = gex_data['near_atm_pcr']
        total_oi  = (gex_data['total_call_oi'] + gex_data['total_put_oi']) * 100
        atm_oi    = (gex_data['atm_call_oi']   + gex_data['atm_put_oi'])   * 100

        # ── Scoring ──────────────────────────────────────────────────────────
        # 1. Support score — closer to historical low = better
        if c['dist_all'] < 10:
            support_score = 95
        elif c['dist_all'] < 20:
            support_score = 80
        elif c['dist_all'] < 35:
            support_score = 60
        else:
            support_score = 40

        # 2. Oversold score
        rsi = c['rsi']
        if rsi < 20:
            rsi_score = 100
        elif rsi < 30:
            rsi_score = 80
        elif rsi < 40:
            rsi_score = 60
        elif rsi < 55:
            rsi_score = 35
        else:
            rsi_score = 0

        # 3. GEX score — negative GEX = explosive potential
        neg_gex   = net_gex < 0
        gex_mag   = abs(net_gex)
        gex_norm  = (gex_mag / (total_oi * spot / 100)) if total_oi > 0 and spot > 0 else 0
        gex_score = min(100, gex_norm * 500) if neg_gex else 0

        # 4. Put/call skew — heavy put OI at ATM = dealers forced to buy on reversal
        pcr_score = 0
        if near_pcr > 1.5:
            pcr_score = min(100, (near_pcr - 1.0) * 30)  # peaks at 100 around PCR 4+

        # 5. Net dealer hedge magnitude
        dh_norm   = abs(net_dh) / (c['vol'] + 1)  # hedge as multiple of 1-day volume
        dh_score  = min(100, dh_norm * 10)

        # Composite gamma squeeze score
        gamma_squeeze_score = (
            gex_score  * 0.35 +
            pcr_score  * 0.25 +
            dh_score   * 0.20 +
            support_score * 0.12 +
            rsi_score  * 0.08
        )

        # Require some minimum gamma signal
        if gex_score < 5 and pcr_score < 10:
            return None

        notes = [
            f"GEX: {'🔴 NEGATIVE' if neg_gex else '🟢 positive'}  ({net_gex:,.0f})",
            f"Net dealer hedge: {net_dh:+,.0f} shares",
            f"ATM P/C: {near_pcr:.2f}  |  All P/C: {pcr:.2f}",
            f"RSI: {rsi:.0f}  |  {c['dist_all']:.1f}% from historical low",
        ]

        return {
            'symbol':              c['sym'],
            'current_price':       round(spot, 2),
            'gamma_squeeze_score': round(gamma_squeeze_score, 1),
            'gex_score':           round(gex_score, 1),
            'pcr_score':           round(pcr_score, 1),
            'dh_score':            round(dh_score, 1),
            'support_score':       round(support_score, 1),
            'rsi_score':           round(rsi_score, 1),
            'rsi':                 round(rsi, 1),
            'dist_from_low_pct':   round(c['dist_all'], 1),
            'net_gex':             round(net_gex, 0),
            'net_delta_hedge':     round(net_dh, 0),
            'put_call_ratio':      gex_data['put_call_ratio'],
            'near_atm_pcr':        gex_data['near_atm_pcr'],
            'total_call_oi':       gex_data['total_call_oi'],
            'total_put_oi':        gex_data['total_put_oi'],
            'atm_call_oi':         gex_data['atm_call_oi'],
            'atm_put_oi':          gex_data['atm_put_oi'],
            'is_negative_gex':     neg_gex,
            'hist_vol_pct':        round(c['hv'] * 100, 1),
            'notes':               notes,
        }

    # Run options fetches concurrently — yfinance handles parallel calls well
    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = {pool.submit(_process_one, c): c['sym'] for c in candidates}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                if r:
                    results.append(r)
            except Exception as e:
                logger.debug("Gamma scan worker error: %s", e)

    results.sort(key=lambda x: x['gamma_squeeze_score'], reverse=True)
    for i, r in enumerate(results[:top_n], 1):
        r['rank'] = i

    logger.info(
        "Gamma squeeze scan: %d candidates → %d with options data, returning top %d",
        len(candidates), len(results), min(top_n, len(results))
    )
    return results[:top_n]


# ── FastAPI endpoint ──────────────────────────────────────────────────────────

@router.get("/scan")
async def scan_gamma_squeeze(
    top_n:       int   = Query(default=200, ge=1, le=2000),
    min_volume:  float = Query(default=50_000, ge=0),
    force:       bool  = Query(default=False),
):
    """
    Scan for real gamma squeeze setups at 52-week lows.
    Uses live yfinance options chains + Black-Scholes GEX calculation.
    Negative GEX at a low = market-maker short gamma → explosive reversal potential.
    """
    global _last_scan, _last_scan_time, _scan_in_progress

    now = time.time()
    if not force and _last_scan and (now - _last_scan_time) < _SCAN_TTL:
        return {
            "source":    "cache",
            "scan_age_s": round(now - _last_scan_time, 0),
            "count":     len(_last_scan),
            "results":   _last_scan[:top_n],
        }

    if _scan_in_progress:
        return {
            "source":  "scan_in_progress",
            "message": "Scan running — try again in 60 s",
            "results": _last_scan[:top_n],
        }

    _scan_in_progress = True
    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: _run_gamma_squeeze_scan(top_n, min_volume),
        )
        _last_scan      = results
        _last_scan_time = time.time()
        return {
            "source":  "fresh",
            "count":   len(results),
            "results": results,
        }
    except Exception as exc:
        logger.exception("Gamma squeeze scan failed: %s", exc)
        return {"source": "error", "message": str(exc), "results": []}
    finally:
        _scan_in_progress = False
