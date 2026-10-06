# trade_signal_engine/scanner.py
"""
Batch scanner — orchestrates Universe → Collector → Scorer → Recommender pipeline.

Scans multiple symbols and returns ranked trade recommendations.
Supports streaming mode (yield progress per symbol) for SSE endpoints.
"""

from __future__ import annotations

import logging
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Generator

logger = logging.getLogger(__name__)

# ── MA Cluster Breakout Detection ────────────────────────────────────────────
# Checks if price is breaking above ALL major EMAs on 15m & 1h with fresh
# bullish stochastic.  Used as a client-side filter checkbox on the UI.

def _detect_ma_cluster_breakout(sd) -> dict:
    """
    Detect MA cluster breakout: price above EMA 4,9,21,50,100,200,300
    on EITHER 15m or 1h timeframe, with slow stochastic freshly bullish.

    Returns {"breakout": bool, "details": {...}}
    """
    import numpy as np
    from trade_signal_engine.scorer import _slow_stochastic_series, _detect_stoch_setup

    EMA_PERIODS = [4, 9, 21, 50, 100, 200, 300]
    result = {"breakout": False, "tf_results": {}, "stoch": {}}

    def _compute_ema(closes: np.ndarray, period: int) -> float:
        """EMA of the last value."""
        if len(closes) < period:
            return float('nan')
        alpha = 2.0 / (period + 1)
        ema = float(closes[0])
        for i in range(1, len(closes)):
            ema = alpha * float(closes[i]) + (1 - alpha) * ema
        return ema

    def _check_tf(df, tf_name: str) -> dict:
        """Check if price is above all EMAs on this timeframe."""
        if df is None or len(df) < 300:
            # Need enough bars to compute EMA 300
            if df is None or len(df) < 50:
                return {"pass": False, "reason": f"insufficient data ({len(df) if df is not None else 0} bars)"}
            # If < 300 bars, only check EMAs we can compute
            pass

        closes = df["close"].values.astype(float)
        current = float(closes[-1])
        if current <= 0:
            return {"pass": False, "reason": "no price"}

        above = {}
        below = {}
        for period in EMA_PERIODS:
            if len(closes) < period:
                continue  # skip EMAs we can't compute
            ema_val = _compute_ema(closes, period)
            if np.isnan(ema_val):
                continue
            if current > ema_val:
                above[f"EMA{period}"] = round(ema_val, 4)
            else:
                below[f"EMA{period}"] = round(ema_val, 4)

        # Must be above ALL computable EMAs (at least EMA 4,9,21,50 minimum)
        computable = len(above) + len(below)
        all_above = len(below) == 0 and computable >= 4
        return {
            "pass": all_above,
            "above": list(above.keys()),
            "below": list(below.keys()),
            "price": round(current, 2),
        }

    # Check 15m and 1h timeframes — EITHER passing is enough
    tf_pass = False
    for tf in ["15m", "1h"]:
        df = sd.bars.get(tf)
        tf_result = _check_tf(df, tf)
        result["tf_results"][tf] = tf_result
        if tf_result["pass"]:
            tf_pass = True

    if not tf_pass:
        return result

    # Check stochastic: MUST have a fresh bullish cross on the 1h chart
    # No loose fallbacks — only a real cross from oversold counts
    stoch_bullish = False
    for tf in ["1h", "15m"]:
        df = sd.bars.get(tf)
        if df is not None and len(df) >= 20:
            stoch = _detect_stoch_setup(df, lookback=5)
            result["stoch"][tf] = {
                "k": stoch["k"], "d": stoch["d"],
                "long_cross": stoch["long_cross"],
                "long_score": stoch["long_score"],
                "label": stoch["label"],
            }
            # Only accept a genuine fresh bullish cross (k crossed above d from oversold)
            if stoch["long_cross"] and stoch["long_score"] >= 0.6:
                stoch_bullish = True

    result["breakout"] = tf_pass and stoch_bullish
    return result


# ── Relative Strength vs SPY Detection ─────────────────────────────────────────

def _detect_rs_leader(sd, spy_bars) -> dict:
    """
    Compare a stock's 10-day and 20-day % change vs SPY's same periods.

    Args:
        sd: SymbolData object (uses sd.bars.get("1d"))
        spy_bars: DataFrame of SPY daily bars (must have "close" column)

    Returns:
        {"rs_leader": bool, "rs_10d": float, "rs_20d": float,
         "spy_10d": float, "spy_20d": float, "rs_score": float}
    """
    default = {"rs_leader": False, "rs_10d": 0.0, "rs_20d": 0.0,
               "spy_10d": 0.0, "spy_20d": 0.0, "rs_score": 0.0}

    # Validate SPY bars
    if spy_bars is None or len(spy_bars) < 21:
        return default

    # Get stock daily bars
    stock_df = sd.bars.get("1d")
    if stock_df is None or len(stock_df) < 21:
        return default

    try:
        # Extract close arrays
        stock_closes = stock_df["close"].values
        spy_closes = spy_bars["close"].values

        # Helper to safely get float from potentially multi-level value
        def _float(val):
            if hasattr(val, 'iloc'):
                return float(val.iloc[0])
            return float(val)

        # Stock % changes
        stock_cur = _float(stock_closes[-1])
        stock_10ago = _float(stock_closes[-11])
        stock_20ago = _float(stock_closes[-21])

        if stock_10ago <= 0 or stock_20ago <= 0 or stock_cur <= 0:
            return default

        stock_10d = (stock_cur - stock_10ago) / stock_10ago * 100
        stock_20d = (stock_cur - stock_20ago) / stock_20ago * 100

        # SPY % changes
        spy_cur = _float(spy_closes[-1])
        spy_10ago = _float(spy_closes[-11])
        spy_20ago = _float(spy_closes[-21])

        if spy_10ago <= 0 or spy_20ago <= 0 or spy_cur <= 0:
            return default

        spy_10d = (spy_cur - spy_10ago) / spy_10ago * 100
        spy_20d = (spy_cur - spy_20ago) / spy_20ago * 100

        # RS leader = outperformed SPY over BOTH periods
        rs_leader = (stock_10d > spy_10d) and (stock_20d > spy_20d)

        # RS score = average of relative outperformance across both periods
        rs_score = ((stock_10d - spy_10d) + (stock_20d - spy_20d)) / 2.0

        return {
            "rs_leader": rs_leader,
            "rs_10d": round(stock_10d, 2),
            "rs_20d": round(stock_20d, 2),
            "spy_10d": round(spy_10d, 2),
            "spy_20d": round(spy_20d, 2),
            "rs_score": round(rs_score, 2),
        }
    except (IndexError, TypeError, ValueError, KeyError):
        return default


# ── TTM Squeeze / Bollinger Compression Detection ─────────────────────────────

def _detect_squeeze(sd) -> dict:
    """
    Detect TTM Squeeze: Bollinger Bands compress inside Keltner Channels,
    signalling energy accumulation ready to expand.

    Returns {"squeeze": bool, "squeeze_on": bool, "squeeze_fired": bool,
             "bb_width": float, "bb_width_percentile": float, "momentum_rising": bool}
    """
    import numpy as np

    result = {
        "squeeze": False, "squeeze_on": False, "squeeze_fired": False,
        "bb_width": 0.0, "bb_width_percentile": 0.0, "momentum_rising": False,
    }

    df = sd.bars.get("1d")
    if df is None or len(df) < 30:
        return result

    closes = df["close"].values.astype(float)
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)

    n = len(closes)
    if n < 30:
        return result

    # ── 20-period Bollinger Bands (close +/- 2*std) ───────────────
    bb_period = 20
    bb_mult = 2.0
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    bb_mid = np.full(n, np.nan)

    for i in range(bb_period - 1, n):
        window = closes[i - bb_period + 1 : i + 1]
        mid = np.mean(window)
        std = np.std(window, ddof=1) if len(window) > 1 else 0.0
        bb_mid[i] = mid
        bb_upper[i] = mid + bb_mult * std
        bb_lower[i] = mid - bb_mult * std

    # ── 20-period EMA for Keltner mid-line ────────────────────────
    kc_period = 20
    kc_mult = 1.5
    atr_period = 14

    ema20 = np.full(n, np.nan)
    alpha = 2.0 / (kc_period + 1)
    ema20[kc_period - 1] = np.mean(closes[:kc_period])
    for i in range(kc_period, n):
        ema20[i] = alpha * closes[i] + (1 - alpha) * ema20[i - 1]

    # ── ATR(14) for Keltner width ─────────────────────────────────
    atr_arr = np.full(n, np.nan)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    if n >= atr_period:
        atr_arr[atr_period - 1] = np.mean(tr[:atr_period])
        for i in range(atr_period, n):
            atr_arr[i] = (atr_arr[i - 1] * (atr_period - 1) + tr[i]) / atr_period

    # ── Keltner Channels ──────────────────────────────────────────
    kc_upper = ema20 + kc_mult * atr_arr
    kc_lower = ema20 - kc_mult * atr_arr

    # ── Squeeze state per bar ─────────────────────────────────────
    min_idx = max(bb_period - 1, kc_period, atr_period)
    squeeze_on_arr = np.zeros(n, dtype=bool)
    for i in range(min_idx, n):
        if np.isnan(bb_upper[i]) or np.isnan(kc_upper[i]):
            continue
        squeeze_on_arr[i] = (bb_upper[i] < kc_upper[i]) and (bb_lower[i] > kc_lower[i])

    # Current state
    cur = n - 1
    squeeze_on_now = bool(squeeze_on_arr[cur])
    result["squeeze_on"] = squeeze_on_now

    # BB width & percentile
    bb_width_arr = bb_upper - bb_lower
    cur_bb_width = float(bb_width_arr[cur]) if not np.isnan(bb_width_arr[cur]) else 0.0
    result["bb_width"] = round(cur_bb_width, 4)

    lookback = min(120, n)
    valid_widths = bb_width_arr[cur - lookback + 1 : cur + 1]
    valid_widths = valid_widths[~np.isnan(valid_widths)]
    if len(valid_widths) > 1 and cur_bb_width > 0:
        pct = float(np.sum(valid_widths < cur_bb_width) / len(valid_widths) * 100)
        result["bb_width_percentile"] = round(pct, 1)

    # Momentum: close - EMA20
    momentum = np.full(n, np.nan)
    for i in range(min_idx, n):
        if not np.isnan(ema20[i]):
            momentum[i] = closes[i] - ema20[i]

    # Momentum rising for 3+ bars
    momentum_rising = False
    if cur >= 3 and not np.isnan(momentum[cur]):
        diffs = [momentum[cur - j] - momentum[cur - j - 1]
                 for j in range(3)
                 if cur - j - 1 >= 0
                 and not np.isnan(momentum[cur - j])
                 and not np.isnan(momentum[cur - j - 1])]
        momentum_rising = len(diffs) >= 3 and all(d > 0 for d in diffs)
    result["momentum_rising"] = momentum_rising

    # Squeeze FIRED: was on recently, now releasing or momentum breaking out
    was_on_recently = any(squeeze_on_arr[max(0, cur - 3) : cur])  # previous 3 bars

    squeeze_released = was_on_recently and not squeeze_on_now
    momentum_breakout = squeeze_on_now and momentum_rising

    squeeze_fired = squeeze_released or momentum_breakout
    result["squeeze_fired"] = squeeze_fired
    result["squeeze"] = squeeze_fired

    return result


# ── Volume Climax at Support Detection ────────────────────────────────────────

def _detect_volume_climax(sd) -> dict:
    """
    Detect institutional buying: massive volume spike at or near support.
    Checks for Wyckoff spring (price dip below support that closes back above).

    Returns {"volume_climax": bool, "vol_ratio": float, "near_support": bool,
             "spring": bool, "buying_candle": bool}
    """
    import numpy as np

    result = {
        "volume_climax": False, "vol_ratio": 0.0,
        "near_support": False, "spring": False, "buying_candle": False,
    }

    df = sd.bars.get("1d")
    if df is None or len(df) < 25:
        return result

    closes = df["close"].values.astype(float)
    opens = df["open"].values.astype(float)
    highs = df["high"].values.astype(float)
    lows_arr = df["low"].values.astype(float)

    if "volume" not in df.columns:
        return result
    volumes = df["volume"].values.astype(float)

    n = len(closes)
    if n < 25:
        return result

    # ── 20-day average volume (exclude last 3 bars to avoid self-reference) ──
    avg_vol_20 = float(np.mean(volumes[max(0, n - 23) : n - 3]))
    if avg_vol_20 <= 0:
        return result

    # ── 20-day low as support proxy (exclude last 3 bars) ─────────
    support_low = float(np.min(lows_arr[max(0, n - 20) : n - 3]))

    # ── Check last 3 bars for climax conditions ───────────────────
    best_vol_ratio = 0.0
    any_near_support = False
    any_spring = False
    any_buying = False

    for offset in range(3):
        idx = n - 1 - offset
        if idx < 0:
            continue

        c = closes[idx]
        o = opens[idx]
        h = highs[idx]
        lo = lows_arr[idx]
        v = volumes[idx]

        if c <= 0 or avg_vol_20 <= 0:
            continue

        vol_ratio = v / avg_vol_20

        # Volume spike: > 2x average
        if vol_ratio <= 2.0:
            continue

        if vol_ratio > best_vol_ratio:
            best_vol_ratio = vol_ratio

        # Near support: close within 3% of 20-day low
        if support_low > 0:
            pct_from_support = (c - support_low) / support_low
            if pct_from_support <= 0.03:
                any_near_support = True

        # Wyckoff spring: price dipped below support but closed back above
        if lo < support_low and c > support_low:
            any_spring = True

        # Buying candle: close > open (green) OR close in upper 60% of range
        bar_range = h - lo
        if bar_range > 0:
            close_position = (c - lo) / bar_range  # 0 = bottom, 1 = top
        else:
            close_position = 0.5

        if c > o or close_position >= 0.40:  # upper 60% means position >= 0.40
            any_buying = True

    result["vol_ratio"] = round(best_vol_ratio, 2)
    result["near_support"] = any_near_support
    result["spring"] = any_spring
    result["buying_candle"] = any_buying

    # Final signal: vol spike + (near support or spring) + buying candle
    result["volume_climax"] = (
        best_vol_ratio > 2.0
        and (any_near_support or any_spring)
        and any_buying
    )

    return result


# ── Pre-screen cache (5 min TTL) ─────────────────────────────────────────────
_PRESCREEN_CACHE: tuple[float, list[dict]] | None = None
_PRESCREEN_TTL = 300  # seconds

# Parallel workers for deep scan (tune to CPU count / API rate limits)
_SCAN_WORKERS = 12  # yfinance-based fast_mode allows more parallelism


def scan_universe(
    symbols: list[str],
    profile: str = "SWING",
    account_size: float = 10000.0,
    options_symbols: list[str] | None = None,
    max_workers: int = 4,
) -> dict[str, Any]:
    """
    Scan a list of symbols through the full pipeline.

    Args:
        symbols: List of ticker symbols to scan
        profile: Trade profile (SCALP/SWING/POSITION)
        account_size: Account size for position sizing
        options_symbols: Symbols to also fetch options data for
        max_workers: Thread pool size for data collection

    Returns:
        {
            "scan_time": float (seconds),
            "symbols_scanned": int,
            "recommendations": [...],  # sorted by score desc
            "top_trades": [...],       # score >= 70
            "watchlist": [...],        # score 55-69
            "no_trade": [...],         # NO_TRADE verdicts
            "errors": [...],           # failed symbols
        }
    """
    from trade_signal_engine.collector import collect_batch, SymbolData
    from trade_signal_engine.scorer import UnifiedScorer
    from trade_signal_engine.recommender import TradeRecommender

    start = time.time()
    scorer = UnifiedScorer()
    recommender = TradeRecommender()

    # Step 1: Collect data for all symbols
    logger.info("Collecting data for %d symbols (%s profile)...", len(symbols), profile)
    symbol_data_list = collect_batch(
        symbols, profile=profile,
        fetch_options_for=options_symbols or [],
        max_workers=max_workers,
    )

    # Step 2: Score + Recommend each
    recommendations = []
    errors = []

    for sd in symbol_data_list:
        if sd.error:
            errors.append({"symbol": sd.symbol, "error": sd.error})
            continue

        try:
            signal = scorer.score(sd)

            # Compute ATR from primary bars
            atr = 0.0
            for tf in ["5m", "1h", "4h", "1d"]:
                if tf in sd.bars and len(sd.bars[tf]) >= 14:
                    import numpy as np
                    h = sd.bars[tf]["high"].values
                    l = sd.bars[tf]["low"].values
                    c = sd.bars[tf]["close"].values
                    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
                    atr = float(np.mean(tr[-14:]))
                    break

            rec = recommender.recommend(
                signal, atr=atr,
                current_price=sd.current_price,
                account_size=account_size,
            )

            # MA Cluster Breakout
            try:
                _ma = _detect_ma_cluster_breakout(sd)
                _ma_bo = _ma.get("breakout", False)
            except Exception:
                _ma_bo = False

            # Squeeze detection
            try:
                _sq = _detect_squeeze(sd)
                _sq_firing = _sq.get("squeeze", False)
            except Exception:
                _sq_firing = False

            # Volume climax detection
            try:
                _vc = _detect_volume_climax(sd)
                _vc_flag = _vc.get("volume_climax", False)
            except Exception:
                _vc_flag = False

            recommendations.append({
                "symbol": rec.symbol,
                "profile": rec.profile,
                "verdict": rec.verdict,
                "direction": rec.direction,
                "score": rec.score,
                "confidence": rec.confidence,
                "entry": rec.entry,
                "stop": rec.stop,
                "target": rec.target,
                "risk_reward": rec.risk_reward,
                "level_source": rec.level_source,
                "position_size": rec.position_size,
                "position_value": rec.position_value,
                "risk_amount": rec.risk_amount,
                "projected_profit": rec.projected_profit,
                "projected_profit_pct": rec.projected_profit_pct,
                "entry_notes": rec.entry_notes,
                "rejection_reason": rec.rejection_reason,
                "component_summary": rec.component_summary,
                "bonuses": signal.bonuses if hasattr(signal, 'bonuses') else [],
                "components": signal.components,
                "ma_cluster_breakout": _ma_bo,
                "squeeze_firing": _sq_firing,
                "volume_climax": _vc_flag,
            })
        except Exception as e:
            logger.warning("Scoring failed for %s: %s", sd.symbol, e)
            errors.append({"symbol": sd.symbol, "error": str(e)})

    # Sort by score descending
    recommendations.sort(key=lambda r: r["score"], reverse=True)

    # Categorize
    top_trades = [r for r in recommendations if r["verdict"] in ("BUY", "SELL")]
    watchlist = [r for r in recommendations if r["verdict"] == "HOLD"]
    no_trade = [r for r in recommendations if r["verdict"] == "NO_TRADE"]

    elapsed = round(time.time() - start, 2)
    logger.info("Scan complete: %d symbols in %.1fs, %d trades, %d watchlist",
                len(symbols), elapsed, len(top_trades), len(watchlist))

    return {
        "scan_time": elapsed,
        "symbols_scanned": len(symbols),
        "profile": profile,
        "account_size": account_size,
        "recommendations": recommendations,
        "top_trades": top_trades,
        "watchlist": watchlist,
        "no_trade": no_trade,
        "errors": errors,
    }


def _detect_market_regime() -> dict[str, Any]:
    """
    Detect macro market regime from SPY to determine bull/bear bias.
    Returns {"regime": str, "bias": str, "hedges": list[str]}
    """
    result = {"regime": "UNKNOWN", "bias": "NEUTRAL", "hedges": [], "spy_trend": "NEUTRAL",
              "spy_price": 0.0, "ma20": 0.0, "ma50": 0.0}
    try:
        from trade_signal_engine.collector import _fetch_daily_bars
        spy = _fetch_daily_bars("SPY", bars=60)
        if spy is not None and len(spy) >= 20:
            import numpy as np
            closes = spy["close"].values
            ma20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else float(np.mean(closes))
            ma50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else float(np.mean(closes))
            price = float(closes[-1])
            result["spy_price"] = round(price, 2)
            result["ma20"] = round(ma20, 2)
            result["ma50"] = round(ma50, 2)

            # Trend direction
            if price > ma20 > ma50:
                result["regime"] = "BULL"
                result["bias"] = "LONG"
                result["spy_trend"] = "UP"
            elif price < ma20 < ma50:
                result["regime"] = "BEAR"
                result["bias"] = "SHORT"
                result["spy_trend"] = "DOWN"
                # Auto-include inverse/hedge ETFs
                result["hedges"] = [
                    "SQQQ", "SPXS", "SDOW", "TZA", "SOXS", "LABD",
                    "FAZ", "TECS", "DUST", "UVXY", "VIXY", "VXX",
                    "SH", "PSQ", "DOG", "RWM",  # 1x inverse
                    "TLT", "GLD", "SLV",          # safe havens
                ]
            elif price < ma50:
                result["regime"] = "CORRECTION"
                result["bias"] = "SHORT"
                result["spy_trend"] = "DOWN"
                result["hedges"] = [
                    "SQQQ", "SPXS", "UVXY", "VXX", "SH", "TLT", "GLD",
                ]
            else:
                result["regime"] = "RANGE"
                result["bias"] = "NEUTRAL"
                result["spy_trend"] = "NEUTRAL"

            # Check for recent volatility expansion (VIX proxy via ATR)
            highs = spy["high"].values
            lows = spy["low"].values
            tr = highs[-14:] - lows[-14:]
            atr_recent = float(np.mean(tr[-5:]))
            atr_avg = float(np.mean(tr))
            if atr_recent > atr_avg * 1.5:
                result["regime"] += "_VOLATILE"
                if "UVXY" not in result["hedges"]:
                    result["hedges"].extend(["UVXY", "VXX", "VIXY"])

            logger.info("Market regime: %s, bias: %s, SPY=%.2f MA20=%.2f MA50=%.2f",
                        result["regime"], result["bias"], price, ma20, ma50)
    except Exception as e:
        logger.warning("Market regime detection failed: %s", e)
    return result


def _prescreen_movers() -> list[dict]:
    """
    Quick pre-screen: batch download ALL exchange tickers to find unusual movers.
    Results are cached for 5 minutes — repeat scans are instant.
    Returns list of {"symbol": str, "change_pct": float, "volume_ratio": float, "flags": list}.
    """
    global _PRESCREEN_CACHE
    now = time.time()

    # Return cached results if fresh
    if _PRESCREEN_CACHE is not None:
        ts, cached = _PRESCREEN_CACHE
        if now - ts < _PRESCREEN_TTL:
            logger.info("Pre-screen cache hit — %d movers (%.0fs old)", len(cached), now - ts)
            return cached

    try:
        from trade_signal_engine.universe import get_full_universe
        candidates = get_full_universe(
            min_volume_ratio=1.2,
            min_change_pct=1.5,
            max_candidates=300,     # capped at 300 — enough signal without slowing deep scan
            include_all_tracked=False,
        )
        result = [
            {
                "symbol": c.symbol,
                "change_pct": c.change_pct,
                "volume_ratio": c.volume_ratio,
                "price": c.price,
                "flags": c.flags,
            }
            for c in candidates
            if c.flags
        ]
        _PRESCREEN_CACHE = (now, result)
        return result
    except Exception as e:
        logger.warning("Pre-screen failed: %s", e)
        return []


def _score_one(sym, profile, scorer, recommender, account_size, price_range=None):
    """Score a single symbol (legacy — fetches its own data). Returns (rec_dict, error_dict)."""
    from trade_signal_engine.collector import collect
    sd = collect(sym, profile=profile, fetch_options=False, fast_mode=True)
    if sd.error:
        return None, {"symbol": sym, "error": sd.error}
    return _score_one_from_data(sym, sd, profile, scorer, recommender, account_size, price_range)


def _score_one_from_data(sym, sd, profile, scorer, recommender, account_size, price_range=None, spy_daily=None):
    """Score a symbol using pre-fetched SymbolData. Returns (rec_dict, error_dict)."""
    import numpy as np
    import pandas as pd

    # ── Flatten multi-level columns from yfinance ─────────────
    for tf, df in list(sd.bars.items()):
        if df is not None and not df.empty and hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
            df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
            sd.bars[tf] = df

    # ── Establish RELIABLE baseline price ────────────────────────
    # Daily/weekly bars from the DB are trustworthy. Intraday from yfinance
    # may be garbage when rate-limited. Use DB daily close as the price anchor.
    reliable_price = 0.0
    for tf in ["1d", "1wk"]:
        df = sd.bars.get(tf)
        if df is not None and not df.empty and "close" in df.columns:
            try:
                val = df["close"].iloc[-1]
                p = float(val.iloc[0]) if hasattr(val, 'iloc') else float(val)
                if p > 0:
                    reliable_price = p
                    break
            except (TypeError, ValueError, IndexError):
                continue

    # If no daily/weekly data, try Finnhub quote as fallback
    if reliable_price == 0:
        try:
            from data_engine.market_data_service import _finnhub_quote
            q = _finnhub_quote(sym)
            if q and q["price"] > 0:
                reliable_price = q["price"]
        except Exception:
            pass

    # Override current_price with reliable price (collector may have used garbage intraday)
    if reliable_price > 0:
        sd.current_price = reliable_price

    # ── Validate & drop garbage intraday bars ────────────────────
    # Any TF whose last close differs >30% from reliable_price is corrupt
    if reliable_price > 0:
        for tf in ["1h", "15m", "5m", "4h"]:
            df = sd.bars.get(tf)
            if df is not None and not df.empty and "close" in df.columns:
                try:
                    val = df["close"].iloc[-1]
                    close_price = float(val.iloc[0]) if hasattr(val, 'iloc') else float(val)
                except (TypeError, ValueError, IndexError):
                    sd.bars.pop(tf, None)
                    continue
                if close_price > 0 and abs(close_price - reliable_price) / reliable_price > 0.30:
                    sd.bars.pop(tf, None)

    if sd.current_price <= 0:
        return None, {"symbol": sym, "error": "No reliable price data"}

    # ── Price range filter (from category selection) ─────────────
    if price_range and sd.current_price > 0:
        lo, hi = price_range
        if sd.current_price < lo or sd.current_price > hi:
            return None, {"symbol": sym, "error": f"Price ${sd.current_price:.2f} outside range ${lo:.0f}-${hi:.0f}"}

    # ── Validate symbol is real and tradeable ────────────────────
    has_valid_bars = False
    for tf in ["1d", "1h", "5m", "4h", "1wk"]:
        df = sd.bars.get(tf)
        if df is not None and not df.empty and isinstance(df.index, pd.DatetimeIndex):
            has_valid_bars = True
            if tf == "1d":
                last_bar_date = df.index[-1]
                try:
                    now = pd.Timestamp.now(tz=last_bar_date.tz if last_bar_date.tz else None)
                    days_stale = (now - last_bar_date).days
                except Exception:
                    days_stale = 999
                if days_stale > 45:
                    return None, {"symbol": sym, "error": f"Stale data — last bar {days_stale} days old (delisted?)"}
            if tf in ("1d", "1h") and "volume" in df.columns:
                try:
                    avg_vol = float(df["volume"].tail(10).mean())
                except (TypeError, ValueError):
                    avg_vol = 0
                if avg_vol < 500_000:
                    return None, {"symbol": sym, "error": f"Low liquidity ({avg_vol/1e6:.1f}M avg vol) — min 500K required"}
            break
    if not has_valid_bars:
        return None, {"symbol": sym, "error": "No valid bar data with timestamps"}

    signal = scorer.score(sd)

    atr = 0.0
    for tf in ["5m", "1h", "4h", "1d"]:
        if tf in sd.bars and len(sd.bars[tf]) >= 14:
            h = sd.bars[tf]["high"].values
            l = sd.bars[tf]["low"].values
            c = sd.bars[tf]["close"].values
            tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
            atr = float(np.mean(tr[-14:]))
            break

    rec = recommender.recommend(signal, atr=atr, current_price=sd.current_price, account_size=account_size)

    # Day's % change: last close vs previous close from daily bars
    _day_change_pct = 0.0
    _daily_df = sd.bars.get("1d")
    if _daily_df is not None and len(_daily_df) >= 2:
        try:
            _lc = _daily_df["close"].iloc[-1]
            _pc = _daily_df["close"].iloc[-2]
            _last_close = float(_lc.iloc[0]) if hasattr(_lc, 'iloc') else float(_lc)
            _prev_close = float(_pc.iloc[0]) if hasattr(_pc, 'iloc') else float(_pc)
            if _prev_close > 0 and _last_close > 0:
                _day_change_pct = round((_last_close - _prev_close) / _prev_close * 100, 2)
        except Exception:
            pass

    # ── MA Cluster Breakout detection ──────────────────────────────
    try:
        _ma_result = _detect_ma_cluster_breakout(sd)
        _ma_breakout = _ma_result.get("breakout", False)
    except Exception:
        _ma_breakout = False
        _ma_result = {"breakout": False}

    # ── Relative Strength vs SPY detection ─────────────────────────
    try:
        _rs_result = _detect_rs_leader(sd, spy_daily)
        _rs_leader = _rs_result.get("rs_leader", False)
    except Exception:
        _rs_leader = False
        _rs_result = {"rs_leader": False}

    # ── Squeeze detection ─────────────────────────────────────────
    try:
        _sq_result = _detect_squeeze(sd)
        _sq_firing = _sq_result.get("squeeze", False)
    except Exception:
        _sq_firing = False
        _sq_result = {"squeeze": False}

    # ── Volume climax detection ───────────────────────────────────
    try:
        _vc_result = _detect_volume_climax(sd)
        _vc_flag = _vc_result.get("volume_climax", False)
    except Exception:
        _vc_flag = False
        _vc_result = {"volume_climax": False}

    rec_dict = {
        "symbol": rec.symbol, "profile": rec.profile, "verdict": rec.verdict,
        "direction": rec.direction, "score": rec.score, "confidence": rec.confidence,
        "change_pct": _day_change_pct,
        "entry": rec.entry, "stop": rec.stop, "target": rec.target,
        "risk_reward": rec.risk_reward, "level_source": rec.level_source,
        "position_size": rec.position_size, "position_value": rec.position_value,
        "risk_amount": rec.risk_amount, "projected_profit": rec.projected_profit,
        "projected_profit_pct": rec.projected_profit_pct, "entry_notes": rec.entry_notes,
        "rejection_reason": rec.rejection_reason,
        "component_summary": rec.component_summary,
        "bonuses": signal.bonuses if hasattr(signal, 'bonuses') else [],
        "components": signal.components,
        "ma_cluster_breakout": _ma_breakout,
        "rs_leader": _rs_leader,
        "squeeze_firing": _sq_firing,
        "volume_climax": _vc_flag,
        "component_details": {
            k: v for k, v in (signal.component_details if hasattr(signal, 'component_details') else {}).items()
            if k in ("smc", "gex", "dealer", "catalyst", "levels", "exhaustion", "wave", "stoch_gate", "golden_death_cross", "ma_cluster_breakout")
        },
    }
    if _ma_breakout:
        rec_dict["component_details"]["ma_cluster_breakout"] = _ma_result
    if _rs_leader:
        rec_dict["component_details"]["rs_leader"] = _rs_result
    if _sq_firing:
        rec_dict["component_details"]["squeeze"] = _sq_result
    if _vc_flag:
        rec_dict["component_details"]["volume_climax"] = _vc_result
    return rec_dict, None


def scan_universe_streaming(
    symbols: list[str],
    profile: str = "SWING",
    account_size: float = 10000.0,
    discover_movers: bool = False,
    scan_label: str = "",
    price_range: tuple[float, float] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """
    Scan symbols one-by-one, yielding progress events for SSE streaming.

    If discover_movers=True, first pre-screens the FULL market (~6400 tickers)
    for unusual movers, merges them into the scan list, and detects market regime
    to auto-include hedges when bearish.

    Yields:
      {"type": "regime", ...market regime info...}
      {"type": "status", "message": str}     -- one or more during pre-screen (per-second heartbeats)
      {"type": "discovery", "movers_found": int, "hedges_added": int, "total_symbols": int}
      {"type": "progress", "completed": int, "total": int, "symbol": str, "verdict": str, "score": float}
      {"type": "trade", "trade": {...}}  -- emitted live for actionable symbols
      {"type": "result", ...final scan result...}
    """
    from trade_signal_engine.scorer import UnifiedScorer
    from trade_signal_engine.recommender import TradeRecommender

    start = time.time()
    scorer = UnifiedScorer()
    recommender = TradeRecommender()

    # ── Phase 0: Market regime detection ───────────────────────────
    regime = _detect_market_regime()
    yield {"type": "regime", **regime}

    # ── Phase 1: Discover hidden gems from full market ─────────────
    final_symbols = list(dict.fromkeys(symbols))  # deduplicate
    movers_found = 0
    hedges_added = 0

    if not discover_movers and scan_label:
        # Category-specific scan — show precise count, skip full-market sweep
        yield {"type": "status", "message": f"Scanning {scan_label}..."}

    if discover_movers:
        # Check if we'll use cache (instant) or do a fresh scan
        now = time.time()
        cached = _PRESCREEN_CACHE and (now - _PRESCREEN_CACHE[0]) < _PRESCREEN_TTL
        if cached:
            age = int(now - _PRESCREEN_CACHE[0])
            yield {"type": "status", "message": f"Using cached market scan ({age}s old)..."}
            movers = _prescreen_movers()
        else:
            # Run pre-screen in background thread; yield per-second heartbeats
            _ps_result: list = []
            _ps_done = threading.Event()

            def _run_prescreen() -> None:
                _ps_result.append(_prescreen_movers())
                _ps_done.set()

            _ps_thread = threading.Thread(target=_run_prescreen, daemon=True)
            _ps_thread.start()

            elapsed_secs = 0
            _MAX_PRESCREEN_WAIT = 120  # hard cap — never wait more than 2 minutes
            yield {"type": "status", "message": "Pre-screening 6,400+ stocks for unusual activity..."}
            while not _ps_done.is_set() and elapsed_secs < _MAX_PRESCREEN_WAIT:
                _ps_done.wait(timeout=1.0)
                elapsed_secs += 1
                if not _ps_done.is_set():
                    yield {"type": "status",
                           "message": f"Pre-screening 6,400+ stocks for unusual activity... ({elapsed_secs}s)"}

            if not _ps_done.is_set():
                logger.warning("Pre-screen timed out after %ds — proceeding with seed list only", elapsed_secs)
                yield {"type": "status", "message": f"Pre-screen timed out ({elapsed_secs}s) — scanning seed list..."}
                movers = []
            else:
                movers = _ps_result[0] if _ps_result else []

        movers_found = len(movers)

        # Add movers that aren't already in the list
        existing = set(s.upper() for s in final_symbols)
        for m in movers:
            sym = m["symbol"].upper()
            if sym not in existing:
                final_symbols.append(sym)
                existing.add(sym)

    # Add hedges if market is bearish (but only if they fit the price range)
    if regime["bias"] == "SHORT" and regime["hedges"] and not price_range:
        existing = set(s.upper() for s in final_symbols)
        for h in regime["hedges"]:
            if h not in existing:
                final_symbols.append(h)
                existing.add(h)
                hedges_added += 1

    yield {
        "type": "discovery",
        "movers_found": movers_found,
        "hedges_added": hedges_added,
        "total_symbols": len(final_symbols),
        "regime": regime["regime"],
        "bias": regime["bias"],
    }

    # ── Phase 2: BATCH DATA COLLECTION ──────────────────────────────
    # Download ALL symbols per timeframe in bulk (one yf.download call each).
    # ~10 seconds for 500 symbols vs 40+ minutes with serial approach.
    total = len(final_symbols)
    recommendations: list[dict] = []
    errors: list[dict] = []

    # Show cache freshness in status
    from trade_signal_engine.data_cache import get_cache_age
    cache_age = get_cache_age()
    if cache_age < 300:
        yield {"type": "status", "message": f"Loading {total} symbols from cache (fresh, {int(cache_age)}s old)..."}
    elif cache_age < 3600:
        yield {"type": "status", "message": f"Loading {total} symbols (cache {int(cache_age/60)}m old, refreshing in background)..."}
    else:
        yield {"type": "status", "message": f"Downloading data for {total} symbols (no cache or stale)..."}

    from trade_signal_engine.collector import batch_collect

    _use_cache_only = len(final_symbols) > 1000
    if _use_cache_only:
        yield {"type": "status", "message": f"Large scan ({total} symbols) — using cached data only for speed"}

    # Run batch_collect in a thread so we can yield progress events
    _collect_q: queue.Queue = queue.Queue()
    _collect_result: list = []
    _collect_done = threading.Event()

    def _progress_cb(msg: str):
        _collect_q.put(msg)

    def _run_collect():
        try:
            result = batch_collect(final_symbols, profile=profile, progress_callback=_progress_cb, cache_only=_use_cache_only)
            _collect_result.append(result)
        except Exception as e:
            _collect_result.append({"__error__": str(e)})
        finally:
            _collect_done.set()

    threading.Thread(target=_run_collect, daemon=True).start()

    # Yield progress messages while batch_collect runs
    while not _collect_done.is_set():
        try:
            msg = _collect_q.get(timeout=1.0)
            yield {"type": "status", "message": msg}
        except queue.Empty:
            pass  # heartbeat — keep connection alive

    # Drain any remaining messages
    while not _collect_q.empty():
        try:
            msg = _collect_q.get_nowait()
            yield {"type": "status", "message": msg}
        except queue.Empty:
            break

    if not _collect_result or isinstance(_collect_result[0], dict) and "__error__" in _collect_result[0]:
        err_msg = _collect_result[0].get("__error__", "Unknown") if _collect_result else "Collect timeout"
        yield {"type": "status", "message": f"Data collection failed: {err_msg}"}
        yield {"type": "result", "scan_time": round(time.time() - start, 2),
               "symbols_scanned": 0, "profile": profile, "account_size": account_size,
               "regime": regime, "recommendations": [], "top_trades": [],
               "watchlist": [], "no_trade": [], "errors": [{"symbol": "ALL", "error": err_msg}]}
        return

    batch_data = _collect_result[0]

    # ── Extract SPY daily bars for Relative Strength comparison ────
    _spy_sd = batch_data.get("SPY") or batch_data.get("spy")
    _spy_daily_df = None
    if _spy_sd is not None and hasattr(_spy_sd, 'bars'):
        _spy_daily_df = _spy_sd.bars.get("1d")

    yield {"type": "status", "message": f"Scoring {total} symbols..."}

    # ── Phase 3: Score from pre-fetched data (CPU-bound, fast) ────
    def _score_from_data(sym: str, sd) -> tuple:
        """Score a single symbol using pre-fetched SymbolData."""
        try:
            return _score_one_from_data(sym, sd, profile, scorer, recommender, account_size, price_range, spy_daily=_spy_daily_df)
        except Exception as e:
            logger.warning("Scoring failed for %s: %s", sym, e)
            return None, {"symbol": sym, "error": str(e)}

    # Score in parallel threads (no I/O — just CPU scoring)
    result_q: queue.Queue = queue.Queue()

    def _score_worker(sym: str) -> None:
        sd = batch_data.get(sym)
        if sd is None or sd.error:
            err_msg = sd.error if sd else f"No data for {sym}"
            result_q.put((sym, None, {"symbol": sym, "error": err_msg}))
            return
        rec_dict, err = _score_from_data(sym, sd)
        # Inline pattern detection using daily bars already in memory
        if rec_dict and not err:
            try:
                from pattern_engine.chart_patterns import detect_all as _detect_all_patterns
                _pat_df = None
                for _ptf in ("1d", "daily"):
                    if _ptf in sd.bars and len(sd.bars[_ptf]) >= 30:
                        _pat_df = sd.bars[_ptf]
                        break
                if _pat_df is not None:
                    # Ensure 'date' column exists (pattern engine expects it)
                    if "date" not in _pat_df.columns and _pat_df.index.name == "date":
                        _pat_df = _pat_df.reset_index()
                    elif "date" not in _pat_df.columns:
                        _pat_df = _pat_df.copy()
                        _pat_df["date"] = _pat_df.index
                    _all_pats = _detect_all_patterns(sym, _pat_df)
                    _n_bars = len(_pat_df)
                    _recency_cutoff = int(_n_bars * 0.75)  # rightmost point must be in last 25% of data
                    _sym_patterns = []
                    for p in _all_pats:
                        if p.get("status") not in ("FORMING", "READY", "BREAKOUT", "COMPLETED"):
                            continue
                        if (p.get("confidence") or 0) < 40:
                            continue
                        # Recency check: pattern must touch recent bars
                        _pts = p.get("points") or []
                        _max_idx = max((pt[0] if isinstance(pt, (list, tuple)) else 0 for pt in _pts), default=0)
                        if _max_idx < _recency_cutoff:
                            continue  # stale pattern, skip
                        _sym_patterns.append(p)
                    _sym_patterns.sort(key=lambda x: x.get("confidence", 0), reverse=True)
                else:
                    _sym_patterns = []
                    # Still try chart identity with shorter data
                    for _ptf2 in ("1d", "daily"):
                        if _ptf2 in sd.bars and len(sd.bars[_ptf2]) >= 5:
                            _pat_df = sd.bars[_ptf2]
                            break
                pattern_names = [p.get("pattern_name", "") for p in _sym_patterns[:5]]
                # Fallback: chart identity (candlestick, trend, volume features)
                if not pattern_names and _pat_df is not None:
                    try:
                        from pattern_engine.chart_identity import detect_chart_identity
                        pattern_names = detect_chart_identity(sym, _pat_df)
                    except Exception:
                        pattern_names = []
                pattern_categories = list(set(p.get("pattern_category", "") for p in _sym_patterns if p.get("pattern_category")))
                pattern_score = sum(min((p.get("confidence") or 0) / 100 * 10, 8) for p in _sym_patterns[:3])
                rec_dict["patterns"] = pattern_names
                rec_dict["pattern_categories"] = pattern_categories
                rec_dict["pattern_score"] = round(pattern_score, 1)
                rec_dict["pattern_count"] = len(_sym_patterns)
                if pattern_score > 0:
                    rec_dict["score"] = round(rec_dict.get("score", 0) + pattern_score, 1)
            except Exception:
                rec_dict["patterns"] = []
                rec_dict["pattern_categories"] = []
                rec_dict["pattern_score"] = 0
                rec_dict["pattern_count"] = 0
        result_q.put((sym, rec_dict, err))

    executor = ThreadPoolExecutor(max_workers=_SCAN_WORKERS)
    futures = [executor.submit(_score_worker, sym) for sym in final_symbols]

    completed_count = 0
    while completed_count < total:
        try:
            sym, rec_dict, err = result_q.get(timeout=60)
        except queue.Empty:
            break

        completed_count += 1

        if err:
            errors.append(err)
            yield {"type": "progress", "completed": completed_count, "total": total,
                   "symbol": sym, "verdict": "ERROR", "score": 0}
            continue

        recommendations.append(rec_dict)
        yield {"type": "progress", "completed": completed_count, "total": total,
               "symbol": sym, "verdict": rec_dict["verdict"], "score": rec_dict["score"]}

        if rec_dict["verdict"] in ("BUY", "SELL", "HOLD"):
            yield {"type": "trade", "trade": rec_dict}

    executor.shutdown(wait=False)

    # Final result
    recommendations.sort(key=lambda r: r["score"], reverse=True)
    top_trades = [r for r in recommendations if r["verdict"] in ("BUY", "SELL")]
    watchlist = [r for r in recommendations if r["verdict"] == "HOLD"]
    no_trade = [r for r in recommendations if r["verdict"] == "NO_TRADE"]
    elapsed = round(time.time() - start, 2)

    yield {
        "type": "result",
        "scan_time": elapsed,
        "symbols_scanned": total,
        "profile": profile,
        "account_size": account_size,
        "regime": regime,
        "recommendations": recommendations,
        "top_trades": top_trades,
        "watchlist": watchlist,
        "no_trade": no_trade,
        "errors": errors,
    }


def scan_symbol(
    symbol: str,
    profile: str = "SWING",
    account_size: float = 10000.0,
    fetch_options: bool = True,
) -> dict[str, Any]:
    """
    Deep scan a single symbol — returns full recommendation with component breakdown.
    Used for the "symbol lookup" feature.
    """
    from trade_signal_engine.collector import collect
    from trade_signal_engine.scorer import UnifiedScorer
    from trade_signal_engine.recommender import TradeRecommender

    sd = collect(symbol, profile=profile, fetch_options=fetch_options)
    if sd.error:
        return {"symbol": symbol, "error": sd.error}

    scorer = UnifiedScorer()
    signal = scorer.score(sd)

    import numpy as np
    atr = 0.0
    for tf in ["5m", "1h", "4h", "1d"]:
        if tf in sd.bars and len(sd.bars[tf]) >= 14:
            h = sd.bars[tf]["high"].values
            l = sd.bars[tf]["low"].values
            c = sd.bars[tf]["close"].values
            tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
            atr = float(np.mean(tr[-14:]))
            break

    # Day's % change: last close vs previous close from daily bars
    _day_change_pct = 0.0
    _daily_df = sd.bars.get("1d")
    if _daily_df is not None and len(_daily_df) >= 2:
        _last_close = float(_daily_df["close"].iloc[-1])
        _prev_close = float(_daily_df["close"].iloc[-2])
        if _prev_close > 0 and _last_close > 0:
            _day_change_pct = round((_last_close - _prev_close) / _prev_close * 100, 2)

    recommender = TradeRecommender()
    rec = recommender.recommend(signal, atr=atr, current_price=sd.current_price, account_size=account_size)

    # MA Cluster Breakout
    try:
        _ma_result = _detect_ma_cluster_breakout(sd)
        _ma_breakout = _ma_result.get("breakout", False)
    except Exception:
        _ma_breakout = False
        _ma_result = {"breakout": False}

    # Relative Strength vs SPY
    _rs_leader = False
    _rs_result = {"rs_leader": False}
    try:
        from trade_signal_engine.collector import _fetch_daily_bars
        _spy_daily = _fetch_daily_bars("SPY", bars=60)
        _rs_result = _detect_rs_leader(sd, _spy_daily)
        _rs_leader = _rs_result.get("rs_leader", False)
    except Exception:
        pass

    # Squeeze detection
    try:
        _sq_result = _detect_squeeze(sd)
        _sq_firing = _sq_result.get("squeeze", False)
    except Exception:
        _sq_firing = False
        _sq_result = {"squeeze": False}

    # Volume climax detection
    try:
        _vc_result = _detect_volume_climax(sd)
        _vc_flag = _vc_result.get("volume_climax", False)
    except Exception:
        _vc_flag = False
        _vc_result = {"volume_climax": False}

    _details = {
        k: v for k, v in signal.component_details.items()
        if k in ("smc", "gex", "dealer", "catalyst", "levels", "exhaustion", "wave", "stoch_gate", "golden_death_cross", "ma_cluster_breakout")
    }
    if _ma_breakout:
        _details["ma_cluster_breakout"] = _ma_result
    if _rs_leader:
        _details["rs_leader"] = _rs_result
    if _sq_firing:
        _details["squeeze"] = _sq_result
    if _vc_flag:
        _details["volume_climax"] = _vc_result

    return {
        "symbol": rec.symbol,
        "profile": rec.profile,
        "current_price": sd.current_price,
        "change_pct": _day_change_pct,
        "verdict": rec.verdict,
        "direction": rec.direction,
        "score": rec.score,
        "confidence": rec.confidence,
        "entry": rec.entry,
        "stop": rec.stop,
        "target": rec.target,
        "risk_reward": rec.risk_reward,
        "level_source": rec.level_source,
        "position_size": rec.position_size,
        "position_value": rec.position_value,
        "risk_amount": rec.risk_amount,
        "projected_profit": rec.projected_profit,
        "projected_profit_pct": rec.projected_profit_pct,
        "entry_notes": rec.entry_notes,
        "rejection_reason": rec.rejection_reason,
        "component_summary": rec.component_summary,
        "bonuses": signal.bonuses if hasattr(signal, 'bonuses') else [],
        "components": signal.components,
        "ma_cluster_breakout": _ma_breakout,
        "rs_leader": _rs_leader,
        "squeeze_firing": _sq_firing,
        "volume_climax": _vc_flag,
        "component_details": _details,
    }


def scan_symbol_multi(
    symbol: str,
    account_size: float = 10000.0,
    fetch_options: bool = True,
) -> dict[str, Any]:
    """
    Scan a single symbol across ALL 3 profiles (SCALP, SWING, POSITION).
    Returns each profile's trade opportunity + cross-horizon warnings.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    profiles = ["SCALP", "SWING", "POSITION"]
    results: dict[str, dict] = {}

    # Run all 3 profiles in parallel — data comes from cache so it's fast
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(scan_symbol, symbol, p, account_size, fetch_options): p
            for p in profiles
        }
        for fut in as_completed(futures):
            profile = futures[fut]
            try:
                results[profile] = fut.result()
            except Exception as e:
                results[profile] = {"symbol": symbol, "profile": profile, "error": str(e)}

    # ── Determine overall trend from highest TF (POSITION daily/weekly) ──
    pos = results.get("POSITION", {})
    swing = results.get("SWING", {})
    scalp = results.get("SCALP", {})

    # Extract stoch gate direction from each profile
    def _get_direction(r: dict) -> str:
        gate = r.get("component_details", {}).get("stoch_gate", {})
        d = gate.get("regime_direction", r.get("direction", "NEUTRAL"))
        # Normalize NO_TRADE → NEUTRAL for comparison purposes
        return "NEUTRAL" if d in ("NO_TRADE", "NONE", "", None) else d

    pos_dir = _get_direction(pos)
    swing_dir = _get_direction(swing)
    scalp_dir = _get_direction(scalp)

    # ── Generate cross-horizon warnings ──────────────────────────────────
    def _add_warnings(profile_result: dict, prof_name: str) -> list[str]:
        warnings = []
        d = _get_direction(profile_result)

        if prof_name == "SCALP":
            # Scalp against higher TFs
            if d and pos_dir and d != pos_dir and pos_dir != "NEUTRAL" and d != "NEUTRAL":
                warnings.append(f"⚠ Counter-trend scalp: POSITION trend is {pos_dir}")
                warnings.append("Quick in-and-out only — don't hold")
            if d and swing_dir and d != swing_dir and swing_dir != "NEUTRAL" and d != "NEUTRAL":
                warnings.append(f"⚠ Against SWING direction ({swing_dir}) — tight stops")
            if pos_dir == d and swing_dir == d and d != "NEUTRAL":
                warnings.append(f"✦ All timeframes aligned {d} — high conviction scalp")

        elif prof_name == "SWING":
            if d and pos_dir and d != pos_dir and pos_dir != "NEUTRAL" and d != "NEUTRAL":
                warnings.append(f"⚠ Counter-trend swing: POSITION trend is {pos_dir}")
                warnings.append("Take profits early — don't overstay")
            elif d and pos_dir == d and d != "NEUTRAL":
                warnings.append(f"✦ POSITION trend confirms {d} — can hold longer")
            if d and scalp_dir and scalp_dir == d and d != "NEUTRAL":
                warnings.append(f"✦ SCALP momentum also {d} — strong near-term flow")

        elif prof_name == "POSITION":
            if d and swing_dir and swing_dir != d and swing_dir != "NEUTRAL" and d != "NEUTRAL":
                warnings.append(f"⚠ SWING showing {swing_dir} — wait for pullback entry")
            if d and scalp_dir and scalp_dir != d and scalp_dir != "NEUTRAL" and d != "NEUTRAL":
                warnings.append(f"⚠ Short-term flow is {scalp_dir} — be patient on entry")
            if swing_dir == d and scalp_dir == d and d != "NEUTRAL":
                warnings.append(f"✦ All horizons aligned {d} — ideal entry window")

        return warnings

    for prof_name in profiles:
        r = results.get(prof_name, {})
        if "error" not in r:
            r["warnings"] = _add_warnings(r, prof_name)

    # ── Determine if any profile is worth trading ────────────────────────
    tradeable = [p for p in profiles if results.get(p, {}).get("verdict") in ("BUY", "SELL")]
    best_profile = max(tradeable, key=lambda p: results[p].get("score", 0)) if tradeable else None

    # Current price from any successful result
    current_price = 0.0
    for p in profiles:
        cp = results.get(p, {}).get("current_price", 0)
        if cp > 0:
            current_price = cp
            break

    return {
        "symbol": symbol.upper(),
        "current_price": current_price,
        "overall_trend": pos_dir,
        "best_profile": best_profile,
        "profiles": {p: results.get(p, {}) for p in profiles},
    }
