# trade_signal_engine/scanner.py
"""
Batch scanner — orchestrates Universe → Collector → Scorer → Recommender pipeline.

Scans multiple symbols and returns ranked trade recommendations.
Supports streaming mode (yield progress per symbol) for SSE endpoints.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Generator

logger = logging.getLogger(__name__)


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
    Detect macro market regime from SPY + crypto cross-asset signals.

    SPY MA20/MA50 remains the primary vote (it drives the equity universe).
    Crypto signals — BTC/ETH trend, BTC-vs-SPY correlation break, BTC perp
    funding, USDT/USDC supply flow — are treated as *modifiers* that can:
      - Confirm a SPY signal → high-conviction hedges when bearish
      - Downgrade a SPY BULL to NEUTRAL when crypto diverges or positioning
        is stretched (equity moves after risk sentiment turns)
      - Add crypto-specific inverse hedges (BITI) when bearish is confirmed

    Returns:
      {
        "regime":    "BULL"|"BEAR"|"CORRECTION"|"RANGE"|"UNKNOWN" (+ optional "_VOLATILE"),
        "bias":      "LONG"|"SHORT"|"NEUTRAL",
        "hedges":    list[str] — inverse ETFs / safe havens to add when bearish,
        "spy_trend": "UP"|"DOWN"|"NEUTRAL",
        "crypto":    dict — full crypto risk read (see crypto_fetcher.compute_crypto_risk_read),
        "modifiers": list[str] — human-readable notes about crypto adjustments applied,
      }
    """
    result: dict[str, Any] = {
        "regime": "UNKNOWN",
        "bias": "NEUTRAL",
        "hedges": [],
        "spy_trend": "NEUTRAL",
        "crypto": {},
        "modifiers": [],
    }
    try:
        from trade_signal_engine.collector import _fetch_daily_bars
        spy = _fetch_daily_bars("SPY", bars=60)
        if spy is None or len(spy) < 20:
            return result

        import numpy as np
        closes = spy["close"].values
        ma20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else float(np.mean(closes))
        ma50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else float(np.mean(closes))
        price = float(closes[-1])

        # ── SPY-only base decision (unchanged behavior)
        if price > ma20 > ma50:
            result["regime"] = "BULL"
            result["bias"] = "LONG"
            result["spy_trend"] = "UP"
        elif price < ma20 < ma50:
            result["regime"] = "BEAR"
            result["bias"] = "SHORT"
            result["spy_trend"] = "DOWN"
            result["hedges"] = [
                "SQQQ", "SPXS", "SDOW", "TZA", "SOXS", "LABD",
                "FAZ", "TECS", "DUST", "UVXY", "VIXY", "VXX",
                "SH", "PSQ", "DOG", "RWM",
                "TLT", "GLD", "SLV",
            ]
        elif price < ma50:
            result["regime"] = "CORRECTION"
            result["bias"] = "SHORT"
            result["spy_trend"] = "DOWN"
            result["hedges"] = ["SQQQ", "SPXS", "UVXY", "VXX", "SH", "TLT", "GLD"]
        else:
            result["regime"] = "RANGE"
            result["bias"] = "NEUTRAL"
            result["spy_trend"] = "NEUTRAL"

        # ── Volatility expansion (unchanged)
        highs = spy["high"].values
        lows = spy["low"].values
        tr = highs[-14:] - lows[-14:]
        atr_recent = float(np.mean(tr[-5:]))
        atr_avg = float(np.mean(tr))
        if atr_recent > atr_avg * 1.5:
            result["regime"] += "_VOLATILE"
            if "UVXY" not in result["hedges"]:
                result["hedges"].extend(["UVXY", "VXX", "VIXY"])

        # ── Crypto cross-asset overlay (new)
        try:
            from data_engine.crypto_fetcher import compute_crypto_risk_read
            crypto = compute_crypto_risk_read(spy_closes=np.asarray(closes, dtype=float))
            result["crypto"] = crypto

            risk = crypto.get("risk_signal", "NEUTRAL")
            btc_trend = crypto.get("btc_trend", "NEUTRAL")

            # Rule 1: SPY BULL + crypto DIVERGE/STRETCHED → downgrade to NEUTRAL
            # Rationale: positioning stretched or risk sentiment turning ahead
            # of equity price action. Don't add hedges, but stop taking longs.
            if result["bias"] == "LONG" and risk in ("DIVERGE", "STRETCHED"):
                result["bias"] = "NEUTRAL"
                result["modifiers"].append(f"downgraded LONG→NEUTRAL: crypto {risk.lower()}")

            # Rule 2: SPY BEAR + crypto CONFIRM_SHORT → add crypto-hedges
            if result["bias"] == "SHORT" and risk == "CONFIRM_SHORT":
                for h in ("BITI", "BITS"):  # inverse-BTC ETFs
                    if h not in result["hedges"]:
                        result["hedges"].append(h)
                result["modifiers"].append("high-conviction bear: added crypto inverse ETFs")

            # Rule 3: SPY RANGE + crypto CONFIRM_SHORT → nudge to CORRECTION-lite
            # Crypto often leads sentiment; if equities are drifting but crypto
            # is confirming shorts, prep some hedges.
            if result["bias"] == "NEUTRAL" and risk == "CONFIRM_SHORT":
                result["modifiers"].append("crypto confirming short — adding light hedges")
                for h in ("UVXY", "SH", "GLD"):
                    if h not in result["hedges"]:
                        result["hedges"].append(h)

            # Rule 4: correlation break — always worth flagging
            if crypto.get("correlation_broken"):
                result["modifiers"].append("BTC-SPY correlation broken (regime shift risk)")

            logger.info(
                "Market regime: %s, bias: %s, SPY=%.2f MA20=%.2f MA50=%.2f | "
                "BTC=%s ETH=%s funding=%s stables=%s risk=%s%s",
                result["regime"], result["bias"], price, ma20, ma50,
                btc_trend, crypto.get("eth_trend"),
                crypto.get("funding_state"), crypto.get("stablecoin_flow"),
                risk,
                f" [{'; '.join(result['modifiers'])}]" if result["modifiers"] else "",
            )
        except Exception as ce:
            # Crypto layer is optional — never break the base regime signal
            logger.warning("Crypto regime overlay failed (using SPY-only): %s", ce)
            logger.info("Market regime: %s, bias: %s, SPY=%.2f MA20=%.2f MA50=%.2f",
                        result["regime"], result["bias"], price, ma20, ma50)

    except Exception as e:
        logger.warning("Market regime detection failed: %s", e)
    return result


def _prescreen_movers() -> list[dict]:
    """
    Quick pre-screen: batch download ALL exchange tickers to find unusual movers.
    Returns list of {"symbol": str, "change_pct": float, "volume_ratio": float, "flags": list}.
    """
    try:
        from trade_signal_engine.universe import get_full_universe
        candidates = get_full_universe(
            min_volume_ratio=1.5,
            min_change_pct=2.0,
            max_candidates=200,
            include_all_tracked=False,
        )
        return [
            {
                "symbol": c.symbol,
                "change_pct": c.change_pct,
                "volume_ratio": c.volume_ratio,
                "price": c.price,
                "flags": c.flags,
            }
            for c in candidates
            if c.flags  # only stocks with unusual activity
        ]
    except Exception as e:
        logger.warning("Pre-screen failed: %s", e)
        return []


def _score_one(sym, profile, scorer, recommender, account_size):
    """Score a single symbol. Returns (rec_dict, error_dict) or raises."""
    import numpy as np
    from trade_signal_engine.collector import collect

    sd = collect(sym, profile=profile, fetch_options=False)
    if sd.error:
        return None, {"symbol": sym, "error": sd.error}

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

    rec_dict = {
        "symbol": rec.symbol, "profile": rec.profile, "verdict": rec.verdict,
        "direction": rec.direction, "score": rec.score, "confidence": rec.confidence,
        "entry": rec.entry, "stop": rec.stop, "target": rec.target,
        "risk_reward": rec.risk_reward, "level_source": rec.level_source,
        "position_size": rec.position_size, "position_value": rec.position_value,
        "risk_amount": rec.risk_amount, "projected_profit": rec.projected_profit,
        "projected_profit_pct": rec.projected_profit_pct, "entry_notes": rec.entry_notes,
        "rejection_reason": rec.rejection_reason,
        "component_summary": rec.component_summary,
        "bonuses": signal.bonuses if hasattr(signal, 'bonuses') else [],
        "components": signal.components,
    }
    return rec_dict, None


def scan_universe_streaming(
    symbols: list[str],
    profile: str = "SWING",
    account_size: float = 10000.0,
    discover_movers: bool = False,
) -> Generator[dict[str, Any], None, None]:
    """
    Scan symbols one-by-one, yielding progress events for SSE streaming.

    If discover_movers=True, first pre-screens the FULL market (~6400 tickers)
    for unusual movers, merges them into the scan list, and detects market regime
    to auto-include hedges when bearish.

    Yields:
      {"type": "regime", ...market regime info...}
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

    if discover_movers:
        yield {"type": "status", "message": "Pre-screening 6,400+ stocks for unusual activity..."}
        movers = _prescreen_movers()
        movers_found = len(movers)

        # Add movers that aren't already in the list
        existing = set(s.upper() for s in final_symbols)
        for m in movers:
            sym = m["symbol"].upper()
            if sym not in existing:
                final_symbols.append(sym)
                existing.add(sym)

    # Add hedges if market is bearish
    if regime["bias"] == "SHORT" and regime["hedges"]:
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

    # ── Phase 2: Deep scan each symbol ─────────────────────────────
    total = len(final_symbols)
    recommendations: list[dict] = []
    errors: list[dict] = []

    for i, sym in enumerate(final_symbols):
        try:
            rec_dict, err = _score_one(sym, profile, scorer, recommender, account_size)
            if err:
                errors.append(err)
                yield {"type": "progress", "completed": i + 1, "total": total,
                       "symbol": sym, "verdict": "ERROR", "score": 0}
                continue

            recommendations.append(rec_dict)

            # Emit progress
            yield {"type": "progress", "completed": i + 1, "total": total,
                   "symbol": sym, "verdict": rec_dict["verdict"], "score": rec_dict["score"]}

            # Emit trade immediately if actionable (BUY/SELL/HOLD)
            if rec_dict["verdict"] in ("BUY", "SELL", "HOLD"):
                yield {"type": "trade", "trade": rec_dict}

        except Exception as e:
            logger.warning("Scoring failed for %s: %s", sym, e)
            errors.append({"symbol": sym, "error": str(e)})
            yield {"type": "progress", "completed": i + 1, "total": total,
                   "symbol": sym, "verdict": "ERROR", "score": 0}

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

    recommender = TradeRecommender()
    rec = recommender.recommend(signal, atr=atr, current_price=sd.current_price, account_size=account_size)

    return {
        "symbol": rec.symbol,
        "profile": rec.profile,
        "current_price": sd.current_price,
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
        "component_details": {
            k: v for k, v in signal.component_details.items()
            if k in ("smc", "gex", "dealer", "catalyst", "levels", "exhaustion", "wave")
        },
    }
