# trade_signal_engine/scanner.py
"""
Batch scanner — orchestrates Universe → Collector → Scorer → Recommender pipeline.

Scans multiple symbols and returns ranked trade recommendations.
"""

from __future__ import annotations

import logging
import time
from typing import Any

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
