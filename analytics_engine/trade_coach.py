"""
RoboAlgo — Trade Coach Engine
Explains signals, evaluates historical performance, generates risk warnings,
and reviews completed trades.

The system uses deterministic statistical analysis of existing engine outputs —
no black-box AI models are used.
"""

import logging
from datetime import date, datetime
from typing import Optional

from sqlalchemy import select, desc, func

from database.connection import get_session

logger = logging.getLogger(__name__)


# ── Data classes (plain dicts for JSON serialisation) ─────────────────────────

def _signal_explanation(symbol: str) -> dict:
    """
    Gather signal evidence for a given symbol and build a structured explanation.

    Returns:
        {
          symbol, market_state, setup_type, setup_quality_score,
          evidence: [str], risk_factors: [str],
          scenario_map: [{label, probability, price_target, direction}]
        }
    """
    evidence     = []
    risk_factors = []
    scenario_map = []

    try:
        from database.models import (
            Instrument, MarketState, Feature, ConfluenceSignal,
            BreakoutGate, RangeCompression, LiquidityShelfScore,
        )

        with get_session() as session:
            inst = session.execute(
                select(Instrument).where(Instrument.symbol == symbol.upper())
            ).scalar_one_or_none()
            if inst is None:
                return _empty_explanation(symbol, f"Symbol {symbol} not found in database")

            today = date.today()

            # Market state
            ms_row = session.execute(
                select(MarketState)
                .where(MarketState.instrument_id == inst.id, MarketState.date <= today)
                .order_by(desc(MarketState.date)).limit(1)
            ).scalar_one_or_none()
            market_state = ms_row.state if ms_row else "UNKNOWN"

            # Latest features
            feat = session.execute(
                select(Feature)
                .where(Feature.instrument_id == inst.id, Feature.date <= today)
                .order_by(desc(Feature.date)).limit(1)
            ).scalar_one_or_none()

            # Confluence signal
            conf = session.execute(
                select(ConfluenceSignal)
                .where(ConfluenceSignal.instrument_id == inst.id, ConfluenceSignal.date <= today)
                .order_by(desc(ConfluenceSignal.date)).limit(1)
            ).scalar_one_or_none()

            # Breakout gate
            bo = session.execute(
                select(BreakoutGate)
                .where(BreakoutGate.instrument_id == inst.id, BreakoutGate.date <= today)
                .order_by(desc(BreakoutGate.date)).limit(1)
            ).scalar_one_or_none()

            # Compression
            comp = session.execute(
                select(RangeCompression)
                .where(RangeCompression.instrument_id == inst.id, RangeCompression.date <= today)
                .order_by(desc(RangeCompression.date)).limit(1)
            ).scalar_one_or_none()

            # Liquidity shelf
            shelf = session.execute(
                select(LiquidityShelfScore)
                .where(LiquidityShelfScore.instrument_id == inst.id, LiquidityShelfScore.date <= today)
                .order_by(desc(LiquidityShelfScore.date)).limit(1)
            ).scalar_one_or_none()

        # ── Build evidence list ──────────────────────────────────────────────
        setup_quality_score = float(conf.setup_quality_score) if conf and conf.setup_quality_score else None
        breakout_quality    = float(bo.breakout_quality_score) if bo and bo.breakout_quality_score else None
        compression_score   = float(comp.compression_score)   if comp and comp.compression_score else None
        shelf_score         = float(shelf.liquidity_shelf_score) if shelf and shelf.liquidity_shelf_score else None

        if setup_quality_score is not None:
            label = "Excellent" if setup_quality_score >= 80 else "Good" if setup_quality_score >= 65 else "Fair" if setup_quality_score >= 50 else "Weak"
            evidence.append(f"Setup quality: {setup_quality_score:.0f}/100 ({label})")

        if breakout_quality is not None:
            label = "Strong" if breakout_quality >= 70 else "Moderate"
            evidence.append(f"Breakout quality gate: {breakout_quality:.0f}/100 ({label})")
            if bo and bo.gate_passed:
                evidence.append("Breakout gate: PASSED — momentum conditions met")
            elif bo:
                risk_factors.append("Breakout gate not yet triggered — wait for confirmation")

        if compression_score is not None:
            if compression_score >= 70:
                evidence.append(f"Volatility compression: {compression_score:.0f}/100 — coil is tight, breakout potential high")
            elif compression_score >= 40:
                evidence.append(f"Volatility compression: {compression_score:.0f}/100 — moderate compression")
            else:
                risk_factors.append(f"Volatility is not compressed ({compression_score:.0f}/100) — setup may be early")

        if shelf_score is not None:
            if shelf_score >= 60:
                evidence.append(f"Liquidity shelf: {shelf_score:.0f}/100 — institutional absorption detected at this level")
            elif shelf_score >= 35:
                evidence.append(f"Liquidity shelf: {shelf_score:.0f}/100 — moderate support absorption")
            else:
                risk_factors.append(f"Liquidity shelf score low ({shelf_score:.0f}/100) — limited institutional support")

        if feat:
            if feat.trend_strength is not None:
                ts = float(feat.trend_strength)
                if ts >= 70:
                    evidence.append(f"Trend strength: {ts:.0f}/100 — strong directional momentum")
                elif ts >= 40:
                    evidence.append(f"Trend strength: {ts:.0f}/100 — moderate momentum")
                else:
                    risk_factors.append(f"Trend is weak ({ts:.0f}/100) — choppy conditions possible")

            if feat.volatility_percentile is not None:
                vp = float(feat.volatility_percentile) * 100
                if vp > 75:
                    risk_factors.append(f"Volatility is elevated (top {100-vp:.0f}% historically) — widen stops or reduce size")
                elif vp < 30:
                    evidence.append(f"Volatility is suppressed ({vp:.0f}th percentile) — favourable for compression breakouts")

            if feat.volume_ratio is not None:
                vr = float(feat.volume_ratio)
                if vr > 1.5:
                    evidence.append(f"Volume surge: {vr:.1f}× average — institutional participation visible")
                elif vr > 1.0:
                    evidence.append(f"Volume above average: {vr:.1f}× — normal participation")
                else:
                    risk_factors.append(f"Volume is light ({vr:.1f}× average) — low conviction move possible")

        if market_state == "COMPRESSION":
            evidence.append("Market state: COMPRESSION — optimal window for breakout entry preparation")
        elif market_state == "TREND":
            evidence.append("Market state: TREND — pullback entries aligned with primary direction")
        elif market_state == "EXPANSION":
            evidence.append("Market state: EXPANSION — breakout already in progress, scale or trail stops")
        elif market_state == "CHAOS":
            risk_factors.append("Market state: CHAOS — high-volatility regime, reduce position size or stand aside")

        # ── Scenario map (probability-weighted outcomes) ─────────────────────
        if conf and conf.expected_move_pct:
            move  = float(conf.expected_move_pct)
            # Derive a "current price" approximation from conf entry price
            entry = float(conf.entry_price) if conf.entry_price else 0
            stop  = float(conf.stop_price)  if conf.stop_price  else 0

            if entry > 0:
                breakout_target = entry * (1 + move)
                pullback_target = entry * (1 - abs(move) * 0.4)
                failure_target  = stop if stop else entry * 0.94

                # Quality-adjusted probabilities
                q = (setup_quality_score or 50) / 100
                p_breakout = round(0.40 + 0.30 * q, 2)
                p_pullback = round(0.35 - 0.15 * q, 2)
                p_failure  = round(max(0.05, 1 - p_breakout - p_pullback), 2)

                scenario_map = [
                    {
                        "label":         "Breakout",
                        "probability":   p_breakout,
                        "price_target":  round(breakout_target, 2),
                        "direction":     "bullish",
                        "description":   f"Compression resolves upward, target +{move*100:.1f}%",
                    },
                    {
                        "label":         "Pullback / Consolidation",
                        "probability":   p_pullback,
                        "price_target":  round(pullback_target, 2),
                        "direction":     "neutral",
                        "description":   "Retest of support before resuming trend",
                    },
                    {
                        "label":         "Failure / Stop",
                        "probability":   p_failure,
                        "price_target":  round(failure_target, 2),
                        "direction":     "bearish",
                        "description":   "Setup fails, stop loss triggered",
                    },
                ]

        setup_type = conf.setup_type if conf and conf.setup_type else "—"

        return {
            "symbol":             symbol.upper(),
            "market_state":       market_state,
            "setup_type":         setup_type,
            "setup_quality_score":setup_quality_score,
            "evidence":           evidence,
            "risk_factors":       risk_factors,
            "scenario_map":       scenario_map,
            "computed_at":        datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        logger.warning("TradeCoach._signal_explanation(%s) failed: %s", symbol, e)
        return _empty_explanation(symbol, str(e))


def _empty_explanation(symbol: str, error: str) -> dict:
    return {
        "symbol":             symbol.upper(),
        "market_state":       "UNKNOWN",
        "setup_type":         "—",
        "setup_quality_score":None,
        "evidence":           [],
        "risk_factors":       [],
        "scenario_map":       [],
        "error":              error,
        "computed_at":        datetime.utcnow().isoformat() + "Z",
    }


def _find_similar_setups(symbol: str) -> dict:
    """
    Query historical trades with same setup_type and similar market_state.
    Return win rate, average return, drawdown, and sample size.
    """
    try:
        from database.models import TradeLifecycle, ConfluenceSignal, Instrument
        from sqlalchemy import and_

        with get_session() as session:
            inst = session.execute(
                select(Instrument).where(Instrument.symbol == symbol.upper())
            ).scalar_one_or_none()

            # Get the current setup type for this symbol
            today = date.today()
            conf  = session.execute(
                select(ConfluenceSignal)
                .where(
                    ConfluenceSignal.instrument_id == inst.id if inst else True,
                    ConfluenceSignal.date <= today,
                )
                .order_by(desc(ConfluenceSignal.date))
                .limit(1)
            ).scalar_one_or_none() if inst else None

            setup_type   = conf.setup_type   if conf and conf.setup_type   else None
            quality_score = float(conf.setup_quality_score) if conf and conf.setup_quality_score else 50.0

            # Query closed trades with similar setup
            query = select(TradeLifecycle).where(
                TradeLifecycle.state == "CLOSED",
                TradeLifecycle.exit_price != None,
            )
            if setup_type:
                query = query.where(TradeLifecycle.setup_type == setup_type)

            trades = session.execute(query.limit(200)).scalars().all()

        if not trades:
            return {
                "symbol":     symbol.upper(),
                "setup_type": setup_type or "unknown",
                "found":      False,
                "message":    "No historical closed trades found for this setup type",
                "computed_at":datetime.utcnow().isoformat() + "Z",
            }

        returns = []
        for t in trades:
            if t.entry_price and t.exit_price and t.entry_price > 0:
                r = (float(t.exit_price) - float(t.entry_price)) / float(t.entry_price)
                returns.append(r)

        if not returns:
            return {"symbol": symbol.upper(), "found": False, "message": "No return data on closed trades"}

        wins         = sum(1 for r in returns if r > 0)
        win_rate     = wins / len(returns)
        avg_return   = sum(returns) / len(returns)
        max_drawdown = min(returns) if returns else 0
        profit_factor = (
            sum(r for r in returns if r > 0) / abs(sum(r for r in returns if r < 0) or 1)
        )

        return {
            "symbol":        symbol.upper(),
            "setup_type":    setup_type or "all",
            "found":         True,
            "sample_size":   len(returns),
            "win_rate":      round(win_rate, 3),
            "avg_return":    round(avg_return, 4),
            "max_drawdown":  round(max_drawdown, 4),
            "profit_factor": round(profit_factor, 2),
            "computed_at":   datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        logger.warning("TradeCoach._find_similar_setups(%s) failed: %s", symbol, e)
        return {"symbol": symbol.upper(), "found": False, "error": str(e)}


def _review_completed_trade(trade_id: int) -> dict:
    """Review a completed trade comparing actual vs optimal entry/exit."""
    try:
        from database.models import TradeLifecycle

        with get_session() as session:
            trade = session.get(TradeLifecycle, trade_id)
            if trade is None:
                return {"trade_id": trade_id, "error": "Trade not found"}

            entry  = float(trade.entry_price)  if trade.entry_price  else None
            exit_p = float(trade.exit_price)   if trade.exit_price   else None
            stop   = float(trade.stop_price)   if trade.stop_price   else None
            t1     = float(trade.tier1_sell)   if trade.tier1_sell   else None
            t2     = float(trade.tier2_sell)   if trade.tier2_sell   else None

        if not entry or not exit_p:
            return {"trade_id": trade_id, "error": "Incomplete trade data"}

        actual_return = (exit_p - entry) / entry

        # Entry quality: compare vs optimal (ideal entry = ~entry, poor = far from stop)
        entry_quality_score = 75.0  # baseline — no optimal entry data without OHLC context

        # Exit quality: did we hit T1/T2?
        exit_quality_score = 50.0
        missed_profit      = 0.0

        if t1 and actual_return > 0:
            if exit_p >= t1:
                exit_quality_score = 85.0
                if t2 and exit_p < t2:
                    missed_profit = (t2 - exit_p) / entry
            else:
                exit_quality_score = 40.0
                missed_profit = (t1 - exit_p) / entry if exit_p < t1 else 0

        # Summary verdict
        if entry_quality_score >= 75 and exit_quality_score >= 75:
            verdict = "Well executed — entry and exit both within plan"
        elif entry_quality_score >= 75:
            verdict = "Good entry — consider holding longer to reach tier targets"
        elif exit_quality_score >= 75:
            verdict = "Good exit — but could improve entry timing"
        elif actual_return > 0:
            verdict = "Profitable but execution could be improved"
        else:
            verdict = "Loss — review stop placement and entry conditions"

        return {
            "trade_id":           trade_id,
            "symbol":             trade.symbol if hasattr(trade, 'symbol') else "—",
            "actual_return":      round(actual_return, 4),
            "entry_quality":      round(entry_quality_score, 0),
            "exit_quality":       round(exit_quality_score, 0),
            "missed_profit_pct":  round(missed_profit * 100, 2),
            "verdict":            verdict,
            "computed_at":        datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        logger.warning("TradeCoach._review_completed_trade(%d) failed: %s", trade_id, e)
        return {"trade_id": trade_id, "error": str(e)}


# ── Public API class ──────────────────────────────────────────────────────────

class TradeCoachEngine:
    """
    Provides signal explanations, similar setup analysis, and trade reviews.
    All methods are deterministic and use existing engine outputs.
    """

    def generate_signal_explanation(self, symbol: str) -> dict:
        """Return structured explanation for the current signal on `symbol`."""
        return _signal_explanation(symbol)

    def generate_risk_warnings(self, symbol: str) -> list[str]:
        """Return a list of risk warning strings for the current setup."""
        expl = _signal_explanation(symbol)
        return expl.get("risk_factors", [])

    def find_similar_setups(self, symbol: str) -> dict:
        """Find historical trades with the same setup type and return stats."""
        return _find_similar_setups(symbol)

    def review_completed_trade(self, trade_id: int) -> dict:
        """Review a completed trade and score entry/exit quality."""
        return _review_completed_trade(trade_id)
