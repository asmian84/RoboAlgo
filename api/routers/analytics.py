"""RoboAlgo — Analytics API Router
Endpoints for expectancy, setup performance, and trade history.
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/expectancy")
def get_expectancy():
    """
    Return expected value (EV) for all setup types.
    EV = (win_rate × avg_win) − (loss_rate × avg_loss) per (setup_type, market_state).
    Requires at least 10 trades to be considered reliable.
    """
    from analytics_engine.expectancy import ExpectancyEngine
    engine = ExpectancyEngine()
    data = engine.get_all_ev()
    return {"expectancy": data}


@router.get("/expectancy/{setup_type}")
def get_setup_expectancy(setup_type: str, market_state: Optional[str] = None):
    """Return EV for a specific setup type, optionally filtered by market state."""
    from analytics_engine.expectancy import ExpectancyEngine
    engine = ExpectancyEngine()
    ev = engine.get_setup_ev(setup_type=setup_type, market_state=market_state)
    if not ev:
        raise HTTPException(
            status_code=404,
            detail=f"No performance data for setup_type={setup_type}"
        )
    return ev


@router.get("/setup-performance")
def get_setup_performance(
    market_state: Optional[str] = Query(None, description="Filter by state: COMPRESSION/TREND/EXPANSION/CHAOS"),
    min_trades:   int           = Query(5, description="Minimum trade count to include"),
):
    """
    Return historical performance by (setup_type, market_state) combination.
    Sorted by expected_value descending.
    """
    from analytics_engine.expectancy import ExpectancyEngine
    engine = ExpectancyEngine()
    rows   = engine.get_all_ev(market_state=market_state, min_trades=min_trades)
    return {"setup_performance": rows, "count": len(rows)}


@router.get("/trade-history")
def get_trade_history(
    symbol:      Optional[str] = Query(None),
    limit:       int           = Query(100, le=500),
    state:       Optional[str] = Query(None, description="Filter by lifecycle state"),
):
    """
    Return trade history from TradeLifecycle table.
    Defaults to EXIT (completed) trades.
    """
    from trade_engine.lifecycle import TradeLifecycleEngine
    engine = TradeLifecycleEngine()

    if state and state.upper() != "EXIT":
        trades = engine.get_active_trades(symbol=symbol)
    else:
        trades = engine.get_trade_history(symbol=symbol, limit=limit)

    return {
        "trades": trades,
        "count":  len(trades),
    }


@router.get("/trade-history/{trade_id}")
def get_single_trade(trade_id: int):
    """Return details for a specific trade lifecycle record."""
    from trade_engine.lifecycle import TradeLifecycleEngine
    engine = TradeLifecycleEngine()
    trade  = engine.get_trade(trade_id)
    if not trade:
        raise HTTPException(status_code=404, detail=f"Trade id={trade_id} not found")
    return trade


@router.get("/open-positions")
def get_open_positions():
    """Return all active (open) positions with their lifecycle state."""
    from trade_engine.lifecycle import TradeLifecycleEngine
    engine = TradeLifecycleEngine()
    trades = engine.get_active_trades()
    return {
        "open_positions": trades,
        "count":          len(trades),
    }


@router.get("/regime-performance")
def get_regime_performance():
    """
    Return strategy performance broken down by (market_state, setup_type).
    Sourced from regime_strategy_performance table.
    """
    from strategy_engine.regime_adaptive import RegimeAdaptiveEngine
    engine = RegimeAdaptiveEngine()
    try:
        rows = engine.get_regime_strategy_performance()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"regime_performance": rows, "count": len(rows)}


@router.get("/data-quality")
def get_data_quality(symbol: Optional[str] = Query(None)):
    """
    Run data quality validation.
    Returns issues with severity levels and quality scores.
    """
    from data_engine.validation import DataValidator
    validator = DataValidator()

    if symbol:
        report = validator.validate(symbol.upper())
        if "error" in report:
            raise HTTPException(status_code=404, detail=report["error"])
        return report

    summary = validator.get_quality_summary()
    return summary


@router.get("/setup-quality/{symbol}")
def get_setup_quality_symbol(symbol: str):
    """
    Compute a live SetupQualityScore (0-100, grade A/B/C/D) for a single symbol.
    Pulls the latest confluence, breakout, feature, market-state, and vol-regime
    data and combines them into a weighted composite score.
    """
    from analytics_engine.setup_quality import SetupQualityScorer
    try:
        scorer = SetupQualityScorer()
        return scorer.score_symbol(symbol.upper())
    except Exception as e:
        logger.exception("setup-quality error for %s", symbol)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/setup-quality")
def get_setup_quality_leaderboard(
    min_grade: str = Query("B", description="Minimum grade to include: A/B/C/D"),
    limit:     int = Query(30,  le=100, description="Max results"),
):
    """
    Return top-scoring setups from the SetupQualityScore table (today).
    Ordered by quality_score descending.
    Grades: A ≥ 78  |  B 62-77  |  C 48-61  |  D < 48
    """
    from analytics_engine.setup_quality import SetupQualityScorer
    try:
        scorer = SetupQualityScorer()
        scores = scorer.get_scores(min_grade=min_grade.upper(), limit=limit)
        return {
            "scores":    scores,
            "count":     len(scores),
            "min_grade": min_grade.upper(),
        }
    except Exception as e:
        logger.exception("setup-quality leaderboard error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/regime-timeline")
def get_regime_timeline(
    symbol:     str           = Query(...,  description="Ticker symbol, e.g. SOXL"),
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD (default: 365 days ago)"),
    end_date:   Optional[str] = Query(None, description="End date YYYY-MM-DD (default: today)"),
):
    """
    Return a date-indexed market-regime + trade-event timeline for *symbol*.
    Includes regime bands, cumulative P&L, and per-trade entry/exit markers.
    Cached for 5 minutes per (symbol, start, end) combination.
    """
    from analytics_engine.regime_timeline import RegimeTimelineEngine
    try:
        engine = RegimeTimelineEngine()
        return engine.get_timeline(
            symbol=symbol.upper(),
            start_date=start_date,
            end_date=end_date,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("regime-timeline error for %s", symbol)
        raise HTTPException(status_code=500, detail=str(e))
