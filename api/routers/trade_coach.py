"""
RoboAlgo — Trade Coach API Router

GET  /api/trade-coach/signal/{symbol}    — signal explanation + scenario map
GET  /api/trade-coach/similar/{symbol}   — historical setup statistics
GET  /api/trade-coach/review/{trade_id}  — completed trade review
"""

import logging
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/signal/{symbol}")
def get_signal_explanation(symbol: str):
    """
    Return a structured explanation of the current signal for `symbol`.

    Includes:
      - market_state, setup_type, setup_quality_score
      - evidence[] — reasons supporting the setup
      - risk_factors[] — warnings about this setup
      - scenario_map[] — probability-weighted price scenarios
    """
    try:
        from analytics_engine.trade_coach import TradeCoachEngine
        return TradeCoachEngine().generate_signal_explanation(symbol)
    except Exception as e:
        logger.error("trade-coach/signal/%s failed: %s", symbol, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/similar/{symbol}")
def get_similar_setups(symbol: str):
    """
    Return historical performance stats for the same setup type as `symbol`.

    Includes win rate, average return, max drawdown, profit factor, sample size.
    """
    try:
        from analytics_engine.trade_coach import TradeCoachEngine
        return TradeCoachEngine().find_similar_setups(symbol)
    except Exception as e:
        logger.error("trade-coach/similar/%s failed: %s", symbol, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/review/{trade_id}")
def review_trade(trade_id: int):
    """
    Review a completed trade and score entry/exit quality.

    Returns entry_quality, exit_quality, missed_profit_pct, and a verdict string.
    """
    try:
        from analytics_engine.trade_coach import TradeCoachEngine
        result = TradeCoachEngine().review_completed_trade(trade_id)
        if "error" in result and result["error"] == "Trade not found":
            raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("trade-coach/review/%d failed: %s", trade_id, e)
        raise HTTPException(status_code=500, detail=str(e))
