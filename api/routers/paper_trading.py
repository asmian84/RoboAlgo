"""
RoboAlgo — Paper Trading API
Endpoints for running and querying the paper trading simulation.
"""

from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from paper_engine.trader import PaperTrader, STARTING_BALANCE

router = APIRouter()


# ── Request Models ─────────────────────────────────────────────────────────────

class SimulateRequest(BaseModel):
    start_date: date
    end_date: date
    reset: bool = False


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/simulate")
def run_simulation(req: SimulateRequest):
    """
    Run paper trading simulation over a date range.

    Body:
      - start_date: ISO date string (e.g. "2025-01-01")
      - end_date:   ISO date string (e.g. "2025-03-01")
      - reset:      true to clear existing data before running (default false)

    Returns performance summary + trade list.
    """
    if req.end_date < req.start_date:
        raise HTTPException(status_code=400, detail="end_date must be >= start_date")

    max_days = 365 * 3
    if (req.end_date - req.start_date).days > max_days:
        raise HTTPException(status_code=400, detail=f"Max simulation window is {max_days} days")

    trader = PaperTrader()
    try:
        summary = trader.run_simulation(
            start_date=req.start_date,
            end_date=req.end_date,
            reset=req.reset,
        )
        return summary
    finally:
        trader.close()


@router.get("/account")
def get_account():
    """Current account balance and most recent daily snapshot."""
    trader = PaperTrader()
    try:
        snapshots = trader._get_snapshots()
        if not snapshots:
            return {
                "balance":           STARTING_BALANCE,
                "total_return_pct":  0.0,
                "total_pnl":         0.0,
                "open_positions":    0,
                "last_snapshot":     None,
            }
        last = snapshots[-1]
        total_pnl = last["ending_balance"] - STARTING_BALANCE
        return {
            "balance":          round(last["ending_balance"], 2),
            "total_return_pct": round(total_pnl / STARTING_BALANCE * 100, 2),
            "total_pnl":        round(total_pnl, 2),
            "open_positions":   last["open_positions"],
            "last_snapshot":    last,
        }
    finally:
        trader.close()


@router.get("/positions")
def get_open_positions():
    """All currently open paper positions."""
    trader = PaperTrader()
    try:
        return trader.get_open_positions()
    finally:
        trader.close()


@router.get("/trades")
def get_trade_history(limit: int = Query(default=200, ge=1, le=1000)):
    """Closed trade history, newest first."""
    trader = PaperTrader()
    try:
        return trader.get_trade_history(limit=limit)
    finally:
        trader.close()


@router.get("/report")
def get_report(
    start_date: Optional[date] = Query(default=None),
    end_date:   Optional[date] = Query(default=None),
):
    """
    Full performance report. Optionally filter by date range.
    Includes: summary stats, daily equity curve, trade list.
    """
    trader = PaperTrader()
    try:
        return trader.get_summary(start_date=start_date, end_date=end_date)
    finally:
        trader.close()


@router.delete("/reset")
def reset_paper_account():
    """Clear ALL paper trading data (positions, trades, account history)."""
    trader = PaperTrader()
    try:
        trader.reset()
        return {"status": "reset", "message": "All paper trading data cleared."}
    finally:
        trader.close()


@router.get("/equity")
def get_equity_curve(
    start_date: Optional[date] = Query(default=None),
    end_date:   Optional[date] = Query(default=None),
):
    """Daily equity curve data for charting."""
    trader = PaperTrader()
    try:
        return trader._get_snapshots(start_date=start_date, end_date=end_date)
    finally:
        trader.close()
