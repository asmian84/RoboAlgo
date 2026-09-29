"""
api/routers/paper_account.py
────────────────────────────────────────────────────────────────────────────────
Read/administer the PaperBroker account — the local simulated broker the
auto-trader actually trades into (`broker_engine/paper.py`, backed by the
`paper_account` / `paper_positions` / `paper_trades` tables).

Why this router exists rather than reusing `/api/paper` (paper_trading.py):
  `/api/paper` wires to `paper_engine/trader.py`, a HISTORICAL BACKTEST
  simulator with different accounting semantics — it treats `ending_balance`
  as equity. PaperBroker treats `ending_balance` as a cash ledger and derives
  equity as cash + open position market value on every read. The two are not
  interchangeable, so the PAPER page's ACCOUNT tab needs its own endpoints
  reading PaperBroker directly rather than the backtester.

Endpoints:
  GET   /api/paper-account/account     Cash, equity, buying power
  GET   /api/paper-account/positions   Open positions incl. stop/target being managed to
  GET   /api/paper-account/trades      Closed trades, newest first
  POST  /api/paper-account/mark-to-market  Re-price positions, close any past stop/target
  POST  /api/paper-account/reset       Wipe the simulation, restore starting cash

No auth — this is a local simulated account with zero credentials and no real
money at risk (mirrors PaperBroker's own zero-credential design).
"""
from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy import select, desc

from broker_engine.paper import PaperBroker
from database.connection import get_session
from database.models import PaperPosition, PaperTrade

router = APIRouter()

_broker = PaperBroker()


# ── Account ───────────────────────────────────────────────────────────────────

@router.get("/account")
def get_account():
    """Cash, equity (cash + open position market value), buying power."""
    acct = _broker.get_account()
    return {
        "account_id": acct.account_id,
        "cash": acct.cash,
        "equity": acct.portfolio_val,
        "buying_power": acct.buying_power,
        "broker": acct.broker,
        "currency": acct.currency,
        "is_paper": acct.is_paper,
    }


# ── Positions ─────────────────────────────────────────────────────────────────

@router.get("/positions")
def get_positions():
    """
    Open positions with current market value / unrealized P&L, plus the
    stop/target each position is being managed to (read directly off
    `paper_positions` — PaperBroker's Position dataclass doesn't carry them).
    """
    positions = _broker.get_positions()

    with get_session() as session:
        rows = session.execute(select(PaperPosition)).scalars().all()
        levels = {
            r.symbol: {"stop": r.stop_price, "target": r.target_price, "entry_date": str(r.entry_date)}
            for r in rows
        }

    return [
        {
            "symbol": p.symbol,
            "qty": p.qty,
            "avg_cost": p.avg_cost,
            "market_value": p.market_value,
            "unrealized_pnl": p.unrealized_pnl,
            "unrealized_pct": p.unrealized_pct,
            "side": p.side,
            "stop": levels.get(p.symbol, {}).get("stop"),
            "target": levels.get(p.symbol, {}).get("target"),
            "entry_date": levels.get(p.symbol, {}).get("entry_date"),
        }
        for p in positions
    ]


# ── Closed trades ─────────────────────────────────────────────────────────────

@router.get("/trades")
def get_trades(limit: int = Query(default=200, ge=1, le=1000)):
    """Closed trades, newest first."""
    with get_session() as session:
        rows = session.execute(
            select(PaperTrade).order_by(desc(PaperTrade.exit_date), desc(PaperTrade.id)).limit(limit)
        ).scalars().all()
        return [
            {
                "id": r.id,
                "symbol": r.symbol,
                "entry_date": str(r.entry_date),
                "exit_date": str(r.exit_date),
                "entry_price": r.entry_price,
                "exit_price": r.exit_price,
                "position_size": r.position_size,
                "direction": r.direction,
                "pnl": r.pnl,
                "return_percent": r.return_percent,
                "exit_reason": r.exit_reason,
            }
            for r in rows
        ]


# ── Actions ───────────────────────────────────────────────────────────────────

@router.post("/mark-to-market")
def mark_to_market():
    """
    Re-price every open position. Closes any whose stop or target has been
    crossed, booking realized P&L — this is what makes the account move.
    """
    return _broker.mark_to_market()


class ResetPayload(BaseModel):
    starting_balance: float | None = None


@router.post("/reset")
def reset(payload: ResetPayload = ResetPayload()):
    """
    Wipe all paper positions/trades and restore the starting cash balance.
    `starting_balance` is optional — omit it to keep PaperBroker's existing
    default (env var `PAPER_STARTING_BALANCE` or $100,000).
    """
    acct = _broker.reset(starting_balance=payload.starting_balance)
    return {
        "status": "reset",
        "account_id": acct.account_id,
        "cash": acct.cash,
        "equity": acct.portfolio_val,
    }
