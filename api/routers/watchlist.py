"""
RoboAlgo — Watchlist API Router

GET    /api/watchlist          — list all saved symbols
POST   /api/watchlist/{symbol} — add a symbol to the watchlist
DELETE /api/watchlist/{symbol} — remove a symbol from the watchlist
"""

from fastapi import APIRouter
from sqlalchemy import select

from database.connection import get_session
from database.models import Watchlist

router = APIRouter()


@router.get("")
def list_watchlist():
    """Return all watchlisted symbols, newest first."""
    session = get_session()
    try:
        rows = session.execute(
            select(Watchlist).order_by(Watchlist.added_at.desc())
        ).scalars().all()
        return [{"symbol": r.symbol, "added_at": r.added_at.isoformat()} for r in rows]
    finally:
        session.close()


@router.post("/{symbol}")
def add_to_watchlist(symbol: str):
    """Add a symbol to the watchlist (idempotent)."""
    session = get_session()
    try:
        sym = symbol.upper()
        existing = session.execute(
            select(Watchlist).where(Watchlist.symbol == sym)
        ).scalar()
        if not existing:
            session.add(Watchlist(symbol=sym))
            session.commit()
        return {"symbol": sym, "watching": True}
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.delete("/{symbol}")
def remove_from_watchlist(symbol: str):
    """Remove a symbol from the watchlist."""
    session = get_session()
    try:
        sym = symbol.upper()
        existing = session.execute(
            select(Watchlist).where(Watchlist.symbol == sym)
        ).scalar()
        if existing:
            session.delete(existing)
            session.commit()
        return {"symbol": sym, "watching": False}
    finally:
        session.close()
