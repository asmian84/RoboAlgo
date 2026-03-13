"""Instrument endpoints."""

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from config.settings import (
    INDEX_DRIVERS, INDEX_LEVERAGED, SECTOR_LEVERAGED,
    COMMODITY_LEVERAGED, SINGLE_STOCK_LEVERAGED, UNDERLYING_LEADERS,
)
from database.connection import get_session
from database.models import Instrument

router = APIRouter()


@router.get("")
def list_instruments():
    """List all instruments with metadata."""
    session = get_session()
    try:
        rows = session.execute(
            select(Instrument).order_by(Instrument.symbol)
        ).scalars().all()
        return [
            {
                "symbol": r.symbol,
                "name": r.name,
                "instrument_type": r.instrument_type,
                "leverage_factor": r.leverage_factor,
                "underlying": r.underlying,
                "pair_symbol": r.pair_symbol,
            }
            for r in rows
        ]
    finally:
        session.close()


@router.get("/categories")
def get_categories():
    """Instruments grouped by category."""
    def pairs_to_symbols(pairs):
        syms = []
        for bull, bear, _, *_ in pairs:
            syms.append(bull)
            if bear:
                syms.append(bear)
        return syms

    return {
        "index_drivers": INDEX_DRIVERS,
        "index_leveraged": pairs_to_symbols(INDEX_LEVERAGED),
        "sector_leveraged": pairs_to_symbols(SECTOR_LEVERAGED),
        "commodity_leveraged": pairs_to_symbols(COMMODITY_LEVERAGED),
        "single_stock_leveraged": pairs_to_symbols(SINGLE_STOCK_LEVERAGED),
        "underlying_leaders": UNDERLYING_LEADERS,
    }


@router.get("/{symbol}")
def get_instrument(symbol: str):
    """Get single instrument detail."""
    session = get_session()
    try:
        inst = session.execute(
            select(Instrument).where(Instrument.symbol == symbol.upper())
        ).scalar()
        if not inst:
            raise HTTPException(404, f"Instrument not found: {symbol}")
        return {
            "symbol": inst.symbol,
            "name": inst.name,
            "instrument_type": inst.instrument_type,
            "leverage_factor": inst.leverage_factor,
            "underlying": inst.underlying,
            "pair_symbol": inst.pair_symbol,
        }
    finally:
        session.close()
