"""Stage 7 — Price-Time Confluence Nodes.

Combines cycle projections, geometry levels, pattern data, and swing
structure to identify Decision Nodes — high-probability price-time zones.

Pipeline position:
  … → Stage 5 Price Levels → Stage 6 Cycles
  → ▶ Stage 7 Confluence Nodes → Stage 8 Market Physics
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from database.connection import get_session
from database.models import Instrument, PriceData, ConfluenceNode
from confluence_engine.node_detector import detect_confluence_nodes
from pattern_engine.service import PatternService

logger = logging.getLogger("pipeline.stage7_confluence_nodes")

_pattern_service = PatternService()


def run(symbol: str | None = None) -> int:
    """Run Stage 7 for one or all instruments."""
    with get_session() as session:
        ConfluenceNode.__table__.create(bind=session.bind, checkfirst=True)

        if symbol:
            instruments = session.execute(
                select(Instrument).where(Instrument.symbol == symbol.upper())
            ).scalars().all()
        else:
            instruments = session.execute(select(Instrument)).scalars().all()

        count = 0
        for inst in instruments:
            try:
                _process(session, inst)
                count += 1
            except Exception as exc:
                logger.warning("Stage 7 failed for %s: %s", inst.symbol, exc)

        logger.info("Stage 7 Confluence Nodes: processed %d instruments", count)
        return count


def _process(session, inst: Instrument) -> None:
    """Process a single instrument."""
    rows = session.execute(
        select(PriceData.date, PriceData.open, PriceData.high,
               PriceData.low, PriceData.close, PriceData.volume)
        .where(PriceData.instrument_id == inst.id)
        .order_by(PriceData.date.desc())
        .limit(500)
    ).all()

    if len(rows) < 60:
        return

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = df["date"].astype(str)
    df = df.sort_values("date").reset_index(drop=True)

    # Get latest patterns for this symbol
    try:
        patterns = _pattern_service.detect_for_symbol(inst.symbol, resolution_minutes=0)
    except Exception:
        patterns = []

    nodes = detect_confluence_nodes(df, symbol=inst.symbol, patterns=patterns)

    today = date.today()
    for node in nodes[:6]:  # store top 6 nodes per symbol
        record = {
            "instrument_id": inst.id,
            "date": today,
            "price_low": node["price_low"],
            "price_high": node["price_high"],
            "time_window_start": node.get("time_window_start"),
            "time_window_end": node.get("time_window_end"),
            "confluence_score": node["confluence_score"],
            "component_scores": node.get("component_scores"),
            "supporting_signals": node.get("supporting_signals"),
            "node_type": node.get("node_type"),
            "direction": node.get("direction"),
            "status": node.get("status", "upcoming"),
        }
        session.add(ConfluenceNode(**record))

    session.commit()
