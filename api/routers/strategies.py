"""
RoboAlgo — Strategy Evolution API Router
Serves top evolved strategy genomes and their fitness metrics.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from sqlalchemy import select, desc

router = APIRouter()


@router.get("/top")
def get_top_strategies(
    limit: int = Query(10, ge=1, le=50),
    active_only: bool = Query(False),
) -> dict[str, Any]:
    """Return top evolved strategy genomes ranked by fitness."""
    from database.connection import get_session
    from database.models import StrategyGenome

    with get_session() as session:
        # Ensure table exists
        try:
            StrategyGenome.__table__.create(bind=session.bind, checkfirst=True)
        except Exception:
            pass

        query = select(StrategyGenome).order_by(desc(StrategyGenome.fitness))
        if active_only:
            query = query.where(StrategyGenome.is_active == True)  # noqa: E712
        query = query.limit(limit)

        rows = session.execute(query).scalars().all()

        strategies = []
        for g in rows:
            strategies.append({
                "genome_id": g.genome_id,
                "generation": g.generation,
                "entry_confluence_min": g.entry_confluence_min,
                "pattern_type": g.pattern_type,
                "regime_filter": g.regime_filter,
                "stop_atr_mult": g.stop_atr_mult,
                "target_atr_mult": g.target_atr_mult,
                "hold_days_max": g.hold_days_max,
                "fitness": g.fitness,
                "sharpe_ratio": g.sharpe_ratio,
                "win_rate": g.win_rate,
                "profit_factor": g.profit_factor,
                "max_drawdown": g.max_drawdown,
                "trade_count": g.trade_count,
                "is_active": g.is_active,
            })

        return {"strategies": strategies, "count": len(strategies)}
