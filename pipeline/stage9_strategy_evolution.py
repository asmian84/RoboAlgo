"""Stage 9 — Strategy Evolution.

Runs the genetic algorithm to evolve trading strategies using
historical backtest data. Typically run weekly, not daily.

Pipeline position:
  … → Stage 8 Market Physics → ▶ Stage 9 Strategy Evolution
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
from sqlalchemy import select

from database.connection import get_session
from database.models import Instrument, PriceData, Indicator, StrategyGenome
from strategy_evolution_engine.evolver import StrategyEvolver

logger = logging.getLogger("pipeline.stage9_strategy_evolution")


def run(
    population_size: int = 30,
    n_generations: int = 10,
    symbols: list[str] | None = None,
) -> int:
    """Run Stage 9: evolve strategies on historical data.

    Args:
        population_size: Size of the genome population.
        n_generations: Number of evolution generations.
        symbols: Subset of symbols to use for backtesting (None = all).

    Returns:
        Number of top strategies stored.
    """
    with get_session() as session:
        StrategyGenome.__table__.create(bind=session.bind, checkfirst=True)

        # Load price + ATR data for backtesting
        if symbols:
            instruments = session.execute(
                select(Instrument).where(Instrument.symbol.in_([s.upper() for s in symbols]))
            ).scalars().all()
        else:
            instruments = session.execute(select(Instrument)).scalars().all()

        # Build price_data dict for the evolver
        price_data: dict[str, list] = {}
        for inst in instruments[:50]:  # cap at 50 symbols for performance
            rows = session.execute(
                select(PriceData.date, PriceData.close)
                .where(PriceData.instrument_id == inst.id)
                .order_by(PriceData.date)
            ).all()

            # Get ATR from indicators
            ind_rows = session.execute(
                select(Indicator.date, Indicator.atr)
                .where(Indicator.instrument_id == inst.id)
                .order_by(Indicator.date)
            ).all()

            if len(rows) < 100:
                continue

            atr_map = {str(r.date): float(r.atr or 0) for r in ind_rows}
            bars = []
            for r in rows:
                dt = str(r.date)
                bars.append({
                    "date": dt,
                    "close": float(r.close or 0),
                    "atr": atr_map.get(dt, float(r.close or 1) * 0.02),
                })
            price_data[inst.symbol] = bars

        if not price_data:
            logger.warning("Stage 9: no price data available for evolution")
            return 0

        # Run evolution
        evolver = StrategyEvolver(
            population_size=population_size,
            n_generations=n_generations,
        )
        top_strategies = evolver.evolve(price_data)

        # Store top strategies
        generation = 0
        for entry in top_strategies:
            generation = max(generation, entry.get("generation", 0))

        import json
        count = 0
        for entry in top_strategies[:10]:  # store top 10
            genome = entry["genome"]
            metrics = entry["metrics"]

            record = StrategyGenome(
                generation=entry.get("generation", 0),
                genome_id=genome["genome_id"],
                entry_confluence_min=genome.get("entry_confluence_min"),
                pattern_type=genome.get("pattern_type"),
                regime_filter=genome.get("regime_filter"),
                stop_atr_mult=genome.get("stop_atr_mult"),
                target_atr_mult=genome.get("target_atr_mult"),
                hold_days_max=genome.get("hold_days_max"),
                fitness=entry.get("fitness", 0),
                sharpe_ratio=metrics.get("sharpe_ratio"),
                win_rate=metrics.get("win_rate"),
                profit_factor=metrics.get("profit_factor"),
                max_drawdown=metrics.get("max_drawdown"),
                trade_count=metrics.get("trade_count"),
                is_active=False,
                genome_data=json.dumps(genome),
            )
            session.add(record)
            count += 1

        session.commit()
        logger.info("Stage 9 Strategy Evolution: stored %d top strategies", count)
        return count
