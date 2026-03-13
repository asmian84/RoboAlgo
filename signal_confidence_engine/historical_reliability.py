"""Historical reliability scorer for trading signals.

Evaluates how reliable past signals were for a given symbol by
analysing the win rate and profit factor of recent signal outcomes.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("signal_confidence_engine.historical_reliability")


def compute_historical_reliability(symbol: str) -> float:
    """Score historical signal reliability for *symbol* on a 0-100 scale.

    Logic
    -----
    1. Fetch the last 50 signals for this instrument from the database.
    2. For each signal that has a matching paper trade (outcome data),
       determine whether it was a winner (return_percent > 0).
    3. Compute win rate and profit factor from completed trades.
    4. Score = win_rate * 50 + min(profit_factor, 3) / 3 * 50
    5. If fewer than 10 signals with outcomes exist, return 50.0 (neutral).
    """
    from sqlalchemy import select, desc

    from database.connection import get_session
    from database.models import Signal, Instrument, PaperTrade

    sym = symbol.upper()

    with get_session() as session:
        # Resolve instrument
        inst = session.execute(
            select(Instrument).where(Instrument.symbol == sym)
        ).scalar_one_or_none()
        if inst is None:
            return 50.0

        # Get last 50 signals for this instrument
        signals = session.execute(
            select(Signal)
            .where(Signal.instrument_id == inst.id)
            .order_by(desc(Signal.date))
            .limit(50)
        ).scalars().all()

        if len(signals) < 10:
            return 50.0

        # Get paper trades for this symbol to measure outcomes
        trades = session.execute(
            select(PaperTrade)
            .where(PaperTrade.symbol == sym)
            .order_by(desc(PaperTrade.exit_date))
            .limit(50)
        ).scalars().all()

        if len(trades) < 10:
            return 50.0

        # Compute win rate and profit factor
        wins = 0
        total_profit = 0.0
        total_loss = 0.0

        for trade in trades:
            ret = float(trade.return_percent or 0.0)
            if ret > 0:
                wins += 1
                total_profit += ret
            else:
                total_loss += abs(ret)

        n = len(trades)
        win_rate = wins / n if n > 0 else 0.0

        profit_factor = (
            total_profit / total_loss if total_loss > 0
            else 3.0 if total_profit > 0
            else 1.0
        )

        # Score: win_rate contributes 50 pts, profit_factor contributes 50 pts
        score = win_rate * 50.0 + min(profit_factor, 3.0) / 3.0 * 50.0
        return max(0.0, min(100.0, score))
