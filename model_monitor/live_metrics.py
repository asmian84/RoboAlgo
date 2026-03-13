"""Model Monitor — Live Performance Metrics.

Compute live performance metrics for a strategy/setup type
from completed paper trades (TradeLifecycle with state=EXIT).
"""

from __future__ import annotations

import logging

logger = logging.getLogger("model_monitor.live_metrics")


def compute_live_metrics(setup_type: str, lookback_days: int = 90) -> dict:
    """Compute live performance metrics for a given setup type.

    Queries closed trades (TradeLifecycle state=EXIT) matching *setup_type*
    within the last *lookback_days* and computes:

        win_rate, profit_factor, average_return, max_drawdown, trade_count

    Args:
        setup_type:    Strategy setup type (e.g. "compression_breakout").
        lookback_days: Number of calendar days to look back.

    Returns:
        dict with keys: win_rate, profit_factor, average_return,
        max_drawdown, trade_count.  Returns zero-filled dict when
        no trades are found.
    """
    import numpy as np
    from datetime import date, timedelta
    from sqlalchemy import select

    from database.connection import get_session
    from database.models import TradeLifecycle

    empty = {
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "average_return": 0.0,
        "max_drawdown": 0.0,
        "trade_count": 0,
    }

    cutoff = date.today() - timedelta(days=lookback_days)

    session = get_session()
    try:
        rows = session.execute(
            select(TradeLifecycle)
            .where(
                TradeLifecycle.setup_type == setup_type,
                TradeLifecycle.state == "EXIT",
                TradeLifecycle.exit_timestamp >= cutoff,
            )
            .order_by(TradeLifecycle.exit_timestamp.asc())
        ).scalars().all()

        if not rows:
            return empty

        returns = np.array(
            [float(t.return_percent or 0.0) for t in rows], dtype=np.float64
        )
        pnls = np.array(
            [float(t.pnl or 0.0) for t in rows], dtype=np.float64
        )

        total_closed = len(returns)
        wins = int(np.sum(returns > 0))

        # Win rate
        win_rate = wins / total_closed if total_closed > 0 else 0.0

        # Profit factor
        positive_sum = float(np.sum(returns[returns > 0]))
        negative_sum = float(np.abs(np.sum(returns[returns < 0])))
        profit_factor = (
            positive_sum / negative_sum if negative_sum > 0 else float("inf")
        )

        # Average return
        average_return = float(np.mean(returns))

        # Max drawdown — worst peak-to-trough in cumulative PnL
        cumulative_pnl = np.cumsum(pnls)
        running_peak = np.maximum.accumulate(cumulative_pnl)
        drawdowns = cumulative_pnl - running_peak
        max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        return {
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "average_return": round(average_return, 4),
            "max_drawdown": round(max_drawdown, 4),
            "trade_count": total_closed,
        }
    except Exception as exc:
        logger.error("compute_live_metrics failed for %s: %s", setup_type, exc)
        return empty
    finally:
        session.close()
