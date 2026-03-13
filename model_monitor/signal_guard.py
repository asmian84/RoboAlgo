"""Model Monitor — Signal Guard.

Master guard that monitors all strategies and aggregates health status.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("model_monitor.signal_guard")


def evaluate_all_strategies() -> dict:
    """Evaluate drift for every known setup type and aggregate results.

    Queries unique setup_types from the TradeLifecycle table (closed trades),
    runs :func:`detect_drift` on each, and returns a combined health report.

    Returns:
        dict with keys: overall_health, strategies, healthy_count,
        warning_count, critical_count.
    """
    from sqlalchemy import select, distinct

    from database.connection import get_session
    from database.models import TradeLifecycle
    from model_monitor.drift_detector import detect_drift

    session = get_session()
    try:
        setup_types = (
            session.execute(
                select(distinct(TradeLifecycle.setup_type))
                .where(TradeLifecycle.setup_type.isnot(None))
            )
            .scalars()
            .all()
        )
    finally:
        session.close()

    strategies = []
    healthy_count = 0
    warning_count = 0
    critical_count = 0

    for st in setup_types:
        try:
            result = detect_drift(st)
        except Exception as exc:
            logger.warning("drift detection failed for %s: %s", st, exc)
            continue

        state = result["state"]
        if state == "HEALTHY":
            healthy_count += 1
        elif state == "WARNING":
            warning_count += 1
        else:
            critical_count += 1

        strategies.append({
            "setup_type": st,
            "state": state,
            "action": result["recommended_action"],
            "win_rate": result["recent_metrics"]["win_rate"],
            "profit_factor": result["recent_metrics"]["profit_factor"],
            "win_rate_drift": result["drift_scores"]["win_rate_drift"],
            "trade_count": result["recent_metrics"]["trade_count"],
        })

    # Overall health = worst state among all strategies
    if critical_count > 0:
        overall_health = "CRITICAL"
    elif warning_count > 0:
        overall_health = "WARNING"
    else:
        overall_health = "HEALTHY"

    return {
        "overall_health": overall_health,
        "strategies": strategies,
        "healthy_count": healthy_count,
        "warning_count": warning_count,
        "critical_count": critical_count,
    }
