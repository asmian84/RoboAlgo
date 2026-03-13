"""Stage 11 — Model Monitoring.

Evaluates all strategy setup types for performance drift and
upserts health records into the strategy_health table.

Pipeline position:
  … → Stage 9 Strategy Evolution → … → ▶ Stage 11 Model Monitor
"""

from __future__ import annotations

import json
import logging
from datetime import date

from sqlalchemy.dialects.postgresql import insert as pg_insert

from database.connection import get_session
from database.models import StrategyHealth
from model_monitor.signal_guard import evaluate_all_strategies

logger = logging.getLogger("pipeline.stage11_model_monitor")

UPSERT_COLS = [
    "state",
    "action",
    "win_rate",
    "profit_factor",
    "average_return",
    "max_drawdown",
    "trade_count",
    "win_rate_drift",
    "pf_drift",
    "return_drift",
    "dd_drift",
    "recent_metrics",
    "baseline_metrics",
]


def run() -> int:
    """Run Stage 11: model monitoring for all strategies.

    Returns:
        Number of strategies monitored.
    """
    report = evaluate_all_strategies()
    strategies = report.get("strategies", [])

    if not strategies:
        logger.info("Stage 11 Model Monitor: no strategies to monitor")
        return 0

    today = date.today()

    session = get_session()
    try:
        StrategyHealth.__table__.create(bind=session.bind, checkfirst=True)

        count = 0
        for strat in strategies:
            # Fetch full drift result for recent/baseline JSON
            from model_monitor.drift_detector import detect_drift

            detail = detect_drift(strat["setup_type"])

            record = {
                "setup_type": strat["setup_type"],
                "date": today,
                "state": strat["state"],
                "action": strat["action"],
                "win_rate": strat["win_rate"],
                "profit_factor": strat["profit_factor"],
                "average_return": detail["recent_metrics"]["average_return"],
                "max_drawdown": detail["recent_metrics"]["max_drawdown"],
                "trade_count": strat["trade_count"],
                "win_rate_drift": detail["drift_scores"]["win_rate_drift"],
                "pf_drift": detail["drift_scores"]["pf_drift"],
                "return_drift": detail["drift_scores"]["return_drift"],
                "dd_drift": detail["drift_scores"]["dd_drift"],
                "recent_metrics": json.dumps(detail["recent_metrics"]),
                "baseline_metrics": json.dumps(detail["baseline_metrics"]),
            }

            stmt = pg_insert(StrategyHealth).values([record])
            stmt = stmt.on_conflict_do_update(
                constraint="uq_strathlth_setup_date",
                set_={col: stmt.excluded[col] for col in UPSERT_COLS},
            )
            session.execute(stmt)
            count += 1

        session.commit()
        logger.info(
            "Stage 11 Model Monitor: upserted %d strategies (overall: %s)",
            count,
            report["overall_health"],
        )
        return count
    except Exception as exc:
        session.rollback()
        logger.error("Stage 11 Model Monitor failed: %s", exc)
        raise
    finally:
        session.close()
