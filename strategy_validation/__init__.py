"""
strategy_validation — standalone, pure validation gate for candidate strategies.

NOT wired into auto_trader/, strategy_evolution_engine/, or any pipeline stage.
See strategy_validation.gates for the full rationale.
"""

from strategy_validation.gates import (
    PerformanceStats,
    StrategyCandidate,
    ValidationLimits,
    Verdict,
    deflated_sharpe,
    evaluate,
    expected_max_sharpe,
)

__all__ = [
    "PerformanceStats",
    "StrategyCandidate",
    "ValidationLimits",
    "Verdict",
    "deflated_sharpe",
    "evaluate",
    "expected_max_sharpe",
]
