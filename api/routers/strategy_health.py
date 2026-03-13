"""Strategy Health API — model monitoring endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("")
def get_strategy_health():
    """Get health status for all monitored strategies."""
    from model_monitor.signal_guard import evaluate_all_strategies

    return evaluate_all_strategies()


@router.get("/{setup_type}")
def get_strategy_health_detail(setup_type: str):
    """Get detailed health for a specific strategy."""
    from model_monitor.drift_detector import detect_drift

    return detect_drift(setup_type)
