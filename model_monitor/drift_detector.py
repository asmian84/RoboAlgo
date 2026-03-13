"""Model Monitor — Drift Detector.

Compare recent vs. historical performance metrics to detect
strategy degradation in live markets.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("model_monitor.drift_detector")


def detect_drift(setup_type: str) -> dict:
    """Detect performance drift for a given setup type.

    Computes metrics over two windows:
        recent   = last 30 days
        baseline = last 180 days

    Drift scores are the difference (recent - baseline).  Negative drift
    indicates degradation.

    State thresholds
    ----------------
    HEALTHY:  win_rate_drift > -0.05  AND  pf_drift > -0.3  AND  dd_drift > -0.10
    CRITICAL: win_rate_drift < -0.15  OR   pf_drift < -0.8  OR   dd_drift < -0.25
    WARNING:  everything else

    Returns:
        dict with keys: state, recent_metrics, baseline_metrics,
        drift_scores, recommended_action.
    """
    from model_monitor.live_metrics import compute_live_metrics

    recent = compute_live_metrics(setup_type, lookback_days=30)
    baseline = compute_live_metrics(setup_type, lookback_days=180)

    # Compute drift scores
    win_rate_drift = recent["win_rate"] - baseline["win_rate"]
    pf_drift = recent["profit_factor"] - baseline["profit_factor"]
    return_drift = recent["average_return"] - baseline["average_return"]
    dd_drift = recent["max_drawdown"] - baseline["max_drawdown"]

    drift_scores = {
        "win_rate_drift": round(win_rate_drift, 4),
        "pf_drift": round(pf_drift, 4),
        "return_drift": round(return_drift, 4),
        "dd_drift": round(dd_drift, 4),
    }

    # Determine state
    if (
        win_rate_drift < -0.15
        or pf_drift < -0.8
        or dd_drift < -0.25
    ):
        state = "CRITICAL"
    elif (
        win_rate_drift > -0.05
        and pf_drift > -0.3
        and dd_drift > -0.10
    ):
        state = "HEALTHY"
    else:
        state = "WARNING"

    # Recommended action
    action_map = {
        "HEALTHY": "maintain",
        "WARNING": "reduce_size",
        "CRITICAL": "disable",
    }

    return {
        "state": state,
        "recent_metrics": recent,
        "baseline_metrics": baseline,
        "drift_scores": drift_scores,
        "recommended_action": action_map[state],
    }
