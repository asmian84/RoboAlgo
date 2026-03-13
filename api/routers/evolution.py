"""
RoboAlgo — Strategy Evolution API Router

GET  /api/evolution/report              — full evolution report
GET  /api/evolution/strategy/{setup}    — single strategy fitness
POST /api/evolution/apply               — placeholder (requires human review gate)
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/report")
def get_evolution_report():
    """
    Full strategy evolution report.

    Returns per-strategy fitness scores, performance stats, and
    optimization suggestions. No parameters are changed automatically.
    """
    try:
        from analytics_engine.strategy_evolution import StrategyEvolutionEngine
        return StrategyEvolutionEngine().get_evolution_report()
    except Exception as e:
        logger.error("evolution/report failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategy/{setup_type}")
def get_strategy_fitness(setup_type: str):
    """Return fitness score and performance stats for a single strategy."""
    try:
        from analytics_engine.strategy_evolution import StrategyEvolutionEngine
        return StrategyEvolutionEngine().calculate_strategy_fitness(setup_type)
    except Exception as e:
        logger.error("evolution/strategy/%s failed: %s", setup_type, e)
        raise HTTPException(status_code=500, detail=str(e))


class ApplyRequest(BaseModel):
    setup_type:                str
    new_quality_threshold:     float | None = None
    new_position_multiplier:   float | None = None
    new_risk_per_trade:        float | None = None
    reason:                    str = ""


@router.post("/apply")
def apply_evolution(req: ApplyRequest):
    """
    Apply reviewed optimization suggestions.

    SAFETY: Requires minimum 100 trades before any changes are accepted.
    All changes are logged and reversible via playbook version history.
    """
    try:
        from analytics_engine.strategy_evolution import StrategyEvolutionEngine, MIN_SAMPLE_SIZE
        engine = StrategyEvolutionEngine()
        stats  = engine.calculate_strategy_fitness(req.setup_type)

        sample = stats.get("return_count", 0) if isinstance(stats, dict) else 0
        if sample < MIN_SAMPLE_SIZE:
            raise HTTPException(
                status_code=422,
                detail=f"Insufficient data: {sample} trades < required {MIN_SAMPLE_SIZE}. "
                       f"No parameters have been changed."
            )

        # Record proposed changes (actual DB update to playbook is a future step)
        import logging as _log
        _log.getLogger("strategy_evolution").info(
            "EVOLUTION APPLY: strategy=%s, q_threshold=%s, pos_mult=%s, risk=%s, reason=%s",
            req.setup_type, req.new_quality_threshold,
            req.new_position_multiplier, req.new_risk_per_trade, req.reason,
        )

        return {
            "accepted":   True,
            "setup_type": req.setup_type,
            "sample_size":sample,
            "message":    "Parameter suggestions recorded for review. Manual playbook update required.",
            "warning":    "Automatic parameter application is disabled for safety. Review suggestions manually.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("evolution/apply failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
