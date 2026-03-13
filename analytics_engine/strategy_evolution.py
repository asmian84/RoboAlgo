"""
RoboAlgo — Strategy Evolution Engine
Analyzes historical trades and computes per-strategy fitness scores.
Generates parameter optimization suggestions using deterministic statistical analysis.

Safety rule: parameters are NEVER updated automatically.
All suggestions require human review and explicit activation.
Minimum 100-trade sample required for any parameter change.
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select, func

from database.connection import get_session

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

FITNESS_WEIGHTS = {
    "win_rate":        0.30,
    "avg_return":      0.25,
    "profit_factor":   0.20,
    "drawdown":        0.15,  # inverted: low drawdown = high contribution
    "sample_size":     0.10,
}

MIN_SAMPLE_SIZE = 100   # Minimum trades required before suggesting param changes

STATUS_THRESHOLDS = {
    "strong":     80.0,
    "acceptable": 60.0,
    "weak":       40.0,
    # below 40 → "disabled"
}


# ── Performance computation ───────────────────────────────────────────────────

def _compute_strategy_stats(trades: list) -> dict:
    """Compute key performance metrics for a list of trade records."""
    if not trades:
        return {}

    returns = []
    for t in trades:
        if t.entry_price and t.exit_price and float(t.entry_price) > 0:
            r = (float(t.exit_price) - float(t.entry_price)) / float(t.entry_price)
            returns.append(r)

    if not returns:
        return {"trade_count": len(trades), "return_count": 0}

    wins   = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    win_rate    = len(wins) / len(returns)
    avg_return  = sum(returns) / len(returns)
    max_dd      = min(returns)

    gross_profit = sum(wins)   if wins   else 0.0
    gross_loss   = abs(sum(losses)) if losses else 1e-9
    profit_factor = gross_profit / gross_loss

    # Running drawdown calculation
    cum_return    = 0.0
    peak          = 0.0
    max_drawdown  = 0.0
    for r in returns:
        cum_return += r
        peak = max(peak, cum_return)
        max_drawdown = min(max_drawdown, cum_return - peak)

    # Sharpe ratio approximation
    if len(returns) > 1:
        import statistics
        std = statistics.stdev(returns) or 1e-9
        sharpe = (avg_return / std) * (252 ** 0.5)  # annualized
    else:
        sharpe = 0.0

    return {
        "trade_count":   len(trades),
        "return_count":  len(returns),
        "win_rate":      round(win_rate, 4),
        "avg_return":    round(avg_return, 6),
        "max_return":    round(max(returns), 4),
        "min_return":    round(min(returns), 4),
        "profit_factor": round(profit_factor, 3),
        "max_drawdown":  round(max_drawdown, 4),
        "sharpe_ratio":  round(sharpe, 3),
    }


def _fitness_score(stats: dict) -> float:
    """
    Compute 0–100 fitness score from strategy performance stats.
    Uses weighted combination of normalised metrics.
    """
    if not stats or stats.get("return_count", 0) < 5:
        return 0.0

    win_rate     = stats.get("win_rate", 0)
    avg_return   = stats.get("avg_return", 0)
    pf           = stats.get("profit_factor", 0)
    max_dd       = abs(stats.get("max_drawdown", 0.30))
    sample_size  = stats.get("return_count", 0)

    # Normalise each component to 0–100
    wr_score  = win_rate * 100                                   # 0→100 directly
    ret_score = max(0.0, min(100.0, (avg_return + 0.05) / 0.10 * 100))  # -5% → 0, +5% → 100
    pf_score  = max(0.0, min(100.0, (pf / 3.0) * 100))          # 0 → 0, 3 → 100
    dd_score  = max(0.0, 100.0 - max_dd * 500)                  # 0 dd → 100, 20% dd → 0
    sz_score  = min(100.0, sample_size / 200.0 * 100)           # 200+ trades → 100

    score = (
        wr_score  * FITNESS_WEIGHTS["win_rate"]      +
        ret_score * FITNESS_WEIGHTS["avg_return"]    +
        pf_score  * FITNESS_WEIGHTS["profit_factor"] +
        dd_score  * FITNESS_WEIGHTS["drawdown"]      +
        sz_score  * FITNESS_WEIGHTS["sample_size"]
    )
    return round(min(100.0, max(0.0, score)), 1)


def _status_label(fitness: float) -> str:
    if fitness >= STATUS_THRESHOLDS["strong"]:     return "strong"
    if fitness >= STATUS_THRESHOLDS["acceptable"]: return "acceptable"
    if fitness >= STATUS_THRESHOLDS["weak"]:       return "weak"
    return "disabled"


def _generate_suggestions(setup_type: str, stats: dict, fitness: float) -> list[str]:
    """Generate actionable optimization suggestions for a strategy."""
    suggestions = []
    sample = stats.get("return_count", 0)

    if sample < MIN_SAMPLE_SIZE:
        suggestions.append(
            f"Need {MIN_SAMPLE_SIZE - sample} more trades before optimization is valid."
        )
        return suggestions

    wr    = stats.get("win_rate", 0)
    avg_r = stats.get("avg_return", 0)
    pf    = stats.get("profit_factor", 1.0)
    dd    = abs(stats.get("max_drawdown", 0))

    if fitness >= STATUS_THRESHOLDS["strong"]:
        suggestions.append(f"Strategy is performing well (fitness {fitness:.0f}). Consider increasing allocation to {setup_type}.")
    elif fitness < STATUS_THRESHOLDS["weak"]:
        suggestions.append(f"Consider disabling {setup_type} until performance improves — fitness {fitness:.0f}/100.")

    if wr < 0.40:
        suggestions.append(f"Win rate {wr:.0%} is below 40%. Increase setup_quality_score threshold to filter lower-quality entries.")
    if wr > 0.65 and avg_r < 0.01:
        suggestions.append("High win rate but low average return — exits may be too early. Extend to T2 target more frequently.")

    if pf < 1.2:
        suggestions.append("Profit factor below 1.2 — risk/reward is unfavourable. Widen target or tighten stops.")
    if pf > 3.0:
        suggestions.append(f"Excellent profit factor ({pf:.1f}). Consider slightly increasing position size multiplier.")

    if dd > 0.15:
        suggestions.append(f"Max drawdown {dd:.0%} is high. Apply portfolio heat limits or reduce position multiplier in high-volatility regimes.")

    return suggestions or ["No optimization suggestions at this time."]


# ── Main Engine Class ─────────────────────────────────────────────────────────

class StrategyEvolutionEngine:
    """
    Analyzes strategy performance and generates optimization suggestions.
    Uses deterministic statistical analysis — no ML, fully auditable.
    """

    def analyze_strategy_performance(self) -> list[dict]:
        """Return per-strategy performance stats for all known setup types."""
        try:
            from database.models import TradeLifecycle

            with get_session() as session:
                # Get all closed trades grouped by setup_type
                setup_types = session.execute(
                    select(TradeLifecycle.setup_type)
                    .where(TradeLifecycle.state == "CLOSED")
                    .distinct()
                ).scalars().all()

                results = []
                for st in setup_types:
                    if st is None:
                        continue
                    trades = session.execute(
                        select(TradeLifecycle)
                        .where(
                            TradeLifecycle.setup_type == st,
                            TradeLifecycle.state      == "CLOSED",
                        )
                    ).scalars().all()

                    stats   = _compute_strategy_stats(trades)
                    fitness = _fitness_score(stats)
                    status  = _status_label(fitness)

                    results.append({
                        "setup_type":   st,
                        "fitness_score":fitness,
                        "status":       status,
                        **stats,
                    })

            # Sort by fitness descending
            results.sort(key=lambda x: x.get("fitness_score", 0), reverse=True)
            return results

        except Exception as e:
            logger.warning("StrategyEvolutionEngine.analyze_strategy_performance: %s", e)
            return []

    def calculate_strategy_fitness(self, setup_type: str) -> dict:
        """Compute fitness score for a specific strategy."""
        try:
            from database.models import TradeLifecycle

            with get_session() as session:
                trades = session.execute(
                    select(TradeLifecycle)
                    .where(
                        TradeLifecycle.setup_type == setup_type,
                        TradeLifecycle.state      == "CLOSED",
                    )
                ).scalars().all()

            stats   = _compute_strategy_stats(trades)
            fitness = _fitness_score(stats)
            return {
                "setup_type":    setup_type,
                "fitness_score": fitness,
                "status":        _status_label(fitness),
                "computed_at":   datetime.utcnow().isoformat() + "Z",
                **stats,
            }
        except Exception as e:
            return {"setup_type": setup_type, "error": str(e)}

    def identify_underperforming_strategies(self) -> list[dict]:
        """Return strategies with fitness score below acceptable threshold."""
        all_stats = self.analyze_strategy_performance()
        return [s for s in all_stats if s.get("fitness_score", 100) < STATUS_THRESHOLDS["acceptable"]]

    def generate_optimization_suggestions(self) -> list[dict]:
        """Generate suggestions for each strategy that has enough data."""
        all_stats = self.analyze_strategy_performance()
        output = []
        for s in all_stats:
            suggestions = _generate_suggestions(
                s["setup_type"], s, s.get("fitness_score", 0)
            )
            output.append({
                "setup_type":   s["setup_type"],
                "fitness_score":s.get("fitness_score"),
                "status":       s.get("status"),
                "suggestions":  suggestions,
            })
        return output

    def get_evolution_report(self) -> dict:
        """Full evolution report combining all analysis."""
        strategies    = self.analyze_strategy_performance()
        underperform  = [s for s in strategies if s.get("fitness_score", 100) < STATUS_THRESHOLDS["acceptable"]]
        suggestions   = self.generate_optimization_suggestions()

        system_fitness = (
            sum(s.get("fitness_score", 0) for s in strategies) / len(strategies)
            if strategies else None
        )

        return {
            "system_fitness":          round(system_fitness, 1) if system_fitness else None,
            "total_strategies":        len(strategies),
            "underperforming_count":   len(underperform),
            "strategies":              strategies,
            "suggestions":             suggestions,
            "safety_note":             f"Minimum {MIN_SAMPLE_SIZE} trades required before parameter changes are valid.",
            "generated_at":            datetime.utcnow().isoformat() + "Z",
        }
