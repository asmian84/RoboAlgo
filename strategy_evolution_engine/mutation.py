"""Mutation operators for strategy genomes."""

from __future__ import annotations

import random
from copy import deepcopy

from strategy_evolution_engine.genome import StrategyGenomeParams

# Mutation ranges for numeric parameters
MUTATION_RANGES = {
    "entry_confluence_min": (30.0, 90.0),
    "stop_atr_mult": (0.5, 5.0),
    "target_atr_mult": (1.0, 10.0),
    "hold_days_max": (3, 60),
    "position_size_pct": (0.5, 5.0),
    "max_positions": (1, 10),
}

REGIME_OPTIONS = ["ALL", "COMPRESSION", "TREND", "EXPANSION", "CHAOS"]
PATTERN_OPTIONS = [
    "any", "compression_breakout", "trend_pullback", "breakout_momentum",
    "mean_reversion", "chair_pattern", "cup_handle",
]


def mutate(genome: StrategyGenomeParams, mutation_rate: float = 0.2) -> StrategyGenomeParams:
    """Mutate a genome by randomly perturbing parameters.

    Args:
        genome: The parent genome to mutate.
        mutation_rate: Probability of mutating each parameter (0-1).

    Returns:
        A new mutated genome.
    """
    child = deepcopy(genome)
    child.genome_id = f"{genome.genome_id[:4]}-m{random.randint(0, 999):03d}"

    for param, (lo, hi) in MUTATION_RANGES.items():
        if random.random() < mutation_rate:
            current = getattr(child, param)
            if isinstance(current, int):
                delta = random.randint(-max(1, int((hi - lo) * 0.15)), max(1, int((hi - lo) * 0.15)))
                new_val = max(int(lo), min(int(hi), int(current) + delta))
                setattr(child, param, new_val)
            else:
                delta = random.gauss(0, (hi - lo) * 0.1)
                new_val = max(lo, min(hi, current + delta))
                setattr(child, param, round(new_val, 2))

    # Categorical mutations
    if random.random() < mutation_rate:
        child.regime_filter = random.choice(REGIME_OPTIONS)
    if random.random() < mutation_rate:
        child.pattern_type = random.choice(PATTERN_OPTIONS)

    return child


def crossover(parent1: StrategyGenomeParams, parent2: StrategyGenomeParams) -> StrategyGenomeParams:
    """Single-point crossover between two parent genomes."""
    child = deepcopy(parent1)
    child.genome_id = f"{parent1.genome_id[:2]}x{parent2.genome_id[:2]}-{random.randint(0,999):03d}"

    params = list(MUTATION_RANGES.keys()) + ["regime_filter", "pattern_type"]
    crossover_point = random.randint(1, len(params) - 1)

    for param in params[crossover_point:]:
        setattr(child, param, getattr(parent2, param))

    return child
