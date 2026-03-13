"""Population management for genetic algorithm."""

from __future__ import annotations

import random

from strategy_evolution_engine.genome import StrategyGenomeParams
from strategy_evolution_engine.mutation import mutate, crossover


def generate_initial_population(size: int = 50) -> list[StrategyGenomeParams]:
    """Generate a random initial population of strategy genomes."""
    population: list[StrategyGenomeParams] = []

    regime_options = ["ALL", "COMPRESSION", "TREND", "EXPANSION"]
    pattern_options = [
        "any", "compression_breakout", "trend_pullback",
        "breakout_momentum", "chair_pattern",
    ]

    for _ in range(size):
        genome = StrategyGenomeParams(
            entry_confluence_min=round(random.uniform(40, 85), 1),
            pattern_type=random.choice(pattern_options),
            regime_filter=random.choice(regime_options),
            stop_atr_mult=round(random.uniform(0.8, 4.0), 1),
            target_atr_mult=round(random.uniform(2.0, 8.0), 1),
            hold_days_max=random.randint(5, 40),
            position_size_pct=round(random.uniform(1.0, 4.0), 1),
            max_positions=random.randint(2, 8),
        )
        population.append(genome)

    return population


def select_parents(
    population: list[StrategyGenomeParams],
    fitness_scores: list[float],
    n_parents: int = 10,
) -> list[StrategyGenomeParams]:
    """Tournament selection: pick top performers."""
    paired = list(zip(population, fitness_scores))
    paired.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in paired[:n_parents]]


def create_next_generation(
    parents: list[StrategyGenomeParams],
    population_size: int = 50,
    mutation_rate: float = 0.2,
    elite_count: int = 5,
) -> list[StrategyGenomeParams]:
    """Create next generation from selected parents.

    Preserves top elite_count unchanged, fills rest with crossover + mutation.
    """
    next_gen: list[StrategyGenomeParams] = []

    # Elitism: keep top parents unchanged
    for parent in parents[:elite_count]:
        next_gen.append(parent)

    # Fill remaining slots with crossover + mutation
    while len(next_gen) < population_size:
        p1, p2 = random.sample(parents, min(2, len(parents)))
        child = crossover(p1, p2)
        child = mutate(child, mutation_rate)
        next_gen.append(child)

    return next_gen[:population_size]
