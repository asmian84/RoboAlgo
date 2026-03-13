"""Main evolution loop: evolve strategy genomes over multiple generations."""

from __future__ import annotations

import logging
from typing import Any

from strategy_evolution_engine.genome import StrategyGenomeParams
from strategy_evolution_engine.fitness import compute_fitness, evaluate_genome_simple
from strategy_evolution_engine.population import (
    generate_initial_population,
    select_parents,
    create_next_generation,
)

logger = logging.getLogger("strategy_evolution_engine.evolver")


class StrategyEvolver:
    """Genetic algorithm engine for evolving trading strategies."""

    def __init__(
        self,
        population_size: int = 50,
        n_generations: int = 20,
        mutation_rate: float = 0.2,
        elite_count: int = 5,
    ):
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count

    def evolve(self, price_data: dict[str, list]) -> list[dict[str, Any]]:
        """Run the full evolution loop.

        Args:
            price_data: Dict mapping symbol -> list of {date, close, atr} dicts.

        Returns:
            List of top strategy genomes with their fitness metrics.
        """
        population = generate_initial_population(self.population_size)

        best_ever: list[dict[str, Any]] = []

        for gen in range(self.n_generations):
            # Evaluate fitness for each genome
            fitness_scores: list[float] = []
            genome_metrics: list[dict] = []

            for genome in population:
                metrics = evaluate_genome_simple(genome.to_dict(), price_data)
                fitness = compute_fitness(metrics)
                fitness_scores.append(fitness)
                genome_metrics.append(metrics)

            # Track best
            best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            best_fitness = fitness_scores[best_idx]
            best_genome = population[best_idx]

            logger.info(
                "Generation %d: best_fitness=%.4f  genome=%s",
                gen, best_fitness, best_genome.genome_id,
            )

            # Store top performers
            paired = sorted(
                zip(population, fitness_scores, genome_metrics),
                key=lambda x: x[1],
                reverse=True,
            )
            for genome, fitness, metrics in paired[:self.elite_count]:
                best_ever.append({
                    "generation": gen,
                    "genome": genome.to_dict(),
                    "fitness": fitness,
                    "metrics": metrics,
                })

            # Select parents and create next generation
            parents = select_parents(population, fitness_scores)
            population = create_next_generation(
                parents, self.population_size, self.mutation_rate, self.elite_count,
            )

        # Deduplicate and sort by fitness
        seen_ids = set()
        unique_results: list[dict] = []
        for entry in sorted(best_ever, key=lambda x: x["fitness"], reverse=True):
            gid = entry["genome"]["genome_id"]
            if gid not in seen_ids:
                seen_ids.add(gid)
                unique_results.append(entry)

        return unique_results[:20]  # top 20 strategies
