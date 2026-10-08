from __future__ import annotations

import random

from agents.agent import Agent

from .crossover import crossover
from .mutation import mutate_agent, random_immigrant
from .population import Population
from .selection import (
    elitist_selection,
    tournament_selection,
)
from .speciation import Speciation


class EvolutionEngine:
    def __init__(
        self,
        population: Population,
        elitism: float = 0.10,
        survival_rate: float = 0.20,
        mutation_rate: float = 0.30,
        mutation_sigma: float = 0.80,
        crossover_rate: float = 0.75,
        immigrant_rate: float = 0.10,
        compatibility_threshold: float = 3.0,
        add_node_rate: float = 0.08,
        add_connection_rate: float = 0.15,
        remove_node_rate: float = 0.03,
        remove_connection_rate: float = 0.03,
    ) -> None:
        self.population = population

        self.elitism = elitism
        self.survival_rate = survival_rate
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.crossover_rate = crossover_rate
        self.immigrant_rate = immigrant_rate

        self.add_node_rate = add_node_rate
        self.add_connection_rate = add_connection_rate
        self.remove_node_rate = remove_node_rate
        self.remove_connection_rate = remove_connection_rate

        self.speciation = Speciation(
            compatibility_threshold=
            compatibility_threshold,
        )

        self.history = []

    def evolve(self) -> Agent:
        self.population.sort()

        champion = self.population.best().clone()

        self.history.append(
            {
                "generation":
                    self.population.generation,
                "best_fitness":
                    champion.fitness,
            }
        )

        elites = elitist_selection(
            self.population.agents,
            self.elitism,
        )

        survivors = elitist_selection(
            self.population.agents,
            self.survival_rate,
        )

        next_generation = list(elites)

        while len(next_generation) < self.population.size:
            # Inject a genuinely diverse immigrant.
            if random.random() < self.immigrant_rate:
                base = random.choice(
                    survivors
                )

                child = random_immigrant(
                    base
                )

            else:
                parent_a = tournament_selection(
                    survivors
                )

                parent_b = tournament_selection(
                    survivors
                )

                if random.random() < self.crossover_rate:
                    child = crossover(
                        parent_a,
                        parent_b,
                    )
                else:
                    child = parent_a.clone()

                child = mutate_agent(
                    child,
                    weight_rate=self.mutation_rate,
                    weight_sigma=self.mutation_sigma,
                    add_node_rate=self.add_node_rate,
                    add_connection_rate=
                        self.add_connection_rate,
                    remove_node_rate=
                        self.remove_node_rate,
                    remove_connection_rate=
                        self.remove_connection_rate,
                )

            next_generation.append(
                child
            )

        self.population.agents = (
            next_generation
        )

        self.population.generation += 1

        self.speciation.classify(
            self.population.agents
        )

        return champion