from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from agents.agent import Agent


@dataclass
class Species:
    representative: Agent
    members: List[Agent] = field(default_factory=list)

    @property
    def best_fitness(self) -> float:
        if not self.members:
            return float("-inf")

        return max(
            agent.fitness
            for agent in self.members
        )


class Speciation:
    def __init__(
        self,
        compatibility_threshold: float = 3.0,
    ) -> None:
        self.compatibility_threshold = compatibility_threshold
        self.species: List[Species] = []

    def distance(
        self,
        genome_a,
        genome_b,
    ) -> float:
        genes_a = {
            gene.innovation: gene
            for gene in genome_a.connections
        }

        genes_b = {
            gene.innovation: gene
            for gene in genome_b.connections
        }

        innovations = set(genes_a) | set(genes_b)

        if not innovations:
            return 0.0

        disjoint = 0
        weight_difference = 0.0
        matching = 0

        for innovation in innovations:
            a = genes_a.get(innovation)
            b = genes_b.get(innovation)

            if a is None or b is None:
                disjoint += 1
                continue

            matching += 1
            weight_difference += abs(
                a.weight - b.weight
            )

        average_weight_difference = (
            weight_difference / matching
            if matching
            else 0.0
        )

        return (
            disjoint
            + average_weight_difference
        )

    def classify(
        self,
        agents: List[Agent],
    ) -> List[Species]:
        self.species = []

        for agent in agents:
            assigned = False

            for species in self.species:
                distance = self.distance(
                    agent.genome,
                    species.representative.genome,
                )

                if distance <= self.compatibility_threshold:
                    species.members.append(agent)
                    assigned = True
                    break

            if not assigned:
                self.species.append(
                    Species(
                        representative=agent,
                        members=[agent],
                    )
                )

        return self.species