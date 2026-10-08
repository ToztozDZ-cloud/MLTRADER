from __future__ import annotations

import random

from agents.agent import Agent
from agents.genome import (
    ConnectionGene,
    Genome,
    NodeGene,
    innovation_registry,
)


class Population:
    def __init__(
        self,
        size: int,
        input_size: int,
        output_size: int,
        hidden_size: int = 64,
    ) -> None:
        if size <= 0:
            raise ValueError(
                "Population size must be greater than zero."
            )

        if input_size <= 0:
            raise ValueError(
                "Input size must be greater than zero."
            )

        if output_size <= 0:
            raise ValueError(
                "Output size must be greater than zero."
            )

        self.size = size
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.agents = [
            self._create_agent()
            for _ in range(size)
        ]

        self.generation = 0

    def _create_agent(self) -> Agent:
        nodes: list[NodeGene] = []

        for node_id in range(
            self.input_size
        ):
            nodes.append(
                NodeGene(
                    node_id=node_id,
                    node_type="input",
                    activation="linear",
                )
            )

        bias_id = self.input_size

        nodes.append(
            NodeGene(
                node_id=bias_id,
                node_type="bias",
                activation="linear",
            )
        )

        output_start = (
            self.input_size + 1
        )

        for index in range(
            self.output_size
        ):
            nodes.append(
                NodeGene(
                    node_id=(
                        output_start
                        + index
                    ),
                    node_type="output",
                    activation="tanh",
                )
            )

        connections: list[
            ConnectionGene
        ] = []

        for input_id in range(
            self.input_size
        ):
            for output_index in range(
                self.output_size
            ):
                output_id = (
                    output_start
                    + output_index
                )

                innovation = (
                    innovation_registry
                    .connection_innovation(
                        source=input_id,
                        target=output_id,
                    )
                )

                connections.append(
                    ConnectionGene(
                        innovation=innovation,
                        source=input_id,
                        target=output_id,
                        weight=random.gauss(
                            0.0,
                            0.5,
                        ),
                        enabled=True,
                    )
                )

        for output_index in range(
            self.output_size
        ):
            output_id = (
                output_start
                + output_index
            )

            innovation = (
                innovation_registry
                .connection_innovation(
                    source=bias_id,
                    target=output_id,
                )
            )

            connections.append(
                ConnectionGene(
                    innovation=innovation,
                    source=bias_id,
                    target=output_id,
                    weight=random.gauss(
                        0.0,
                        0.5,
                    ),
                    enabled=True,
                )
            )

        genome = Genome(
            input_size=self.input_size,
            output_size=self.output_size,
            nodes=nodes,
            connections=connections,
        )

        return Agent(
            genome=genome,
            hidden_size=self.hidden_size,
        )

    def best(self) -> Agent:
        return max(
            self.agents,
            key=lambda agent: agent.fitness,
        )

    def sort(self) -> None:
        self.agents.sort(
            key=lambda agent: agent.fitness,
            reverse=True,
        )

    def __len__(self) -> int:
        return len(self.agents)