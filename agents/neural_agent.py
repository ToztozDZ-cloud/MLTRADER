from __future__ import annotations

import math

import numpy as np


def activation(
    name: str,
    value: float,
) -> float:
    if name == "tanh":
        return math.tanh(value)

    if name == "sigmoid":
        value = max(
            -60.0,
            min(60.0, value),
        )

        return 1.0 / (
            1.0 + math.exp(-value)
        )

    if name == "relu":
        return max(
            0.0,
            value,
        )

    if name == "linear":
        return value

    raise ValueError(
        f"Unknown activation: {name}"
    )


class NeuralAgent:
    """
    Executes the directed acyclic graph encoded by a Genome.

    The genome controls:
    - weights
    - enabled connections
    - hidden nodes
    - topology
    - activation functions
    """

    def __init__(self, genome):
        self.genome = genome

    def _topological_order(self) -> list[int]:
        nodes = {
            node.node_id: node
            for node in self.genome.nodes
        }

        indegree = {
            node_id: 0
            for node_id in nodes
        }

        adjacency: dict[
            int,
            list[int],
        ] = {
            node_id: []
            for node_id in nodes
        }

        for connection in (
            self.genome.connections
        ):
            if not connection.enabled:
                continue

            if (
                connection.source
                not in nodes
            ):
                continue

            if (
                connection.target
                not in nodes
            ):
                continue

            adjacency[
                connection.source
            ].append(
                connection.target
            )

            indegree[
                connection.target
            ] += 1

        queue = [
            node_id
            for node_id, degree
            in indegree.items()
            if degree == 0
        ]

        queue.sort()

        order: list[int] = []

        while queue:
            current = queue.pop(0)

            order.append(current)

            for target in adjacency[
                current
            ]:
                indegree[target] -= 1

                if indegree[target] == 0:
                    queue.append(target)

            queue.sort()

        if len(order) != len(nodes):
            raise RuntimeError(
                "Genome contains a directed cycle. "
                "Cycle prevention in mutation failed."
            )

        return order

    def act(
        self,
        observation,
    ) -> float:
        observation = np.asarray(
            observation,
            dtype=np.float64,
        )

        if len(observation) != (
            self.genome.input_size
        ):
            raise ValueError(
                f"Expected "
                f"{self.genome.input_size} "
                f"inputs, received "
                f"{len(observation)}."
            )

        nodes = {
            node.node_id: node
            for node in self.genome.nodes
        }

        input_nodes = sorted(
            [
                node
                for node in self.genome.nodes
                if node.node_type == "input"
            ],
            key=lambda node: node.node_id,
        )

        if len(input_nodes) != (
            self.genome.input_size
        ):
            raise RuntimeError(
                "Genome input node count does not "
                "match genome.input_size."
            )

        values: dict[
            int,
            float,
        ] = {}

        for index, node in enumerate(
            input_nodes
        ):
            values[node.node_id] = float(
                observation[index]
            )

        for node in self.genome.nodes:
            if node.node_type == "bias":
                values[node.node_id] = 1.0

        incoming: dict[
            int,
            list,
        ] = {}

        for connection in (
            self.genome.connections
        ):
            if not connection.enabled:
                continue

            if (
                connection.source
                not in nodes
                or connection.target
                not in nodes
            ):
                continue

            incoming.setdefault(
                connection.target,
                [],
            ).append(connection)

        order = self._topological_order()

        for node_id in order:
            node = nodes[node_id]

            if node.node_type in (
                "input",
                "bias",
            ):
                continue

            connections = incoming.get(
                node_id,
                [],
            )

            total = 0.0

            for connection in connections:
                source_value = values.get(
                    connection.source,
                    0.0,
                )

                total += (
                    source_value
                    * connection.weight
                )

            values[node_id] = activation(
                node.activation,
                total,
            )

        output_nodes = sorted(
            [
                node
                for node in self.genome.nodes
                if node.node_type == "output"
            ],
            key=lambda node: node.node_id,
        )

        if not output_nodes:
            return 0.0

        output = values.get(
            output_nodes[0].node_id,
            0.0,
        )

        return float(
            np.clip(
                output,
                -1.0,
                1.0,
            )
        )

    def state_dict(self) -> dict:
        return {
            "genome": self.genome,
        }

    def load_state_dict(
        self,
        state: dict,
    ) -> None:
        self.genome = state["genome"]