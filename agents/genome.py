from __future__ import annotations

from dataclasses import dataclass, field
import copy
import random


class InnovationRegistry:
    """
    Global registry for homologous evolutionary genes.

    The same structural connection always receives the same
    innovation number during the lifetime of a training run.

    Node IDs created by splitting the same connection are also
    stable across the population.
    """

    def __init__(self) -> None:
        self._connection_innovations: dict[
            tuple[int, int],
            int,
        ] = {}

        self._split_nodes: dict[
            int,
            int,
        ] = {}

        self._next_innovation = 0
        self._next_node_id = 0

    def register_existing_connection(
        self,
        source: int,
        target: int,
        innovation: int,
    ) -> None:
        key = (source, target)

        self._connection_innovations[key] = innovation

        self._next_innovation = max(
            self._next_innovation,
            innovation + 1,
        )

    def register_existing_node(
        self,
        node_id: int,
    ) -> None:
        self._next_node_id = max(
            self._next_node_id,
            node_id + 1,
        )

    def connection_innovation(
        self,
        source: int,
        target: int,
    ) -> int:
        key = (source, target)

        if key not in self._connection_innovations:
            self._connection_innovations[key] = (
                self._next_innovation
            )

            self._next_innovation += 1

        return self._connection_innovations[key]

    def split_node(
        self,
        connection_innovation: int,
    ) -> int:
        if (
            connection_innovation
            not in self._split_nodes
        ):
            self._split_nodes[
                connection_innovation
            ] = self._next_node_id

            self._next_node_id += 1

        return self._split_nodes[
            connection_innovation
        ]


innovation_registry = InnovationRegistry()


@dataclass
class NodeGene:
    node_id: int
    node_type: str
    activation: str = "tanh"


@dataclass
class ConnectionGene:
    innovation: int
    source: int
    target: int
    weight: float
    enabled: bool = True


@dataclass
class Genome:
    input_size: int
    output_size: int

    nodes: list[NodeGene] = field(
        default_factory=list
    )

    connections: list[ConnectionGene] = field(
        default_factory=list
    )

    fitness: float = float("-inf")

    def __post_init__(self) -> None:
        for node in self.nodes:
            innovation_registry.register_existing_node(
                node.node_id
            )

        for connection in self.connections:
            innovation_registry.register_existing_connection(
                source=connection.source,
                target=connection.target,
                innovation=connection.innovation,
            )

    def clone(self) -> "Genome":
        return copy.deepcopy(self)

    def _connection_exists(
        self,
        source: int,
        target: int,
    ) -> bool:
        return any(
            connection.source == source
            and connection.target == target
            for connection in self.connections
        )

    def _node_exists(
        self,
        node_id: int,
    ) -> bool:
        return any(
            node.node_id == node_id
            for node in self.nodes
        )

    def _has_path(
        self,
        start: int,
        target: int,
    ) -> bool:
        """
        Return True if target is reachable from start.

        Used to prevent adding a connection that creates
        a directed cycle.
        """

        if start == target:
            return True

        adjacency: dict[int, list[int]] = {}

        for connection in self.connections:
            if not connection.enabled:
                continue

            adjacency.setdefault(
                connection.source,
                [],
            ).append(
                connection.target
            )

        stack = [start]
        visited: set[int] = set()

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)

            for next_node in adjacency.get(
                current,
                [],
            ):
                if next_node == target:
                    return True

                if next_node not in visited:
                    stack.append(next_node)

        return False

    def mutate_weights(
        self,
        rate: float = 0.10,
        sigma: float = 0.50,
    ) -> None:
        for connection in self.connections:
            if random.random() < rate:
                connection.weight += random.gauss(
                    0.0,
                    sigma,
                )

    def mutate_connections(
        self,
        rate: float = 0.05,
    ) -> None:
        for connection in self.connections:
            if random.random() < rate:
                connection.enabled = not connection.enabled

    def add_connection(
        self,
        attempts: int = 50,
    ) -> None:
        if len(self.nodes) < 2:
            return

        for _ in range(attempts):
            source = random.choice(self.nodes)
            target = random.choice(self.nodes)

            if source.node_id == target.node_id:
                continue

            if target.node_type in (
                "input",
                "bias",
            ):
                continue

            if source.node_type == "output":
                continue

            if self._connection_exists(
                source.node_id,
                target.node_id,
            ):
                continue

            # Adding source -> target would create a
            # cycle if target can already reach source.
            if self._has_path(
                target.node_id,
                source.node_id,
            ):
                continue

            innovation = (
                innovation_registry.connection_innovation(
                    source=source.node_id,
                    target=target.node_id,
                )
            )

            self.connections.append(
                ConnectionGene(
                    innovation=innovation,
                    source=source.node_id,
                    target=target.node_id,
                    weight=random.gauss(
                        0.0,
                        1.0,
                    ),
                    enabled=True,
                )
            )

            return

    def add_node(self) -> None:
        enabled_connections = [
            connection
            for connection in self.connections
            if connection.enabled
        ]

        if not enabled_connections:
            return

        connection = random.choice(
            enabled_connections
        )

        connection.enabled = False

        node_id = innovation_registry.split_node(
            connection.innovation
        )

        if not self._node_exists(node_id):
            self.nodes.append(
                NodeGene(
                    node_id=node_id,
                    node_type="hidden",
                    activation=random.choice(
                        [
                            "tanh",
                            "relu",
                            "sigmoid",
                        ]
                    ),
                )
            )

        first_innovation = (
            innovation_registry.connection_innovation(
                source=connection.source,
                target=node_id,
            )
        )

        second_innovation = (
            innovation_registry.connection_innovation(
                source=node_id,
                target=connection.target,
            )
        )

        if not self._connection_exists(
            connection.source,
            node_id,
        ):
            self.connections.append(
                ConnectionGene(
                    innovation=first_innovation,
                    source=connection.source,
                    target=node_id,
                    weight=1.0,
                    enabled=True,
                )
            )

        if not self._connection_exists(
            node_id,
            connection.target,
        ):
            self.connections.append(
                ConnectionGene(
                    innovation=second_innovation,
                    source=node_id,
                    target=connection.target,
                    weight=connection.weight,
                    enabled=True,
                )
            )

    def remove_connection(self) -> None:
        if not self.connections:
            return

        index = random.randrange(
            len(self.connections)
        )

        del self.connections[index]

    def remove_node(self) -> None:
        hidden_nodes = [
            node
            for node in self.nodes
            if node.node_type == "hidden"
        ]

        if not hidden_nodes:
            return

        node = random.choice(
            hidden_nodes
        )

        self.nodes.remove(node)

        self.connections = [
            connection
            for connection in self.connections
            if connection.source != node.node_id
            and connection.target != node.node_id
        ]

    def mutate(
        self,
        weight_rate: float = 0.10,
        weight_sigma: float = 0.50,
        connection_rate: float = 0.05,
        add_node_rate: float = 0.02,
        add_connection_rate: float = 0.05,
        remove_node_rate: float = 0.01,
        remove_connection_rate: float = 0.01,
    ) -> None:
        self.mutate_weights(
            rate=weight_rate,
            sigma=weight_sigma,
        )

        self.mutate_connections(
            rate=connection_rate,
        )

        if random.random() < add_node_rate:
            self.add_node()

        if random.random() < add_connection_rate:
            self.add_connection()

        if random.random() < remove_node_rate:
            self.remove_node()

        if random.random() < remove_connection_rate:
            self.remove_connection()