from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy

import numpy as np

from .genome import ConnectionGene, Genome, NodeGene, innovation_registry
from .memory import AgentMemory
from .neural_agent import NeuralAgent


@dataclass
class Agent:
    genome: Genome
    hidden_size: int = 0
    memory_size: int = 0
    metadata: dict = field(default_factory=dict)

    neural_agent: NeuralAgent = field(init=False)
    memory: AgentMemory = field(init=False)

    fitness: float = float("-inf")

    def __post_init__(self) -> None:
        self.hidden_size = max(0, int(self.hidden_size))
        self.memory_size = max(0, int(self.memory_size))
        if self.hidden_size:
            self._ensure_hidden_nodes()
        if self.memory_size:
            self._ensure_memory_nodes()

        self.neural_agent = NeuralAgent(
            genome=self.genome,
        )

        self.memory = AgentMemory(
            size=self.memory_size,
        )

    def _ensure_hidden_nodes(self) -> None:
        hidden_nodes = [node for node in self.genome.nodes if node.node_type == "hidden"]
        missing = self.hidden_size - len(hidden_nodes)
        if missing <= 0:
            return

        next_id = max((node.node_id for node in self.genome.nodes), default=-1) + 1
        inputs = [node for node in self.genome.nodes if node.node_type == "input"]
        outputs = [node for node in self.genome.nodes if node.node_type == "output"]
        hidden_ids: list[int] = []
        for offset in range(missing):
            node_id = next_id + offset
            hidden_ids.append(node_id)
            self.genome.nodes.append(NodeGene(node_id=node_id, node_type="hidden", activation="tanh"))

        for input_node in inputs:
            for hidden_id in hidden_ids:
                innovation = innovation_registry.connection_innovation(input_node.node_id, hidden_id)
                self.genome.connections.append(ConnectionGene(innovation, input_node.node_id, hidden_id, 0.0))
        for hidden_id in hidden_ids:
            for output_node in outputs:
                innovation = innovation_registry.connection_innovation(hidden_id, output_node.node_id)
                self.genome.connections.append(ConnectionGene(innovation, hidden_id, output_node.node_id, 0.0))

    def _ensure_memory_nodes(self) -> None:
        memory_nodes = [node for node in self.genome.nodes if node.node_type == "memory"]
        missing = self.memory_size - len(memory_nodes)
        if missing <= 0:
            return

        next_id = max((node.node_id for node in self.genome.nodes), default=-1) + 1
        outputs = [node for node in self.genome.nodes if node.node_type == "output"]
        memory_ids: list[int] = []
        for offset in range(missing):
            node_id = next_id + offset
            memory_ids.append(node_id)
            self.genome.nodes.append(NodeGene(node_id=node_id, node_type="memory", activation="linear"))
        for memory_id in memory_ids:
            for output_node in outputs:
                innovation = innovation_registry.connection_innovation(memory_id, output_node.node_id)
                self.genome.connections.append(ConnectionGene(innovation, memory_id, output_node.node_id, 0.0))
        self.genome.input_size += missing

    def reset(self) -> None:
        self.memory.reset()

    def act(self, observation: np.ndarray) -> float:
        observation = np.asarray(observation, dtype=np.float64)
        if self.memory_size:
            combined = np.concatenate((observation, self.memory.get().astype(np.float64)))
            action = self.neural_agent.act(combined)
            previous = self.memory.get()
            if self.memory_size > 1:
                updated = np.concatenate((previous[1:], np.asarray([action], dtype=np.float32)))
            else:
                updated = np.asarray([action], dtype=np.float32)
            self.memory.set(updated)
            return action
        return self.neural_agent.act(observation)

    def clone(self) -> "Agent":
        cloned = Agent(
            genome=self.genome.clone(),
            hidden_size=self.hidden_size,
            memory_size=self.memory_size,
            metadata=deepcopy(self.metadata),
        )

        cloned.fitness = self.fitness

        return cloned
