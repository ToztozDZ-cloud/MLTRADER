from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .genome import Genome
from .memory import AgentMemory
from .neural_agent import NeuralAgent


@dataclass
class Agent:
    genome: Genome
    hidden_size: int = 64
    memory_size: int = 0

    neural_agent: NeuralAgent = field(init=False)
    memory: AgentMemory = field(init=False)

    fitness: float = float("-inf")

    def __post_init__(self) -> None:
        self.neural_agent = NeuralAgent(
            genome=self.genome,
        )

        self.memory = AgentMemory(
            size=self.memory_size,
        )

    def reset(self) -> None:
        self.memory.reset()

    def act(self, observation: np.ndarray) -> float:
        return self.neural_agent.act(
            observation
        )

    def clone(self) -> "Agent":
        cloned = Agent(
            genome=self.genome.clone(),
            hidden_size=self.hidden_size,
            memory_size=self.memory_size,
        )

        cloned.fitness = self.fitness

        return cloned