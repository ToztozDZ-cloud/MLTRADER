from __future__ import annotations

import numpy as np


class AgentMemory:
    """
    Persistent recurrent state for an agent.

    Memory is deliberately kept separate from the market environment.
    """

    def __init__(self, size: int = 0) -> None:
        self.size = max(0, int(size))
        self.state = np.zeros(self.size, dtype=np.float32)

    def reset(self) -> None:
        self.state.fill(0.0)

    def get(self) -> np.ndarray:
        return self.state.copy()

    def set(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float32)

        if len(state) != self.size:
            raise ValueError(
                f"Expected memory size {self.size}, "
                f"got {len(state)}."
            )

        self.state = state.copy()