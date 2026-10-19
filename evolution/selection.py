from __future__ import annotations

import random
import math
from typing import List

from agents.agent import Agent


def elitist_selection(
    agents: List[Agent],
    fraction: float,
) -> List[Agent]:
    if not agents:
        return []
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("selection fraction must be in [0, 1]")
    if fraction == 0.0:
        return []

    valid = [agent for agent in agents if math.isfinite(float(agent.fitness))]
    if not valid:
        raise ValueError("Cannot select from a population without finite fitness")

    count = max(
        1,
        int(len(valid) * fraction),
    )

    ranked = sorted(
        valid,
        key=lambda agent: agent.fitness,
        reverse=True,
    )

    return [
        agent.clone()
        for agent in ranked[:count]
    ]


def tournament_selection(
    agents: List[Agent],
    tournament_size: int = 3,
) -> Agent:
    if not agents:
        raise ValueError("Cannot select from an empty population.")
    if tournament_size < 1:
        raise ValueError("tournament_size must be positive")

    valid = [agent for agent in agents if math.isfinite(float(agent.fitness))]
    if not valid:
        raise ValueError("Cannot select without finite fitness values")

    tournament_size = min(
        tournament_size,
        len(valid),
    )

    candidates = random.sample(
        valid,
        tournament_size,
    )

    return max(
        candidates,
        key=lambda agent: agent.fitness,
    )
