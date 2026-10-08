from __future__ import annotations

import random
from typing import List

from agents.agent import Agent


def elitist_selection(
    agents: List[Agent],
    fraction: float,
) -> List[Agent]:
    if not agents:
        return []

    count = max(
        1,
        int(len(agents) * fraction),
    )

    ranked = sorted(
        agents,
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

    tournament_size = min(
        tournament_size,
        len(agents),
    )

    candidates = random.sample(
        agents,
        tournament_size,
    )

    return max(
        candidates,
        key=lambda agent: agent.fitness,
    )