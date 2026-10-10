from __future__ import annotations

import random

from agents.agent import Agent


def _clip_weights(agent: Agent, limit: float = 8.0) -> None:
    for connection in agent.genome.connections:
        connection.weight = max(-limit, min(limit, connection.weight))


def mutate_agent(
    agent: Agent,
    weight_rate: float = 0.60,
    weight_sigma: float = 1.00,
    connection_rate: float = 0.30,
    add_node_rate: float = 0.15,
    add_connection_rate: float = 0.30,
    remove_node_rate: float = 0.08,
    remove_connection_rate: float = 0.08,
) -> Agent:
    """Aggressively explore weights/topology without repeatedly wrecking children."""
    child = agent.clone()
    child.genome.mutate(
        weight_rate=weight_rate,
        weight_sigma=weight_sigma,
        connection_rate=connection_rate,
        add_node_rate=add_node_rate,
        add_connection_rate=add_connection_rate,
        remove_node_rate=remove_node_rate,
        remove_connection_rate=remove_connection_rate,
    )

    # Occasional second pass gives a subset of children a large exploratory jump.
    if random.random() < 0.35:
        child.genome.mutate(
            weight_rate=min(1.0, max(0.35, weight_rate)),
            weight_sigma=weight_sigma * 1.5,
            connection_rate=min(1.0, connection_rate * 1.25),
            add_node_rate=min(1.0, add_node_rate * 1.5),
            add_connection_rate=min(1.0, add_connection_rate * 1.5),
            remove_node_rate=remove_node_rate,
            remove_connection_rate=remove_connection_rate,
        )

    if child.genome.connections and random.random() < 0.25:
        gene = random.choice(child.genome.connections)
        gene.weight += random.gauss(0.0, weight_sigma * 2.5)

    _clip_weights(child)
    child.fitness = float("-inf")
    return child


def random_immigrant(agent: Agent) -> Agent:
    """Inject a strongly perturbed genome to restore population diversity."""
    immigrant = agent.clone()
    for connection in immigrant.genome.connections:
        connection.weight = random.gauss(0.0, 2.0)
        connection.enabled = True

    for _ in range(random.randint(2, 4)):
        immigrant.genome.mutate(
            weight_rate=0.80,
            weight_sigma=1.75,
            connection_rate=0.30,
            add_node_rate=0.35,
            add_connection_rate=0.50,
            remove_node_rate=0.05,
            remove_connection_rate=0.05,
        )

    _clip_weights(immigrant)
    immigrant.fitness = float("-inf")
    return immigrant
