
from __future__ import annotations

import random

from agents.agent import Agent


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
    """
    Create an aggressively mutated child.

    Designed for evolutionary runs that have stagnated for many
    generations. The mutation combines normal genome mutation with
    additional direct mutations and occasional large weight jumps.
    """

    mutated = agent.clone()
    genome = mutated.genome

    # ---------------------------------------------------------
    # 1. Normal genome mutation
    # ---------------------------------------------------------

    genome.mutate(
        weight_rate=weight_rate,
        weight_sigma=weight_sigma,
        connection_rate=connection_rate,
        add_node_rate=add_node_rate,
        add_connection_rate=add_connection_rate,
        remove_node_rate=remove_node_rate,
        remove_connection_rate=remove_connection_rate,
    )

    # ---------------------------------------------------------
    # 2. Additional aggressive weight mutations
    # ---------------------------------------------------------

    if genome.connections:

        # Mutate several connections instead of relying only on
        # genome.mutate().
        num_mutations = random.randint(
            1,
            max(2, int(len(genome.connections) * 0.40)),
        )

        for _ in range(num_mutations):

            connection = random.choice(
                genome.connections
            )

            # Most mutations are normal-sized.
            if random.random() < 0.85:
                connection.weight += random.gauss(
                    0.0,
                    weight_sigma,
                )

            # 15% are large evolutionary jumps.
            else:
                connection.weight += random.gauss(
                    0.0,
                    weight_sigma * 3.0,
                )

            # Prevent pathological weights.
            connection.weight = max(
                -8.0,
                min(8.0, connection.weight),
            )

    # ---------------------------------------------------------
    # 3. Occasionally perform multiple structural mutations
    # ---------------------------------------------------------

    if random.random() < 0.45:

        extra_mutations = random.randint(1, 3)

        for _ in range(extra_mutations):

            genome.mutate(
                weight_rate=0.25,
                weight_sigma=1.25,
                connection_rate=0.40,
                add_node_rate=0.25,
                add_connection_rate=0.40,
                remove_node_rate=0.10,
                remove_connection_rate=0.10,
            )

    # ---------------------------------------------------------
    # 4. Force at least one meaningful mutation
    # ---------------------------------------------------------

    if genome.connections:

        connection = random.choice(
            genome.connections
        )

        connection.weight += random.gauss(
            0.0,
            weight_sigma * 1.5,
        )

        connection.weight = max(
            -8.0,
            min(8.0, connection.weight),
        )

    # ---------------------------------------------------------
    # 5. Recalculate fitness
    # ---------------------------------------------------------

    mutated.fitness = float("-inf")

    return mutated


def random_immigrant(
    agent: Agent,
) -> Agent:
    """
    Create an extremely diverse immigrant.

    Existing topology is retained as a starting point, but weights
    are heavily randomized and several structural mutations are
    applied.
    """

    immigrant = agent.clone()
    genome = immigrant.genome

    # ---------------------------------------------------------
    # 1. Completely randomize existing weights
    # ---------------------------------------------------------

    for connection in genome.connections:

        connection.weight = random.gauss(
            0.0,
            2.0,
        )

        connection.weight = max(
            -8.0,
            min(8.0, connection.weight),
        )

        connection.enabled = True

    # ---------------------------------------------------------
    # 2. Aggressive structural mutation
    # ---------------------------------------------------------

    genome.mutate(
        weight_rate=0.90,
        weight_sigma=2.0,
        connection_rate=0.50,
        add_node_rate=0.30,
        add_connection_rate=0.50,
        remove_node_rate=0.10,
        remove_connection_rate=0.10,
    )

    # ---------------------------------------------------------
    # 3. Additional structural exploration
    # ---------------------------------------------------------

    if random.random() < 0.75:

        for _ in range(random.randint(1, 4)):

            genome.mutate(
                weight_rate=0.50,
                weight_sigma=1.75,
                connection_rate=0.50,
                add_node_rate=0.30,
                add_connection_rate=0.50,
                remove_node_rate=0.10,
                remove_connection_rate=0.10,
            )

    # ---------------------------------------------------------
    # 4. Guarantee usable random weights
    # ---------------------------------------------------------

    if genome.connections:

        for connection in genome.connections:

            if random.random() < 0.70:

                connection.weight = random.gauss(
                    0.0,
                    2.0,
                )

                connection.weight = max(
                    -8.0,
                    min(8.0, connection.weight),
                )

                connection.enabled = True

    immigrant.fitness = float("-inf")

    return immigrant