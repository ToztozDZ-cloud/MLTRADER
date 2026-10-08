from __future__ import annotations

import random

from agents.agent import Agent
from agents.genome import (
    ConnectionGene,
    Genome,
    NodeGene,
)


def crossover(
    parent_a: Agent,
    parent_b: Agent,
) -> Agent:
    """
    NEAT-style topology-aware crossover.

    The fitter parent contributes all of its excess/disjoint
    connection genes. Matching innovation genes are inherited
    randomly from either parent.

    Node genes are merged by their globally stable node IDs.
    """

    if parent_b.fitness > parent_a.fitness:
        parent_a, parent_b = (
            parent_b,
            parent_a,
        )

    genes_a = {
        gene.innovation: gene
        for gene in parent_a.genome.connections
    }

    genes_b = {
        gene.innovation: gene
        for gene in parent_b.genome.connections
    }

    child_connections: list[
        ConnectionGene
    ] = []

    innovations = sorted(
        set(genes_a) | set(genes_b)
    )

    for innovation in innovations:
        gene_a = genes_a.get(
            innovation
        )

        gene_b = genes_b.get(
            innovation
        )

        if (
            gene_a is not None
            and gene_b is not None
        ):
            selected = random.choice(
                [
                    gene_a,
                    gene_b,
                ]
            )

            enabled = selected.enabled

            # If either matching gene is disabled,
            # occasionally preserve that disabled state.
            if (
                not gene_a.enabled
                or not gene_b.enabled
            ):
                if random.random() < 0.75:
                    enabled = False

        elif gene_a is not None:
            # Excess/disjoint genes from the fitter
            # parent are retained.
            selected = gene_a
            enabled = selected.enabled

        else:
            # Genes unique to the less-fit parent
            # are not inherited.
            continue

        child_connections.append(
            ConnectionGene(
                innovation=selected.innovation,
                source=selected.source,
                target=selected.target,
                weight=selected.weight,
                enabled=enabled,
            )
        )

    # Collect every node referenced by the selected
    # connection genes.
    required_node_ids = {
        connection.source
        for connection in child_connections
    }

    required_node_ids.update(
        connection.target
        for connection in child_connections
    )

    # Inputs, bias and outputs are always required.
    for node in parent_a.genome.nodes:
        if node.node_type in (
            "input",
            "bias",
            "output",
        ):
            required_node_ids.add(
                node.node_id
            )

    nodes_by_id: dict[
        int,
        NodeGene,
    ] = {}

    for node in parent_a.genome.nodes:
        if node.node_id in required_node_ids:
            nodes_by_id[node.node_id] = (
                NodeGene(
                    node_id=node.node_id,
                    node_type=node.node_type,
                    activation=node.activation,
                )
            )

    for node in parent_b.genome.nodes:
        if node.node_id not in required_node_ids:
            continue

        if node.node_id not in nodes_by_id:
            nodes_by_id[node.node_id] = (
                NodeGene(
                    node_id=node.node_id,
                    node_type=node.node_type,
                    activation=node.activation,
                )
            )

    child_nodes = sorted(
        nodes_by_id.values(),
        key=lambda node: node.node_id,
    )

    valid_node_ids = {
        node.node_id
        for node in child_nodes
    }

    child_connections = [
        connection
        for connection in child_connections
        if (
            connection.source
            in valid_node_ids
            and connection.target
            in valid_node_ids
        )
    ]

    child_genome = Genome(
        input_size=parent_a.genome.input_size,
        output_size=parent_a.genome.output_size,
        nodes=child_nodes,
        connections=child_connections,
    )

    child = Agent(
        genome=child_genome,
        hidden_size=parent_a.hidden_size,
        memory_size=parent_a.memory_size,
    )

    return child