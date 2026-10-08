from __future__ import annotations

import json
from pathlib import Path

from .agent import Agent
from .genome import (
    ConnectionGene,
    Genome,
    NodeGene,
)


class AgentSerializer:
    @staticmethod
    def save(
        agent: Agent,
        path: str | Path,
    ) -> None:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "version": 2,
            "agent": {
                "input_size":
                    agent.genome.input_size,
                "output_size":
                    agent.genome.output_size,
                "hidden_size":
                    agent.hidden_size,
                "memory_size":
                    agent.memory_size,
                "fitness":
                    agent.fitness,
            },
            "genome": {
                "nodes": [
                    {
                        "node_id":
                            node.node_id,
                        "node_type":
                            node.node_type,
                        "activation":
                            node.activation,
                    }
                    for node
                    in agent.genome.nodes
                ],
                "connections": [
                    {
                        "innovation":
                            connection.innovation,
                        "source":
                            connection.source,
                        "target":
                            connection.target,
                        "weight":
                            connection.weight,
                        "enabled":
                            connection.enabled,
                    }
                    for connection
                    in agent.genome.connections
                ],
            },
        }

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
            )

    @staticmethod
    def load(
        path: str | Path,
    ) -> Agent:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Agent model not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        version = data.get(
            "version",
            1,
        )

        if version < 2:
            raise ValueError(
                "This model uses the old neural-network "
                "format. Retrain the model with the "
                "current MLTRADER architecture."
            )

        agent_data = data["agent"]
        genome_data = data["genome"]

        nodes = [
            NodeGene(
                node_id=node["node_id"],
                node_type=node["node_type"],
                activation=node.get(
                    "activation",
                    "tanh",
                ),
            )
            for node
            in genome_data["nodes"]
        ]

        connections = [
            ConnectionGene(
                innovation=connection[
                    "innovation"
                ],
                source=connection["source"],
                target=connection["target"],
                weight=connection["weight"],
                enabled=connection.get(
                    "enabled",
                    True,
                ),
            )
            for connection
            in genome_data["connections"]
        ]

        genome = Genome(
            input_size=agent_data[
                "input_size"
            ],
            output_size=agent_data[
                "output_size"
            ],
            nodes=nodes,
            connections=connections,
            fitness=agent_data.get(
                "fitness",
                float("-inf"),
            ),
        )

        agent = Agent(
            genome=genome,
            hidden_size=agent_data.get(
                "hidden_size",
                64,
            ),
            memory_size=agent_data.get(
                "memory_size",
                0,
            ),
        )

        agent.fitness = agent_data.get(
            "fitness",
            float("-inf"),
        )

        return agent