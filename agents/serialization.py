from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile

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
        metadata: dict | None = None,
    ) -> None:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        AgentSerializer._validate_agent(agent)
        artifact_metadata = dict(agent.metadata)
        artifact_metadata.update(metadata or {})
        data = {
            "version": 3,
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
                    agent.fitness if math.isfinite(float(agent.fitness)) else None,
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
            "metadata": artifact_metadata,
        }
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2, allow_nan=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_name, path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise

    @staticmethod
    def load(
        path: str | Path,
        *,
        expected_observation: dict | None = None,
        expected_timeframe: str | None = None,
        require_metadata: bool = False,
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

        if not isinstance(data, dict):
            raise ValueError("Model artifact must contain a JSON object")
        try:
            version = int(data.get("version", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("Model artifact version must be an integer") from exc

        if version < 2:
            raise ValueError(
                "This model uses the old neural-network "
                "format. Retrain the model with the "
                "current MLTRADER architecture."
            )

        if version > 3:
            raise ValueError(f"Unsupported model artifact version: {version}")
        try:
            agent_data = data["agent"]
            genome_data = data["genome"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Model artifact is missing agent or genome data") from exc
        if not isinstance(agent_data, dict) or not isinstance(genome_data, dict):
            raise ValueError("Model agent and genome sections must be objects")

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

        stored_fitness = agent_data.get("fitness")
        fitness = (
            float(stored_fitness)
            if stored_fitness is not None and math.isfinite(float(stored_fitness))
            else float("-inf")
        )
        genome = Genome(
            input_size=agent_data[
                "input_size"
            ],
            output_size=agent_data[
                "output_size"
            ],
            nodes=nodes,
            connections=connections,
            fitness=fitness,
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
            metadata=data.get("metadata", {}) if isinstance(data.get("metadata", {}), dict) else {},
        )

        agent.fitness = fitness

        AgentSerializer._validate_agent(agent)
        if require_metadata and "observation" not in agent.metadata:
            raise ValueError("Model artifact has no observation compatibility metadata")
        if expected_observation is not None:
            AgentSerializer.validate_observation(agent, expected_observation)
        if expected_timeframe is not None:
            training_metadata = agent.metadata.get("training")
            actual_timeframe = (
                str(training_metadata.get("timeframe", "")).strip().lower()
                if isinstance(training_metadata, dict)
                else ""
            )
            if actual_timeframe != str(expected_timeframe).strip().lower():
                raise ValueError(
                    "Model training timeframe is incompatible: "
                    f"expected {expected_timeframe!r}, got {actual_timeframe or 'missing'}"
                )

        return agent

    @staticmethod
    def validate_observation(agent: Agent, expected: dict) -> None:
        actual = agent.metadata.get("observation")
        if not isinstance(actual, dict):
            raise ValueError("Model artifact has no observation compatibility metadata")
        differences = {
            key: (actual.get(key), value)
            for key, value in expected.items()
            if actual.get(key) != value
        }
        if differences:
            raise ValueError(f"Model observation schema is incompatible: {differences}")
        if agent.genome.input_size != expected.get("input_size"):
            raise ValueError("Model input dimension does not match the observation schema")
        if agent.genome.output_size != 1:
            raise ValueError("Paper model must have exactly one target-position output")

    @staticmethod
    def _validate_agent(agent: Agent) -> None:
        genome = agent.genome
        if genome.input_size <= 0 or genome.output_size <= 0:
            raise ValueError("Model input and output dimensions must be positive")
        node_ids = [node.node_id for node in genome.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Model contains duplicate node IDs")
        if not genome.nodes:
            raise ValueError("Model contains no nodes")
        if sum(node.node_type == "input" for node in genome.nodes) != genome.input_size:
            raise ValueError("Model input-node count does not match input_size")
        if sum(node.node_type == "output" for node in genome.nodes) != genome.output_size:
            raise ValueError("Model output-node count does not match output_size")
        if agent.hidden_size <= 0 or agent.memory_size < 0:
            raise ValueError("Model hidden_size and memory_size are invalid")
        available_ids = set(node_ids)
        innovations = set()
        for connection in genome.connections:
            if connection.innovation in innovations:
                raise ValueError("Model contains duplicate connection innovations")
            innovations.add(connection.innovation)
            if connection.source not in available_ids or connection.target not in available_ids:
                raise ValueError("Model connection references an unknown node")
            if not math.isfinite(float(connection.weight)):
                raise ValueError("Model contains a non-finite connection weight")
        if not isinstance(agent.metadata, dict):
            raise ValueError("Model metadata must be an object")
