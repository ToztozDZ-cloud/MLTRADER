from __future__ import annotations

import pytest

from agents.agent import Agent
from agents.genome import ConnectionGene, Genome, NodeGene
from agents.serialization import AgentSerializer
from environment.observation import ObservationBuilder


def _agent(lookback: int = 8) -> Agent:
    input_size = lookback * 5 + 4
    nodes = [NodeGene(index, "input", "linear") for index in range(input_size)]
    nodes.append(NodeGene(input_size, "output", "tanh"))
    return Agent(Genome(input_size, 1, nodes, [ConnectionGene(0, 0, input_size, 0.25)]))


def test_model_artifact_round_trip_and_observation_compatibility(tmp_path) -> None:
    agent = _agent()
    schema = ObservationBuilder.schema(8)
    path = tmp_path / "model.json"
    AgentSerializer.save(
        agent, path, metadata={"observation": schema, "training_seed": 5}
    )

    loaded = AgentSerializer.load(
        path,
        expected_observation=schema,
        require_metadata=True,
    )
    assert loaded.genome.input_size == 44
    assert loaded.genome.output_size == 1
    assert loaded.metadata["training_seed"] == 5


def test_model_artifact_rejects_incompatible_schema(tmp_path) -> None:
    path = tmp_path / "model.json"
    AgentSerializer.save(
        _agent(),
        path,
        metadata={"observation": ObservationBuilder.schema(8)},
    )
    with pytest.raises(ValueError, match="incompatible"):
        AgentSerializer.load(
            path,
            expected_observation=ObservationBuilder.schema(9),
            require_metadata=True,
        )


def test_model_artifact_rejects_incompatible_timeframe(tmp_path) -> None:
    path = tmp_path / "model.json"
    AgentSerializer.save(
        _agent(),
        path,
        metadata={
            "observation": ObservationBuilder.schema(8),
            "training": {"timeframe": "1d"},
        },
    )

    with pytest.raises(ValueError, match="training timeframe is incompatible"):
        AgentSerializer.load(path, expected_timeframe="5m")


def test_model_artifact_rejects_non_finite_weights(tmp_path) -> None:
    agent = _agent()
    agent.genome.connections[0].weight = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        AgentSerializer.save(agent, tmp_path / "bad.json")
