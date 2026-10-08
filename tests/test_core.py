from __future__ import annotations

import random

import numpy as np
import pandas as pd

from agents.agent import Agent
from agents.genome import (
    ConnectionGene,
    Genome,
    NodeGene,
)
from backtest.engine import BacktestEngine
from environment.costs import TradingCosts
from training.evaluator import AgentEvaluator


def create_test_data(rows: int = 300) -> pd.DataFrame:
    random.seed(42)
    np.random.seed(42)

    prices = [100.0]

    for _ in range(rows - 1):
        change = np.random.normal(
            0.0005,
            0.01,
        )

        prices.append(
            prices[-1] * (1.0 + change)
        )

    prices = np.asarray(prices)

    data = pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": np.random.randint(
                1_000_000,
                5_000_000,
                rows,
            ),
        }
    )

    return data


def create_agent(
    input_size: int,
) -> Agent:
    nodes = []

    for node_id in range(input_size):
        nodes.append(
            NodeGene(
                node_id=node_id,
                node_type="input",
                activation="linear",
            )
        )

    output_id = input_size

    nodes.append(
        NodeGene(
            node_id=output_id,
            node_type="output",
            activation="tanh",
        )
    )

    connections = []

    for node_id in range(input_size):
        connections.append(
            ConnectionGene(
                innovation=node_id,
                source=node_id,
                target=output_id,
                weight=random.gauss(
                    0.0,
                    0.1,
                ),
                enabled=True,
            )
        )

    genome = Genome(
        input_size=input_size,
        output_size=1,
        nodes=nodes,
        connections=connections,
    )

    return Agent(
        genome=genome,
    )


def test_agent_action() -> None:
    input_size = 64 * 5 + 4

    agent = create_agent(
        input_size
    )

    observation = np.random.randn(
        input_size
    ).astype(np.float32)

    action = agent.act(
        observation
    )

    assert np.isfinite(action)
    assert -1.0 <= action <= 1.0

    print(
        "PASS: agent action"
    )


def test_mutation_changes_genome() -> None:
    input_size = 64 * 5 + 4

    agent = create_agent(
        input_size
    )

    original = agent.clone()

    agent.genome.mutate(
        weight_rate=1.0,
        weight_sigma=0.5,
        connection_rate=0.0,
        add_node_rate=1.0,
        add_connection_rate=1.0,
        remove_node_rate=0.0,
        remove_connection_rate=0.0,
    )

    changed_weights = (
        agent.genome.connections
        != original.genome.connections
    )

    changed_topology = (
        len(agent.genome.nodes)
        != len(original.genome.nodes)
        or len(agent.genome.connections)
        != len(original.genome.connections)
    )

    assert changed_weights or changed_topology

    print(
        "PASS: genome mutation"
    )


def test_market_evaluation() -> None:
    data = create_test_data()

    input_size = 64 * 5 + 4

    agent = create_agent(
        input_size
    )

    evaluator = AgentEvaluator(
        lookback=64,
        initial_cash=100_000.0,
        costs=TradingCosts(),
        allow_short=False,
        max_position_fraction=1.0,
    )

    result = evaluator.evaluate(
        agent,
        data,
    )

    assert np.isfinite(
        result.fitness
    )

    assert np.isfinite(
        result.final_equity
    )

    assert len(
        result.equity_curve
    ) > 1

    print(
        "PASS: market evaluation"
    )

    print(
        f"  Fitness: {result.fitness:.6f}"
    )

    print(
        f"  Final equity: "
        f"${result.final_equity:,.2f}"
    )

    print(
        f"  Return: "
        f"{result.total_return:.2%}"
    )


def test_backtest() -> None:
    data = create_test_data()

    input_size = 64 * 5 + 4

    agent = create_agent(
        input_size
    )

    engine = BacktestEngine(
        lookback=64,
        initial_cash=100_000.0,
    )

    result = engine.run(
        agent,
        data,
    )

    metrics = result["metrics"]

    assert "total_return" in metrics
    assert "sharpe" in metrics
    assert "max_drawdown" in metrics
    assert "final_equity" in metrics

    print(
        "PASS: backtest engine"
    )


if __name__ == "__main__":
    print("=" * 60)
    print("MLTRADER CORE INTEGRITY TEST")
    print("=" * 60)

    test_agent_action()
    test_mutation_changes_genome()
    test_market_evaluation()
    test_backtest()

    print("=" * 60)
    print("ALL CORE TESTS PASSED")
    print("=" * 60)