from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from agents.agent import Agent
from agents.genome import ConnectionGene, Genome, NodeGene
from environment.costs import TradingCosts
from environment.observation import ObservationBuilder
from training.evaluator import AgentEvaluator


def _market_data(rows: int = 72, intraday: bool = False) -> pd.DataFrame:
    changes = np.sin(np.arange(rows, dtype=np.float64) / 4.0) * 0.002 + 0.0003
    close = 100.0 * np.cumprod(1.0 + changes)
    timestamps = pd.date_range(
        "2024-01-02 15:30" if intraday else "2024-01-02",
        periods=rows,
        freq="5min" if intraday else "1D",
        tz="America/New_York" if intraday else "UTC",
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close * 1.0004,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": 10_000.0 + np.arange(rows) * 10.0,
        }
    )


def _constant_action_agent(lookback: int, action: float) -> Agent:
    input_size = ObservationBuilder.input_size(lookback)
    output_id = input_size
    bias_id = input_size + 1
    return Agent(
        genome=Genome(
            input_size=input_size,
            output_size=1,
            nodes=[
                *[
                    NodeGene(node_id=index, node_type="input", activation="linear")
                    for index in range(input_size)
                ],
                NodeGene(node_id=output_id, node_type="output", activation="tanh"),
                NodeGene(node_id=bias_id, node_type="bias", activation="linear"),
            ],
            connections=[
                ConnectionGene(
                    innovation=0,
                    source=bias_id,
                    target=output_id,
                    weight=float(np.arctanh(action)),
                    enabled=True,
                )
            ],
        )
    )


def _cyclic_agent(lookback: int) -> Agent:
    agent = _constant_action_agent(lookback, 0.1)
    output_id = agent.genome.input_size
    bias_id = output_id + 1
    agent.genome.connections.append(
        ConnectionGene(
            innovation=1,
            source=output_id,
            target=bias_id,
            weight=1.0,
        )
    )
    return agent


def _available_evaluation_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if AgentEvaluator._xpu_available():
        devices.append("xpu")
    return devices


def test_mutate_connections_does_not_reactivate_a_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    genome = Genome(
        input_size=0,
        output_size=0,
        nodes=[
            NodeGene(node_id=node_id, node_type="hidden")
            for node_id in range(3)
        ],
        connections=[
            ConnectionGene(0, 0, 1, 1.0),
            ConnectionGene(1, 1, 2, 1.0),
            ConnectionGene(2, 2, 0, 1.0, enabled=False),
        ],
    )
    draws = iter((1.0, 1.0, 0.0))
    monkeypatch.setattr("agents.genome.random.random", lambda: next(draws))

    genome.mutate_connections(rate=0.5)

    assert not genome.connections[2].enabled
    assert genome._has_path(0, 2)


def test_mutate_connections_allows_acyclic_reactivation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    genome = Genome(
        input_size=0,
        output_size=0,
        nodes=[
            NodeGene(node_id=node_id, node_type="hidden")
            for node_id in range(2)
        ],
        connections=[
            ConnectionGene(0, 0, 1, 1.0, enabled=False),
        ],
    )
    monkeypatch.setattr("agents.genome.random.random", lambda: 0.0)

    genome.mutate_connections(rate=1.0)

    assert genome.connections[0].enabled


@pytest.mark.parametrize(
    ("delay", "allow_short", "intraday"),
    [(0, False, False), (1, True, True)],
)
def test_batched_simulation_matches_reference_accounting(
    delay: int,
    allow_short: bool,
    intraday: bool,
) -> None:
    lookback = 4
    data = _market_data(intraday=intraday)
    evaluator = AgentEvaluator(
        lookback=lookback,
        initial_cash=100_000.0,
        costs=TradingCosts(
            commission_rate=0.0003,
            spread_rate=0.0004,
            slippage_rate=0.0002,
        ),
        allow_short=allow_short,
        max_position_fraction=0.8,
        execution_delay_bars=delay,
        timeframe="5m" if intraday else "1d",
        exchange_timezone="America/New_York",
        flatten_before_close_minutes=10,
        device="cpu",
    )
    agents = [
        _constant_action_agent(lookback, 0.55),
        _constant_action_agent(lookback, -0.45 if allow_short else 0.0),
    ]

    reference = [
        evaluator.evaluate(agent.clone(), data, start_step=lookback - 1)
        for agent in agents
    ]
    batched = evaluator.evaluate_population(
        [agent.clone() for agent in agents],
        data,
        start_step=lookback - 1,
    ).results

    assert len(batched) == len(reference)
    for expected, actual in zip(reference, batched):
        assert actual.trade_count == expected.trade_count
        assert actual.final_equity == pytest.approx(expected.final_equity, abs=0.1)
        assert actual.total_return == pytest.approx(expected.total_return, abs=2e-6)
        assert actual.max_drawdown == pytest.approx(expected.max_drawdown, abs=2e-6)
        assert actual.turnover == pytest.approx(expected.turnover, abs=1e-3)
        assert actual.transaction_costs == pytest.approx(
            expected.transaction_costs,
            abs=0.02,
        )
        assert actual.fitness == pytest.approx(expected.fitness, abs=2e-3)
        np.testing.assert_allclose(
            actual.equity_curve,
            expected.equity_curve,
            rtol=2e-6,
            atol=0.1,
        )


def test_batched_market_features_match_observation_builder() -> None:
    lookback = 5
    data = _market_data(rows=24)
    evaluator = AgentEvaluator(lookback=lookback, device="cpu")

    observations = evaluator._precompute_observations(data, start_step=lookback - 1)
    first_expected = ObservationBuilder.build_from_history(
        data.iloc[:lookback],
        np.zeros(4, dtype=np.float32),
        lookback,
    )

    np.testing.assert_array_equal(observations[0], first_expected)


def test_multi_episode_population_batch_matches_episode_by_episode_results() -> None:
    lookback = 4
    evaluator = AgentEvaluator(
        lookback=lookback,
        initial_cash=100_000.0,
        costs=TradingCosts(
            commission_rate=0.0003,
            spread_rate=0.0004,
            slippage_rate=0.0002,
        ),
        device="cpu",
    )
    agents = [
        _constant_action_agent(lookback, 0.35),
        _constant_action_agent(lookback, -0.15),
    ]
    episodes = [
        _market_data(rows=48),
        _market_data(rows=48).assign(
            open=lambda frame: frame.open * 1.02,
            high=lambda frame: frame.high * 1.02,
            low=lambda frame: frame.low * 1.02,
            close=lambda frame: frame.close * 1.02,
        ),
    ]

    expected = [
        evaluator.evaluate_population(agents, episode, start_step=lookback - 1).results
        for episode in episodes
    ]
    actual = evaluator.evaluate_population_episodes(
        agents, episodes, start_step=lookback - 1
    )

    assert len(actual) == len(expected)
    for batched_episode, reference_episode in zip(actual, expected):
        for batched, reference in zip(batched_episode, reference_episode):
            assert batched.trade_count == reference.trade_count
            assert batched.final_equity == pytest.approx(
                reference.final_equity, abs=0.1
            )
            assert batched.total_return == pytest.approx(
                reference.total_return, abs=2e-6
            )
            assert batched.max_drawdown == pytest.approx(
                reference.max_drawdown, abs=2e-6
            )
            assert batched.turnover == pytest.approx(reference.turnover, abs=1e-3)
            assert batched.transaction_costs == pytest.approx(
                reference.transaction_costs, abs=0.02
            )
            assert batched.fitness == pytest.approx(reference.fitness, abs=2e-3)
            np.testing.assert_allclose(
                batched.equity_curve, reference.equity_curve, rtol=2e-6, atol=0.1
            )


def test_multi_episode_population_batch_rejects_different_window_sizes() -> None:
    evaluator = AgentEvaluator(lookback=4, device="cpu")
    agents = [_constant_action_agent(4, 0.1)]

    with pytest.raises(ValueError, match="same number of rows"):
        evaluator.evaluate_population_episodes(
            agents,
            [_market_data(rows=32), _market_data(rows=33)],
            start_step=3,
        )


def test_cpu_reference_evaluation_rejects_cyclic_genomes() -> None:
    lookback = 4
    evaluator = AgentEvaluator(lookback=lookback, device="cpu")

    with pytest.raises(
        RuntimeError,
        match="Genome contains a cycle or invalid dependency graph",
    ):
        evaluator.evaluate(
            _cyclic_agent(lookback),
            _market_data(),
            start_step=lookback - 1,
        )


@pytest.mark.parametrize("device", _available_evaluation_devices())
def test_population_evaluation_rejects_cyclic_genomes(
    device: str,
) -> None:
    lookback = 4
    evaluator = AgentEvaluator(lookback=lookback, device=device)

    with pytest.raises(
        RuntimeError,
        match="Genome contains a cycle or invalid dependency graph",
    ):
        evaluator.evaluate_population(
            [_cyclic_agent(lookback)],
            _market_data(),
            start_step=lookback - 1,
        )


@pytest.mark.skipif(
    not AgentEvaluator._xpu_available(), reason="PyTorch XPU is unavailable"
)
def test_multi_episode_xpu_batch_matches_cpu_reference() -> None:
    lookback = 4
    cpu_evaluator = AgentEvaluator(lookback=lookback, device="cpu")
    xpu_evaluator = AgentEvaluator(lookback=lookback, device="xpu")
    agents = [
        _constant_action_agent(lookback, 0.3),
        _constant_action_agent(lookback, -0.2),
    ]
    episodes = [
        _market_data(rows=40),
        _market_data(rows=40).assign(
            open=lambda frame: frame.open * 1.01,
            high=lambda frame: frame.high * 1.01,
            low=lambda frame: frame.low * 1.01,
            close=lambda frame: frame.close * 1.01,
        ),
    ]
    expected = [
        cpu_evaluator.evaluate_population(
            agents, episode, start_step=lookback - 1
        ).results
        for episode in episodes
    ]
    actual = xpu_evaluator.evaluate_population_episodes(
        agents, episodes, start_step=lookback - 1
    )

    for xpu_episode, cpu_episode in zip(actual, expected):
        for xpu_result, cpu_result in zip(xpu_episode, cpu_episode):
            assert xpu_result.trade_count == cpu_result.trade_count
            assert xpu_result.final_equity == pytest.approx(
                cpu_result.final_equity, abs=0.1
            )
            assert xpu_result.total_return == pytest.approx(
                cpu_result.total_return, abs=2e-6
            )
            assert xpu_result.max_drawdown == pytest.approx(
                cpu_result.max_drawdown, abs=2e-6
            )
            assert xpu_result.fitness == pytest.approx(cpu_result.fitness, abs=2e-3)
