from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents.agent import Agent
from agents.genome import ConnectionGene, Genome, NodeGene
from backtest.engine import BacktestEngine
from environment.costs import TradingCosts
from validation.out_of_sample import OutOfSampleValidator
from validation.walk_forward import WalkForwardValidator


def _data(rows: int = 100) -> pd.DataFrame:
    close = 100.0 + np.arange(rows, dtype=np.float64) * 0.1
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 1000.0),
            "symbol": "TEST",
        }
    )


def _agent(lookback: int = 4) -> Agent:
    input_size = lookback * 5 + 4
    nodes = [NodeGene(index, "input", "linear") for index in range(input_size)]
    nodes.append(NodeGene(input_size, "output", "tanh"))
    genome = Genome(
        input_size,
        1,
        nodes,
        [ConnectionGene(0, 0, input_size, 0.0)],
    )
    return Agent(genome)


def _engine(lookback: int = 4) -> BacktestEngine:
    return BacktestEngine(
        lookback=lookback,
        initial_cash=10_000.0,
        costs=TradingCosts(0.0, 0.0, 0.0),
        max_position_fraction=0.5,
        timeframe="1d",
    )


def test_walk_forward_retrains_fresh_agents_on_prior_only_windows() -> None:
    data = _data(100)
    validator = WalkForwardValidator(
        _engine(),
        train_size=25,
        validation_size=10,
        step_size=10,
        embargo_bars=2,
        seed=123,
    )
    training_windows = []
    agents = []

    def train_agent(training_data, fold_seed):
        training_windows.append((training_data.copy(), fold_seed))
        agent = _agent()
        agent.fitness = 0.75
        agents.append(agent)
        return agent

    results = validator.validate(data, train_agent)

    assert len(results) == 7
    assert len({id(agent) for agent in agents}) == len(results)
    assert len({seed for _, seed in training_windows}) == len(results)
    assert results[0].training_start == 0
    assert results[0].training_end == 25
    assert results[0].validation_start == 27
    assert results[0].validation_end == 37
    assert training_windows[0][0]["timestamp"].max() < data.iloc[27]["timestamp"]
    assert results[0].training_fitness == 0.75
    assert results[0].metrics["fitness"] == pytest.approx(0.0)
    assert results[0].model_id
    assert set(results[0].benchmarks) == {
        "cash",
        "buy_and_hold",
        "moving_average",
        "agent",
    }


def test_walk_forward_rejects_reusing_the_same_model_instance() -> None:
    validator = WalkForwardValidator(
        _engine(), train_size=20, validation_size=10, step_size=10, embargo_bars=0
    )
    shared = _agent()
    shared.fitness = 0.5

    with pytest.raises(ValueError, match="freshly trained"):
        validator.validate(_data(80), lambda _data, _seed: shared)


def test_chronological_holdout_keeps_embargo_outside_scored_period() -> None:
    partitions = OutOfSampleValidator.split(
        1000,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
        embargo_bars=4,
    )
    assert partitions.train_end == 600
    assert partitions.validation_start == 604
    assert partitions.validation_end == 800
    assert partitions.test_start == 804
    assert partitions.test_end == 1000


def test_holdout_evaluates_only_final_partition_with_historical_warmup() -> None:
    data = _data(1000)
    validator = OutOfSampleValidator(_engine())
    result = validator.validate_holdout(
        _agent(),
        data,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
        embargo_bars=4,
    )
    assert result["evaluated_configurations"] == 1
    assert result["evaluated_folds"] == 1
    assert result["holdout_bars"] == 196
    assert len(result["equity_curve"]) == 196
    assert (
        result["partitions"]["test_start_timestamp"]
        == data.iloc[804]["timestamp"].isoformat()
    )
