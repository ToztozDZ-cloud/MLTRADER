from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
