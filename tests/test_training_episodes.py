from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from evolution.evolution import EvolutionEngine
from evolution.population import Population
from training.evaluator import AgentEvaluator, EvaluationResult
from training.episodes import TrainingEpisodeSampler
from training.trainer import Trainer


def _dataset(symbol: str, rows: int = 240) -> pd.DataFrame:
    close = 100.0 + np.arange(rows, dtype=np.float64) * (
        0.02 if symbol == "AAA" else 0.03
    )
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 1000.0),
            "symbol": symbol,
        }
    )


def test_sampled_episodes_are_bounded_and_seed_reproducible() -> None:
    datasets = [_dataset("AAA"), _dataset("BBB")]
    sampler = TrainingEpisodeSampler(
        lookback=16,
        episode_bars=32,
        randomize_start=True,
        randomize_instrument=True,
        randomize_market_regime=True,
    )
    first = sampler.sample(datasets, episodes=8, rng=np.random.default_rng(1234))
    repeated = sampler.sample(datasets, episodes=8, rng=np.random.default_rng(1234))
    changed = sampler.sample(datasets, episodes=8, rng=np.random.default_rng(9876))

    def signature(items):
        return [
            (item.instrument, item.start_index, item.end_index, item.regime)
            for item in items
        ]

    assert signature(first) == signature(repeated)
    assert signature(first) != signature(changed)
    assert {item.instrument for item in first} == {"AAA", "BBB"}
    for item in first:
        source = next(
            frame for frame in datasets if frame["symbol"].iloc[0] == item.instrument
        )
        assert item.start_index >= 15
        assert item.end_index < len(source)
        assert len(item.data) == 16 + 32
        assert (
            item.data.iloc[15]["timestamp"]
            == source.iloc[item.start_index]["timestamp"]
        )
        assert (
            item.data.iloc[-1]["timestamp"] == source.iloc[item.end_index]["timestamp"]
        )
        assert item.regime in {"low_volatility", "medium_volatility", "high_volatility"}


def test_nonrandom_episode_sampling_spreads_windows_across_training_data() -> None:
    sampler = TrainingEpisodeSampler(
        lookback=16,
        episode_bars=32,
        randomize_start=False,
        randomize_instrument=False,
    )

    episodes = sampler.sample(
        [_dataset("AAA")],
        episodes=4,
        rng=np.random.default_rng(99),
    )

    assert len({episode.start_index for episode in episodes}) == 4


class _RecordingEvaluator:
    lookback = 16
    using_cuda = False
    device = "cpu"

    def __init__(self):
        self.calls = []

    def evaluate(self, agent, market_data, start_step=None):
        self.calls.append((agent, market_data.copy(), start_step))
        fitness = float(market_data["close"].iloc[0] % 7)
        return EvaluationResult(
            fitness=fitness,
            final_equity=1000.0 + fitness,
            total_return=fitness / 1000.0,
            max_drawdown=0.1,
            turnover=0.2,
            equity_curve=[1000.0, 1000.0 + fitness],
        )


def test_trainer_evaluates_every_agent_on_multiple_logged_episodes(tmp_path) -> None:
    population = Population(size=2, input_size=4, output_size=1)
    evolution = EvolutionEngine(population)
    evaluator = _RecordingEvaluator()
    trainer = Trainer(
        population,
        evolution,
        cast(AgentEvaluator, evaluator),
        generations=1,
        checkpoint_dir=tmp_path / "archives",
        champion_dir=tmp_path / "champions",
        episodes_per_agent=4,
        episode_bars=32,
        randomize_start=True,
        randomize_instrument=True,
        randomize_market_regime=True,
        seed=55,
        resume_champion=False,
    )

    results = trainer._evaluate_population([_dataset("AAA"), _dataset("BBB")])

    assert len(evaluator.calls) == 8
    assert all(start_step == 15 for _, _, start_step in evaluator.calls)
    first_agent_episodes = [
        call[1]["timestamp"].iloc[15] for call in evaluator.calls[:4]
    ]
    second_agent_episodes = [
        call[1]["timestamp"].iloc[15] for call in evaluator.calls[4:]
    ]
    assert first_agent_episodes == second_agent_episodes
    assert all(result.trade_count == 0 for result in results)
    log_lines = (
        (tmp_path / "archives" / "training_episodes.jsonl").read_text().splitlines()
    )
    assert len(log_lines) == 2
    assert '"episode_count": 4' in log_lines[0]
    assert '"aggregate"' in log_lines[0]
