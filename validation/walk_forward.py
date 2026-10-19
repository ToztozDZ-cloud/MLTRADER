from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Callable

import numpy as np
import pandas as pd

from agents.agent import Agent
from backtest.benchmarks import compare_strategies
from backtest.engine import BacktestEngine


@dataclass
class WalkForwardResult:
    fold: int
    training_start: int
    training_end: int
    validation_start: int
    validation_end: int
    training_start_timestamp: str | None
    training_end_timestamp: str | None
    validation_start_timestamp: str | None
    validation_end_timestamp: str | None
    seed: int
    training_fitness: float
    model_id: str
    metrics: dict
    benchmarks: dict


class WalkForwardValidator:
    """Chronological walk-forward evaluation that retrains from scratch per fold."""

    def __init__(
        self,
        engine: BacktestEngine,
        train_size: int,
        validation_size: int,
        step_size: int | None = None,
        *,
        expanding: bool = False,
        minimum_history: int | None = None,
        embargo_bars: int = 0,
        seed: int = 0,
    ) -> None:
        self.engine = engine
        self.train_size = int(train_size)
        self.validation_size = int(validation_size)
        self.step_size = int(step_size if step_size is not None else validation_size)
        self.expanding = bool(expanding)
        self.minimum_history = int(minimum_history or train_size)
        self.embargo_bars = int(embargo_bars)
        self.seed = int(seed)
        lookback = self.engine.evaluator.lookback
        if self.train_size < lookback + 2:
            raise ValueError(
                "train_size must include lookback history and at least two scored bars"
            )
        if self.minimum_history < lookback + 2:
            raise ValueError(
                "minimum_history must include lookback history and at least two scored bars"
            )
        if self.validation_size < 2 or self.step_size < 1 or self.embargo_bars < 0:
            raise ValueError("validation_size, step_size, and embargo_bars are invalid")

    def validate(
        self,
        market_data: pd.DataFrame,
        train_agent: Callable[[pd.DataFrame, int], Agent],
    ) -> list[WalkForwardResult]:
        if not callable(train_agent):
            raise TypeError(
                "train_agent must train and return a new Agent for each fold"
            )
        if market_data.empty:
            raise ValueError("Walk-forward market data is empty")

        results: list[WalkForwardResult] = []
        trained_agents: list[Agent] = []
        validation_start = self.train_size + self.embargo_bars
        fold = 0
        while validation_start + self.validation_size <= len(market_data):
            training_end = validation_start - self.embargo_bars
            training_start = (
                0 if self.expanding else max(0, training_end - self.train_size)
            )
            if training_end - training_start < self.minimum_history:
                if self.expanding:
                    raise ValueError("Training window is shorter than minimum_history")
                break

            training_data = market_data.iloc[training_start:training_end].copy()
            validation_end = validation_start + self.validation_size
            fold_seed = int(
                np.random.SeedSequence([self.seed, fold]).generate_state(1)[0]
            )
            agent = train_agent(training_data.copy(), fold_seed)
            if not isinstance(agent, Agent):
                raise TypeError("train_agent must return an agents.agent.Agent")
            if any(agent is previous for previous in trained_agents):
                raise ValueError(
                    "Each walk-forward fold must return a freshly trained Agent instance"
                )
            trained_agents.append(agent)
            training_fitness = float(agent.fitness)

            lookback = self.engine.evaluator.lookback
            history_start = validation_start - lookback + 1
            if history_start < training_start:
                raise ValueError(
                    "Training window is too short to supply the validation lookback history"
                )
            validation_data = (
                market_data.iloc[history_start:validation_end]
                .copy()
                .reset_index(drop=True)
            )
            scored_start = validation_start - history_start
            backtest = self.engine.run(agent, validation_data, start_step=scored_start)
            benchmark_metrics = compare_strategies(
                agent,
                validation_data,
                self.engine,
                start_step=scored_start,
            )

            results.append(
                WalkForwardResult(
                    fold=fold,
                    training_start=training_start,
                    training_end=training_end,
                    validation_start=validation_start,
                    validation_end=validation_end,
                    training_start_timestamp=self._timestamp(
                        market_data, training_start
                    ),
                    training_end_timestamp=self._timestamp(
                        market_data, training_end - 1
                    ),
                    validation_start_timestamp=self._timestamp(
                        market_data, validation_start
                    ),
                    validation_end_timestamp=self._timestamp(
                        market_data, validation_end - 1
                    ),
                    seed=fold_seed,
                    training_fitness=training_fitness,
                    model_id=self._model_id(agent),
                    metrics=backtest["metrics"],
                    benchmarks=benchmark_metrics,
                )
            )
            fold += 1
            validation_start += self.step_size
        return results

    @staticmethod
    def _timestamp(data: pd.DataFrame, index: int) -> str | None:
        if "timestamp" not in data or not 0 <= index < len(data):
            return None
        return pd.Timestamp(data.iloc[index]["timestamp"]).isoformat()

    @staticmethod
    def _model_id(agent: Agent) -> str:
        payload = {
            "input_size": agent.genome.input_size,
            "output_size": agent.genome.output_size,
            "nodes": [
                (node.node_id, node.node_type, node.activation)
                for node in agent.genome.nodes
            ],
            "connections": [
                (gene.innovation, gene.source, gene.target, gene.weight, gene.enabled)
                for gene in agent.genome.connections
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()[:16]

    @staticmethod
    def summarize(results: list[WalkForwardResult]) -> dict:
        if not results:
            return {"fold_count": 0, "median_return": None, "worst_fold_return": None}
        returns = [float(result.metrics["total_return"]) for result in results]
        return {
            "fold_count": len(results),
            "median_return": float(np.median(returns)),
            "mean_return": float(np.mean(returns)),
            "worst_fold_return": float(np.min(returns)),
            "worst_fold_drawdown": float(
                max(float(result.metrics["max_drawdown"]) for result in results)
            ),
            "folds": [asdict(result) for result in results],
        }
