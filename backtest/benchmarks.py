from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from agents.agent import Agent
from backtest.engine import BacktestEngine
import pandas as pd


@dataclass
class ReferencePolicy:
    name: str
    lookback: int
    max_position_fraction: float = 1.0
    fitness: float = float("-inf")

    def reset(self) -> None:
        return None

    def act(self, observation: np.ndarray) -> float:
        if self.name == "cash":
            return 0.0
        if self.name == "buy_and_hold":
            exposure_index = self.lookback * 5 + 1
            exposure = float(observation[exposure_index])
            if abs(exposure) < 1e-8:
                return 1.0
            return float(np.clip(exposure / self.max_position_fraction, -1.0, 1.0))
        if self.name == "moving_average":
            market = np.asarray(observation[: self.lookback * 5]).reshape(
                self.lookback, 5
            )
            closes = market[:, 3]
            short = float(np.mean(closes[-min(5, self.lookback) :]))
            long = float(np.mean(closes[-min(20, self.lookback) :]))
            return 1.0 if short > long else 0.0
        raise ValueError(f"Unknown reference policy: {self.name}")


def compare_strategies(
    agent: Agent,
    market_data: pd.DataFrame,
    engine: BacktestEngine,
    *,
    start_step: int | None = None,
) -> dict[str, dict]:
    """Run policy and transparent baselines through identical bars and costs."""
    policies = {
        "cash": ReferencePolicy("cash", engine.evaluator.lookback),
        "buy_and_hold": ReferencePolicy(
            "buy_and_hold",
            engine.evaluator.lookback,
            engine.evaluator.max_position_fraction,
        ),
        "moving_average": ReferencePolicy("moving_average", engine.evaluator.lookback),
        "agent": agent,
    }
    return {
        name: engine.run(cast(Agent, policy), market_data, start_step=start_step)[
            "metrics"
        ]
        for name, policy in policies.items()
    }
