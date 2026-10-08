from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from agents.agent import Agent
from backtest.engine import BacktestEngine


@dataclass
class WalkForwardResult:
    training_start: int
    training_end: int
    validation_start: int
    validation_end: int
    metrics: dict


class WalkForwardValidator:
    def __init__(
        self,
        engine: BacktestEngine,
        train_size: int,
        validation_size: int,
        step_size: int | None = None,
    ) -> None:
        self.engine = engine
        self.train_size = train_size
        self.validation_size = validation_size
        self.step_size = (
            step_size
            if step_size is not None
            else validation_size
        )

    def validate(
        self,
        agent: Agent,
        market_data: pd.DataFrame,
    ) -> list[WalkForwardResult]:
        results = []

        start = 0

        while True:
            train_end = start + self.train_size
            validation_end = (
                train_end + self.validation_size
            )

            if validation_end > len(market_data):
                break

            validation_data = market_data.iloc[
                train_end:validation_end
            ].copy()

            result = self.engine.run(
                agent,
                validation_data,
            )

            results.append(
                WalkForwardResult(
                    training_start=start,
                    training_end=train_end,
                    validation_start=train_end,
                    validation_end=validation_end,
                    metrics=result["metrics"],
                )
            )

            start += self.step_size

        return results