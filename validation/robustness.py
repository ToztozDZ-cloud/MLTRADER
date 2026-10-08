from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from agents.agent import Agent
from backtest.engine import BacktestEngine
from environment.costs import TradingCosts


@dataclass
class RobustnessResult:
    scenario: str
    metrics: dict


class RobustnessValidator:
    """
    Tests whether the agent remains viable when execution conditions
    become less favorable.
    """

    def __init__(
        self,
        lookback: int = 64,
        initial_cash: float = 100_000.0,
        allow_short: bool = False,
        max_position_fraction: float = 1.0,
    ) -> None:
        self.lookback = lookback
        self.initial_cash = initial_cash
        self.allow_short = allow_short
        self.max_position_fraction = max_position_fraction

    def validate(
        self,
        agent: Agent,
        market_data: pd.DataFrame,
    ) -> list[RobustnessResult]:
        scenarios = {
            "normal": TradingCosts(
                commission_rate=0.0005,
                spread_rate=0.0002,
                slippage_rate=0.0005,
            ),
            "high_cost": TradingCosts(
                commission_rate=0.0010,
                spread_rate=0.0005,
                slippage_rate=0.0010,
            ),
            "severe_cost": TradingCosts(
                commission_rate=0.0020,
                spread_rate=0.0010,
                slippage_rate=0.0020,
            ),
        }

        results = []

        for name, costs in scenarios.items():
            engine = BacktestEngine(
                lookback=self.lookback,
                initial_cash=self.initial_cash,
                costs=costs,
                allow_short=self.allow_short,
                max_position_fraction=self.max_position_fraction,
            )

            result = engine.run(
                agent,
                market_data,
            )

            results.append(
                RobustnessResult(
                    scenario=name,
                    metrics=result["metrics"],
                )
            )

        return results