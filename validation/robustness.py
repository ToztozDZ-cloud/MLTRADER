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
        max_leverage: float = 1.0,
        timeframe: str = "1d",
        base_costs: TradingCosts | None = None,
        execution_delay_bars: int = 0,
    ) -> None:
        self.lookback = lookback
        self.initial_cash = initial_cash
        self.allow_short = allow_short
        self.max_position_fraction = max_position_fraction
        self.max_leverage = max_leverage
        self.timeframe = timeframe
        self.base_costs = base_costs or TradingCosts()
        self.execution_delay_bars = int(execution_delay_bars)
        if self.execution_delay_bars < 0:
            raise ValueError("execution_delay_bars must be non-negative")

    def validate(
        self,
        agent: Agent,
        market_data: pd.DataFrame,
        *,
        start_step: int | None = None,
    ) -> list[RobustnessResult]:
        base = self.base_costs
        scenarios = {
            "configured_costs": base,
            "double_costs": TradingCosts(
                commission_rate=base.commission_rate * 2.0,
                spread_rate=base.spread_rate * 2.0,
                slippage_rate=base.slippage_rate * 2.0,
            ),
            "fourfold_costs": TradingCosts(
                commission_rate=base.commission_rate * 4.0,
                spread_rate=base.spread_rate * 4.0,
                slippage_rate=base.slippage_rate * 4.0,
            ),
            "one_bar_execution_delay": base,
        }

        results = []

        for name, costs in scenarios.items():
            delay = self.execution_delay_bars + (
                1 if name == "one_bar_execution_delay" else 0
            )
            engine = BacktestEngine(
                lookback=self.lookback,
                initial_cash=self.initial_cash,
                costs=costs,
                allow_short=self.allow_short,
                max_position_fraction=self.max_position_fraction,
                max_leverage=self.max_leverage,
                timeframe=self.timeframe,
                execution_delay_bars=delay,
            )

            result = engine.run(
                agent,
                market_data,
                start_step=start_step,
            )

            results.append(
                RobustnessResult(
                    scenario=name,
                    metrics=result["metrics"],
                )
            )

        return results
