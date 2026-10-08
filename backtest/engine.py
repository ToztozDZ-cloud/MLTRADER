from __future__ import annotations

import pandas as pd

from agents.agent import Agent
from environment.costs import TradingCosts
from training.evaluator import AgentEvaluator
from backtest.metrics import calculate_metrics


class BacktestEngine:
    def __init__(
        self,
        lookback: int = 64,
        initial_cash: float = 100_000.0,
        costs: TradingCosts | None = None,
        allow_short: bool = False,
        max_position_fraction: float = 1.0,
        max_leverage: float = 1.0,
        execute_on_next_bar: bool = True,
    ) -> None:
        self.evaluator = AgentEvaluator(
            lookback=lookback,
            initial_cash=initial_cash,
            costs=costs,
            allow_short=allow_short,
            max_position_fraction=max_position_fraction,
            max_leverage=max_leverage,
            execute_on_next_bar=execute_on_next_bar,
        )

    def run(
        self,
        agent: Agent,
        market_data: pd.DataFrame,
    ) -> dict:
        result = self.evaluator.evaluate(
            agent,
            market_data,
        )

        metrics = calculate_metrics(
            equity_curve=result.equity_curve,
            turnover=result.turnover,
        )

        metrics["fitness"] = result.fitness

        return {
            "metrics": metrics,
            "equity_curve": result.equity_curve,
        }