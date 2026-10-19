from __future__ import annotations

import pandas as pd

from agents.agent import Agent
from environment.costs import TradingCosts
from training.evaluator import AgentEvaluator
from training.fitness import FitnessCalculator
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
        timeframe: str = "1d",
        fitness_calculator: FitnessCalculator | None = None,
        execution_delay_bars: int = 0,
    ) -> None:
        self.evaluator = AgentEvaluator(
            lookback=lookback,
            initial_cash=initial_cash,
            costs=costs,
            allow_short=allow_short,
            max_position_fraction=max_position_fraction,
            max_leverage=max_leverage,
            execute_on_next_bar=execute_on_next_bar,
            execution_delay_bars=execution_delay_bars,
            timeframe=timeframe,
            fitness_calculator=fitness_calculator,
        )
        self.periods_per_year = self.evaluator.fitness_calculator.periods_per_year

    def run(
        self,
        agent: Agent,
        market_data: pd.DataFrame,
        *,
        start_step: int | None = None,
    ) -> dict:
        result = self.evaluator.evaluate(
            agent,
            market_data,
            start_step=start_step,
        )

        metrics = calculate_metrics(
            equity_curve=result.equity_curve,
            turnover=result.turnover,
            periods_per_year=self.periods_per_year,
            trade_count=result.trade_count,
            transaction_costs=result.transaction_costs,
            average_gross_exposure=result.average_gross_exposure,
            max_gross_exposure=result.max_gross_exposure,
        )

        metrics["fitness"] = result.fitness

        return {
            "metrics": metrics,
            "equity_curve": result.equity_curve,
        }
