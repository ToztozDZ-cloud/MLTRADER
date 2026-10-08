from __future__ import annotations

import numpy as np


class FitnessCalculator:
    """
    Calculates evolutionary fitness from an equity curve.

    The objective is not simply maximum profit. The fitness function
    rewards returns while penalizing drawdown, volatility and excessive
    trading activity.
    """

    def __init__(
        self,
        return_weight: float = 1.0,
        sharpe_weight: float = 0.5,
        drawdown_penalty: float = 1.0,
        volatility_penalty: float = 0.25,
        turnover_penalty: float = 0.10,
    ) -> None:
        self.return_weight = return_weight
        self.sharpe_weight = sharpe_weight
        self.drawdown_penalty = drawdown_penalty
        self.volatility_penalty = volatility_penalty
        self.turnover_penalty = turnover_penalty

    def calculate(
        self,
        equity_curve: list[float],
        turnover: float = 0.0,
    ) -> float:
        equity = np.asarray(
            equity_curve,
            dtype=np.float64,
        )

        if len(equity) < 2:
            return float("-inf")

        if not np.all(np.isfinite(equity)):
            return float("-inf")

        initial = equity[0]

        if initial <= 0:
            return float("-inf")

        returns = np.diff(equity) / equity[:-1]

        total_return = (
            equity[-1] / initial
        ) - 1.0

        volatility = float(
            np.std(returns)
        )

        if volatility > 0:
            sharpe = float(
                np.mean(returns) / volatility
            ) * np.sqrt(252.0)
        else:
            sharpe = 0.0

        running_max = np.maximum.accumulate(equity)

        drawdowns = (
            equity / running_max
        ) - 1.0

        max_drawdown = abs(
            float(np.min(drawdowns))
        )

        fitness = (
            self.return_weight * total_return
            + self.sharpe_weight * sharpe
            - self.drawdown_penalty * max_drawdown
            - self.volatility_penalty * volatility
            - self.turnover_penalty * turnover
        )

        return float(fitness)