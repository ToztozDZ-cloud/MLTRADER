from __future__ import annotations

import numpy as np


def periods_per_year_for_timeframe(timeframe: str) -> float:
    value = str(timeframe).strip().lower()
    if value in {"1d", "d", "day", "daily"}:
        return 252.0
    if len(value) < 2 or value[-1] not in {"m", "h"} or not value[:-1].isdigit():
        raise ValueError(f"Unsupported annualization timeframe: {timeframe!r}")
    interval = int(value[:-1])
    if interval <= 0:
        raise ValueError("Timeframe interval must be positive")
    minutes = interval if value[-1] == "m" else interval * 60
    return 252.0 * 390.0 / minutes


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
        periods_per_year: float = 252.0,
        max_drawdown_limit: float = 0.50,
        drawdown_excess_penalty: float = 2.0,
        sharpe_cap: float = 5.0,
    ) -> None:
        weights = {
            "return_weight": return_weight,
            "sharpe_weight": sharpe_weight,
            "drawdown_penalty": drawdown_penalty,
            "volatility_penalty": volatility_penalty,
            "turnover_penalty": turnover_penalty,
            "drawdown_excess_penalty": drawdown_excess_penalty,
        }
        for name, value in weights.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 < max_drawdown_limit < 1.0:
            raise ValueError("max_drawdown_limit must be in (0, 1)")
        if not np.isfinite(sharpe_cap) or sharpe_cap <= 0:
            raise ValueError("sharpe_cap must be finite and positive")
        self.return_weight = float(return_weight)
        self.sharpe_weight = float(sharpe_weight)
        self.drawdown_penalty = float(drawdown_penalty)
        self.volatility_penalty = float(volatility_penalty)
        self.turnover_penalty = float(turnover_penalty)
        self.max_drawdown_limit = float(max_drawdown_limit)
        self.drawdown_excess_penalty = float(drawdown_excess_penalty)
        self.sharpe_cap = float(sharpe_cap)
        self.periods_per_year = float(periods_per_year)
        if not np.isfinite(self.periods_per_year) or self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be finite and positive.")

    def calculate(
        self,
        equity_curve: list[float],
        turnover: float = 0.0,
    ) -> float:
        equity: np.ndarray = np.asarray(
            equity_curve,
            dtype=np.float64,
        )

        if len(equity) < 3:
            return float("-inf")

        if not np.all(np.isfinite(equity)) or np.any(equity <= 0):
            return float("-inf")

        initial = equity[0]

        if initial <= 0:
            return float("-inf")

        returns = np.diff(equity) / equity[:-1]

        total_return = (
            equity[-1] / initial
        ) - 1.0

        if not np.all(np.isfinite(returns)) or not np.isfinite(turnover) or turnover < 0:
            return float("-inf")

        volatility = float(np.std(returns, ddof=1))

        if volatility > 0:
            sharpe = float(
                np.mean(returns) / volatility
            ) * np.sqrt(self.periods_per_year)
            sharpe = float(np.clip(sharpe, -self.sharpe_cap, self.sharpe_cap))
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
            - self.drawdown_excess_penalty * max(0.0, max_drawdown - self.max_drawdown_limit)
            - self.volatility_penalty * volatility
            - self.turnover_penalty * turnover
        )

        return float(fitness) if np.isfinite(fitness) else float("-inf")
