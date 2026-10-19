from __future__ import annotations

import numpy as np


def total_return(equity_curve: list[float]) -> float:
    equity: np.ndarray = np.asarray(equity_curve, dtype=np.float64)

    if len(equity) < 2 or not np.isfinite(equity).all() or equity[0] <= 0:
        return 0.0

    return float(equity[-1] / equity[0] - 1.0)


def max_drawdown(equity_curve: list[float]) -> float:
    equity: np.ndarray = np.asarray(equity_curve, dtype=np.float64)

    if len(equity) == 0 or not np.isfinite(equity).all() or np.any(equity <= 0):
        return 1.0 if len(equity) else 0.0
    if len(equity) == 0:
        return 0.0

    running_max = np.maximum.accumulate(equity)

    drawdowns = equity / running_max - 1.0

    return float(abs(np.min(drawdowns)))


def volatility(equity_curve: list[float], periods_per_year: float = 252.0) -> float:
    equity: np.ndarray = np.asarray(equity_curve, dtype=np.float64)

    if len(equity) < 2 or not np.isfinite(equity).all() or np.any(equity <= 0):
        return 0.0

    returns = np.diff(equity) / equity[:-1]

    if not np.isfinite(returns).all():
        return float("inf")
    return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year)) if len(returns) > 1 else 0.0


def sharpe_ratio(equity_curve: list[float], periods_per_year: float = 252.0) -> float:
    equity: np.ndarray = np.asarray(equity_curve, dtype=np.float64)

    if len(equity) < 2 or not np.isfinite(equity).all() or np.any(equity <= 0):
        return 0.0

    returns = np.diff(equity) / equity[:-1]

    std = np.std(returns, ddof=1) if len(returns) > 1 else 0.0

    if std == 0:
        return 0.0

    if not np.isfinite(returns).all():
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(periods_per_year))


def calmar_ratio(equity_curve: list[float], periods_per_year: float = 252.0) -> float:
    drawdown = max_drawdown(equity_curve)
    observations = max(0, len(equity_curve) - 1)
    annual_return = 0.0
    if observations and equity_curve[0] > 0 and equity_curve[-1] > 0:
        annual_return = float(
            (equity_curve[-1] / equity_curve[0]) ** (periods_per_year / observations) - 1.0
        )

    if drawdown == 0:
        return 0.0

    return annual_return / drawdown


def calculate_metrics(
    equity_curve: list[float],
    turnover: float = 0.0,
    *,
    periods_per_year: float = 252.0,
    trade_count: int = 0,
    transaction_costs: float = 0.0,
    average_gross_exposure: float = 0.0,
    max_gross_exposure: float = 0.0,
) -> dict:
    observations = max(0, len(equity_curve) - 1)
    annualized_return = 0.0
    if observations and equity_curve[0] > 0 and equity_curve[-1] > 0:
        annualized_return = float(
            (equity_curve[-1] / equity_curve[0]) ** (periods_per_year / observations) - 1.0
        )
    return {
        "total_return": total_return(equity_curve),
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown(equity_curve),
        "volatility": volatility(equity_curve, periods_per_year),
        "sharpe": sharpe_ratio(equity_curve, periods_per_year),
        "calmar": calmar_ratio(equity_curve, periods_per_year),
        "turnover": float(turnover),
        "trade_count": int(trade_count),
        "transaction_costs": float(transaction_costs),
        "average_gross_exposure": float(average_gross_exposure),
        "max_gross_exposure": float(max_gross_exposure),
        "final_equity": float(equity_curve[-1]),
    }
