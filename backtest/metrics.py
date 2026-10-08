from __future__ import annotations

import numpy as np


def total_return(equity_curve: list[float]) -> float:
    equity = np.asarray(equity_curve, dtype=np.float64)

    if len(equity) < 2 or equity[0] <= 0:
        return 0.0

    return float(equity[-1] / equity[0] - 1.0)


def max_drawdown(equity_curve: list[float]) -> float:
    equity = np.asarray(equity_curve, dtype=np.float64)

    if len(equity) == 0:
        return 0.0

    running_max = np.maximum.accumulate(equity)

    drawdowns = equity / running_max - 1.0

    return float(abs(np.min(drawdowns)))


def volatility(equity_curve: list[float]) -> float:
    equity = np.asarray(equity_curve, dtype=np.float64)

    if len(equity) < 2:
        return 0.0

    returns = np.diff(equity) / equity[:-1]

    return float(np.std(returns))


def sharpe_ratio(equity_curve: list[float]) -> float:
    equity = np.asarray(equity_curve, dtype=np.float64)

    if len(equity) < 2:
        return 0.0

    returns = np.diff(equity) / equity[:-1]

    std = np.std(returns)

    if std == 0:
        return 0.0

    return float(
        np.mean(returns) / std * np.sqrt(252.0)
    )


def calmar_ratio(equity_curve: list[float]) -> float:
    annual_return = total_return(equity_curve)
    drawdown = max_drawdown(equity_curve)

    if drawdown == 0:
        return 0.0

    return annual_return / drawdown


def calculate_metrics(
    equity_curve: list[float],
    turnover: float = 0.0,
) -> dict:
    return {
        "total_return": total_return(equity_curve),
        "max_drawdown": max_drawdown(equity_curve),
        "volatility": volatility(equity_curve),
        "sharpe": sharpe_ratio(equity_curve),
        "calmar": calmar_ratio(equity_curve),
        "turnover": float(turnover),
        "final_equity": float(equity_curve[-1]),
    }