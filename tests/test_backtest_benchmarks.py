from __future__ import annotations

from typing import cast

import pandas as pd
import numpy as np

from agents.agent import Agent
from backtest.benchmarks import ReferencePolicy
from backtest.engine import BacktestEngine
from environment.costs import TradingCosts


def test_buy_and_hold_baseline_enters_once_and_holds_shares() -> None:
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 110.0, 120.0],
            "high": [100.0, 100.0, 120.0, 120.0],
            "low": [100.0, 100.0, 110.0, 120.0],
            "close": [100.0, 100.0, 120.0, 120.0],
            "volume": [1000.0] * 4,
        }
    )
    engine = BacktestEngine(
        lookback=2,
        initial_cash=1000.0,
        costs=TradingCosts(),
        max_position_fraction=1.0,
    )
    policy = ReferencePolicy("buy_and_hold", 2)

    result = engine.run(cast(Agent, policy), data, start_step=1)

    assert result["metrics"]["trade_count"] == 2


def test_buy_and_hold_ignores_floating_point_dust_rebalances() -> None:
    close = 100.0 + 0.1 * np.arange(128) + np.sin(np.arange(128) / 3.0)
    open_price = np.r_[close[0], close[:-1]]
    data = pd.DataFrame(
        {
            "open": open_price,
            "high": np.maximum(open_price, close) + 0.2,
            "low": np.minimum(open_price, close) - 0.2,
            "close": close,
            "volume": np.full(len(close), 1000.0),
        }
    )
    engine = BacktestEngine(lookback=2, initial_cash=1000.0, costs=TradingCosts())

    result = engine.run(
        cast(Agent, ReferencePolicy("buy_and_hold", 2)),
        data,
        start_step=1,
    )

    assert result["metrics"]["trade_count"] == 2
