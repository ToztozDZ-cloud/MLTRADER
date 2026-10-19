from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from agents.agent import Agent
from environment.costs import TradingCosts
from training.evaluator import AgentEvaluator


class _AlwaysLong:
    fitness = float("-inf")

    def reset(self) -> None:
        return None

    def act(self, observation) -> float:
        return 1.0


def test_signal_executes_at_next_bar_open_and_liquidates_at_final_close() -> None:
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 110.0, 120.0],
            "high": [100.0, 100.0, 120.0, 120.0],
            "low": [100.0, 100.0, 110.0, 120.0],
            "close": [100.0, 100.0, 120.0, 120.0],
            "volume": [1000.0] * 4,
        }
    )
    evaluator = AgentEvaluator(
        lookback=2,
        initial_cash=1000.0,
        costs=TradingCosts(commission_rate=0.0, spread_rate=0.0, slippage_rate=0.0),
        max_leverage=1.0,
    )

    result = evaluator.evaluate(cast(Agent, _AlwaysLong()), data, start_step=1)

    assert result.trade_count == 2
    assert result.final_equity == pytest.approx(1000.0 * 120.0 / 110.0)
