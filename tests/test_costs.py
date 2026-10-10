from __future__ import annotations

import pytest

from environment.costs import TradingCosts
from environment.exchange import Exchange
from environment.portfolio import Portfolio


def test_spread_and_slippage_are_in_execution_price_not_cash_fee() -> None:
    costs = TradingCosts(commission_rate=0.001, spread_rate=0.002, slippage_rate=0.003)
    assert costs.execution_price(100.0, 1.0) == pytest.approx(100.4)
    assert costs.execution_price(100.0, -1.0) == pytest.approx(99.6)
    assert costs.total_cost(1000.0) == pytest.approx(1.0)


def test_exchange_deducts_commission_once_and_models_adverse_fill() -> None:
    costs = TradingCosts(commission_rate=0.001, spread_rate=0.002, slippage_rate=0.003)
    portfolio = Portfolio(initial_cash=10000.0)
    exchange = Exchange(portfolio=portfolio, costs=costs, max_leverage=1.0)
    result = exchange.execute_market_order(quantity=10.0, market_price=100.0)
    assert result.execution_price == pytest.approx(100.4)
    assert result.transaction_cost == pytest.approx(1.004)
    assert portfolio.cash == pytest.approx(10000.0 - 10 * 100.4 - 1.004)


def test_cost_rates_must_be_non_negative_and_finite() -> None:
    with pytest.raises(ValueError):
        TradingCosts(commission_rate=-0.01)
    with pytest.raises(ValueError):
        TradingCosts(slippage_rate=float("inf"))
