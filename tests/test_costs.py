from __future__ import annotations

import pytest

from environment.costs import TradingCosts
from environment.exchange import Exchange
from environment.portfolio import Portfolio
from backtest.engine import BacktestEngine
from training.fitness import periods_per_year_for_timeframe


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


def test_hand_calculated_round_trip_cash_realized_pnl_and_costs() -> None:
    costs = TradingCosts(commission_rate=0.001, spread_rate=0.002, slippage_rate=0.003)
    portfolio = Portfolio(initial_cash=10_000.0)
    exchange = Exchange(portfolio=portfolio, costs=costs, max_leverage=1.0)

    buy = exchange.execute_market_order(quantity=10.0, market_price=100.0)
    assert buy.execution_price == pytest.approx(100.4)
    assert buy.transaction_cost == pytest.approx(1.004)
    assert portfolio.cash == pytest.approx(8_994.996)
    assert portfolio.equity(100.0) == pytest.approx(9_994.996)
    assert portfolio.average_entry_price == pytest.approx(100.4)

    sell = exchange.execute_market_order(quantity=-10.0, market_price=110.0)
    assert sell.execution_price == pytest.approx(109.56)
    assert sell.transaction_cost == pytest.approx(1.0956)
    assert portfolio.position == pytest.approx(0.0)
    assert portfolio.realized_pnl == pytest.approx(91.6)
    assert portfolio.cash == pytest.approx(10_089.5004)
    assert portfolio.equity(110.0) == pytest.approx(10_089.5004)


def test_unsupported_same_bar_execution_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="next-bar-open execution"):
        BacktestEngine(execute_on_next_bar=False)


def test_intraday_annualization_matches_configured_timeframe() -> None:
    assert periods_per_year_for_timeframe("5m") == pytest.approx(19_656.0)
    assert periods_per_year_for_timeframe("1h") == pytest.approx(1_638.0)
    assert periods_per_year_for_timeframe("1d") == pytest.approx(252.0)
    with pytest.raises(ValueError, match="Unsupported annualization timeframe"):
        periods_per_year_for_timeframe("weekly")
