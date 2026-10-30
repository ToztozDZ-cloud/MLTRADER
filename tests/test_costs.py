from __future__ import annotations

import pytest

from environment.costs import TradingCosts
from environment.exchange import Exchange
from environment.portfolio import Portfolio
from backtest.engine import BacktestEngine
from backtest.metrics import max_drawdown
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


def test_max_drawdown_empty_and_invalid_curves_keep_defined_results() -> None:
    assert max_drawdown([]) == 0.0
    assert max_drawdown([100.0, 0.0]) == 1.0
    assert max_drawdown([100.0, float("nan")]) == 1.0


def test_execution_quantity_must_be_finite() -> None:
    costs = TradingCosts()
    with pytest.raises(ValueError, match="quantity must be finite"):
        costs.execution_price(100.0, float("inf"))
    with pytest.raises(ValueError, match="quantity must be finite"):
        costs.execution_price(100.0, float("nan"))


def test_trade_logger_writes_utf8_csv_and_appends_without_repeating_header(tmp_path) -> None:
    from trading.trade_logger import TradeLogger

    path = tmp_path / "trades.csv"
    logger = TradeLogger(str(path))
    logger.log("ÉQ", 100.25, 0.5, 1.0, 0.0, 10_000.0, "order-1")

    first = path.read_text(encoding="utf-8")
    assert "timestamp,symbol,price,agent_output" in first
    assert "ÉQ" in first

    logger.log("ÉQ", 101.25, 0.25, 0.5, 1.0, 10_100.0, "order-2")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert sum(line.startswith("timestamp,symbol") for line in lines) == 1


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


def test_process_worker_entry_point_is_module_level_and_invokes_evaluator() -> None:
    class Evaluator:
        def evaluate(self, agent, market_data):
            return (agent, market_data)

    from training.parallel import _evaluate_one

    agent = object()
    market_data = object()
    returned_agent, result = _evaluate_one(agent, Evaluator, market_data)

    assert returned_agent is agent
    assert result == (agent, market_data)
    assert _evaluate_one.__module__ == "training.parallel"


def test_intraday_annualization_matches_configured_timeframe() -> None:
    assert periods_per_year_for_timeframe("5m") == pytest.approx(19_656.0)
    assert periods_per_year_for_timeframe("1h") == pytest.approx(1_638.0)
    assert periods_per_year_for_timeframe("1d") == pytest.approx(252.0)
    with pytest.raises(ValueError, match="Unsupported annualization timeframe"):
        periods_per_year_for_timeframe("weekly")
