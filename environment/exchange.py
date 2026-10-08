from __future__ import annotations

from dataclasses import dataclass

from .costs import TradingCosts
from .portfolio import Portfolio


@dataclass
class OrderResult:
    requested_quantity: float
    executed_quantity: float
    execution_price: float
    notional: float
    transaction_cost: float


class Exchange:
    def __init__(
        self,
        portfolio: Portfolio,
        costs: TradingCosts,
        allow_short: bool = False,
        max_leverage: float = 1.0,
    ) -> None:
        self.portfolio = portfolio
        self.costs = costs
        self.allow_short = allow_short
        self.max_leverage = float(max_leverage)

    def execute_market_order(
        self,
        quantity: float,
        market_price: float,
    ) -> OrderResult:
        requested_quantity = float(quantity)
        market_price = float(market_price)

        if requested_quantity == 0.0:
            return self._empty_result(
                requested_quantity,
                market_price,
            )

        quantity = requested_quantity

        current_position = self.portfolio.position

        # No short selling.
        if not self.allow_short:
            if current_position + quantity < 0.0:
                quantity = -current_position

        # Limit gross exposure.
        equity = self.portfolio.equity(
            market_price
        )

        max_notional = max(
            0.0,
            equity * self.max_leverage,
        )

        target_position = (
            current_position + quantity
        )

        target_notional = abs(
            target_position * market_price
        )

        if target_notional > max_notional:
            allowed_position = (
                max_notional / market_price
                if market_price > 0
                else 0.0
            )

            if target_position >= 0:
                target_position = min(
                    target_position,
                    allowed_position,
                )
            else:
                target_position = max(
                    target_position,
                    -allowed_position,
                )

            quantity = (
                target_position
                - current_position
            )

        if quantity == 0.0:
            return self._empty_result(
                requested_quantity,
                market_price,
            )

        execution_price = (
            self.costs.execution_price(
                market_price,
                quantity,
            )
        )

        notional = quantity * execution_price

        transaction_cost = (
            self.costs.total_cost(
                notional
            )
        )

        # Prevent purchases that exceed available cash.
        if quantity > 0:
            total_required = (
                notional + transaction_cost
            )

            if total_required > self.portfolio.cash:
                affordable = (
                    self.portfolio.cash
                    / (
                        execution_price
                        * (
                            1.0
                            + self.costs.commission_rate
                            + self.costs.spread_rate
                            + self.costs.slippage_rate
                        )
                    )
                )

                quantity = min(
                    quantity,
                    max(0.0, affordable),
                )

                if quantity <= 0.0:
                    return self._empty_result(
                        requested_quantity,
                        market_price,
                    )

                execution_price = (
                    self.costs.execution_price(
                        market_price,
                        quantity,
                    )
                )

                notional = (
                    quantity * execution_price
                )

                transaction_cost = (
                    self.costs.total_cost(
                        notional
                    )
                )

        self.portfolio.apply_trade(
            quantity=quantity,
            price=execution_price,
            transaction_cost=transaction_cost,
        )

        return OrderResult(
            requested_quantity=requested_quantity,
            executed_quantity=quantity,
            execution_price=execution_price,
            notional=notional,
            transaction_cost=transaction_cost,
        )

    @staticmethod
    def _empty_result(
        requested_quantity: float,
        market_price: float,
    ) -> OrderResult:
        return OrderResult(
            requested_quantity=requested_quantity,
            executed_quantity=0.0,
            execution_price=market_price,
            notional=0.0,
            transaction_cost=0.0,
        )