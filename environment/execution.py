from __future__ import annotations

from dataclasses import dataclass

from .exchange import Exchange, OrderResult


@dataclass
class ExecutionEngine:
    exchange: Exchange

    def execute_target_position(
        self,
        target_position: float,
        market_price: float,
    ) -> OrderResult:
        """
        Convert a target position into an order quantity.
        """
        current_position = self.exchange.portfolio.position

        target_position = float(target_position)

        quantity = target_position - current_position

        return self.exchange.execute_market_order(
            quantity=quantity,
            market_price=market_price,
        )