from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TradingCosts:
    commission_rate: float = 0.0005
    spread_rate: float = 0.0002
    slippage_rate: float = 0.0005

    def commission(self, notional: float) -> float:
        return abs(float(notional)) * self.commission_rate

    def spread_cost(self, notional: float) -> float:
        return abs(float(notional)) * self.spread_rate

    def slippage_cost(self, notional: float) -> float:
        return abs(float(notional)) * self.slippage_rate

    def total_cost(self, notional: float) -> float:
        return (
            self.commission(notional)
            + self.spread_cost(notional)
            + self.slippage_cost(notional)
        )

    def execution_price(
        self,
        market_price: float,
        quantity: float,
    ) -> float:
        """
        Approximate adverse execution caused by spread + slippage.
        """
        market_price = float(market_price)

        direction = 1.0 if quantity > 0 else -1.0

        adverse_rate = (
            self.spread_rate
            + self.slippage_rate
        )

        return market_price * (
            1.0 + direction * adverse_rate
        )