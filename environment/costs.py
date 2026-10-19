from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TradingCosts:
    """Simple adverse-fill model.

    spread_rate is the full quoted spread as a fraction of mid-price; each
    fill pays half the spread plus the configured adverse slippage. Commission
    is deducted from cash separately. Spread/slippage must never be deducted
    a second time as explicit cash fees.
    """
    commission_rate: float = 0.0005
    spread_rate: float = 0.0002
    slippage_rate: float = 0.0005

    def __post_init__(self) -> None:
        for name in ("commission_rate", "spread_rate", "slippage_rate"):
            value = float(getattr(self, name))
            if value < 0 or not value < float("inf"):
                raise ValueError(f"{name} must be finite and non-negative.")

    def commission(self, notional: float) -> float:
        return abs(float(notional)) * self.commission_rate

    def spread_cost(self, notional: float) -> float:
        return abs(float(notional)) * self.spread_rate

    def slippage_cost(self, notional: float) -> float:
        return abs(float(notional)) * self.slippage_rate

    def total_cost(self, notional: float) -> float:
        """Explicit cash fee only; spread/slippage are in execution_price."""
        return self.commission(notional)

    def estimated_execution_cost(self, notional: float) -> float:
        """Estimate all-in friction; spread/slippage are already reflected in fills."""
        return abs(float(notional)) * (
            self.commission_rate + self.spread_rate / 2.0 + self.slippage_rate
        )

    def execution_price(self, market_price: float, quantity: float) -> float:
        """Apply half-spread and adverse slippage exactly once."""
        market_price = float(market_price)
        quantity = float(quantity)
        if not market_price > 0 or not market_price < float("inf"):
            raise ValueError("market_price must be finite and positive.")
        if quantity == 0:
            return market_price
        direction = 1.0 if quantity > 0 else -1.0
        adverse_rate = self.spread_rate / 2.0 + self.slippage_rate
        return market_price * (1.0 + direction * adverse_rate)
