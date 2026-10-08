from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Portfolio:
    initial_cash: float
    cash: float = 0.0
    position: float = 0.0
    average_entry_price: float = 0.0
    realized_pnl: float = 0.0

    def __post_init__(self) -> None:
        self.initial_cash = float(self.initial_cash)
        self.cash = self.initial_cash if self.cash == 0.0 else float(self.cash)

    def reset(self) -> None:
        self.cash = self.initial_cash
        self.position = 0.0
        self.average_entry_price = 0.0
        self.realized_pnl = 0.0

    def market_value(self, price: float) -> float:
        return self.position * float(price)

    def equity(self, price: float) -> float:
        return self.cash + self.market_value(price)

    def unrealized_pnl(self, price: float) -> float:
        if self.position == 0.0:
            return 0.0

        return (float(price) - self.average_entry_price) * self.position

    def apply_trade(
        self,
        quantity: float,
        price: float,
        transaction_cost: float = 0.0,
    ) -> None:
        quantity = float(quantity)
        price = float(price)
        transaction_cost = float(transaction_cost)

        if quantity == 0.0:
            return

        old_position = self.position
        new_position = old_position + quantity

        # Cash changes according to the executed trade.
        self.cash -= quantity * price
        self.cash -= transaction_cost

        # Opening or adding to a position.
        if old_position == 0.0 or (
            old_position > 0.0 and quantity > 0.0
        ) or (
            old_position < 0.0 and quantity < 0.0
        ):
            total_cost = (
                abs(old_position) * self.average_entry_price
                + abs(quantity) * price
            )

            total_quantity = abs(old_position) + abs(quantity)

            if total_quantity > 0:
                self.average_entry_price = total_cost / total_quantity

        # Closing/reducing/reversing a position.
        else:
            closing_quantity = min(abs(old_position), abs(quantity))

            if old_position > 0:
                self.realized_pnl += (
                    price - self.average_entry_price
                ) * closing_quantity
            else:
                self.realized_pnl += (
                    self.average_entry_price - price
                ) * closing_quantity

            if new_position == 0.0:
                self.average_entry_price = 0.0

            elif old_position * new_position < 0.0:
                # Position reversed. Remaining quantity starts a new position.
                self.average_entry_price = price

        self.position = new_position

    def snapshot(self, price: float) -> dict:
        return {
            "cash": self.cash,
            "position": self.position,
            "price": float(price),
            "market_value": self.market_value(price),
            "equity": self.equity(price),
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl(price),
        }