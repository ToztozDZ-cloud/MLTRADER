from __future__ import annotations

from decimal import Decimal

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
)
from alpaca.trading.requests import (
    MarketOrderRequest,
)

from trading.config import (
    API_KEY,
    API_SECRET,
    PAPER,
    SYMBOL,
)


class PaperBroker:

    def __init__(self) -> None:

        if not PAPER:
            raise RuntimeError(
                "LIVE TRADING IS DISABLED. "
                "This program is paper-trading only."
            )

        self.client = TradingClient(
            API_KEY,
            API_SECRET,
            paper=True,
        )

    def account(self):

        return self.client.get_account()

    def positions(self):

        return self.client.get_all_positions()

    def position(
        self,
    ):

        positions = self.positions()

        for position in positions:

            if position.symbol == SYMBOL:
                return position

        return None

    def close_position(
        self,
    ):

        position = self.position()

        if position is None:
            return None

        return self.client.close_position(
            SYMBOL,
        )

    def buy(
        self,
        quantity: float,
    ):

        if quantity <= 0:
            return None

        order = MarketOrderRequest(
            symbol=SYMBOL,
            qty=Decimal(
                str(quantity)
            ),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )

        return self.client.submit_order(
            order_data=order,
        )

    def sell(
        self,
        quantity: float,
    ):

        if quantity <= 0:
            return None

        order = MarketOrderRequest(
            symbol=SYMBOL,
            qty=Decimal(
                str(quantity)
            ),
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )

        return self.client.submit_order(
            order_data=order,
        )