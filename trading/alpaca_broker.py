from __future__ import annotations

<<<<<<< HEAD
=======
<<<<<<< HEAD
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
=======
>>>>>>> 7ce9a3260bf1e51b5cc30c7b002019690ed69fb8
from decimal import Decimal, ROUND_DOWN

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, AssetStatus, AssetClass, QueryOrderStatus
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest, GetOrdersRequest

from trading.config import API_KEY, API_SECRET, PAPER


class PaperBroker:
    """Alpaca broker wrapper that refuses to operate outside paper trading."""

    def __init__(self) -> None:
        if not PAPER:
            raise RuntimeError("This program is paper-trading only.")
        self.client = TradingClient(API_KEY, API_SECRET, paper=True)
        self._assets_cache = None

    def account(self):
        return self.client.get_account()

    def market_is_open(self) -> bool:
        return bool(self.client.get_clock().is_open)

    def positions(self):
        return self.client.get_all_positions()

    def open_order_symbols(self) -> set[str]:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
        return {order.symbol for order in self.client.get_orders(filter=request)}

    def position(self, symbol: str):
        for position in self.positions():
            if position.symbol == symbol:
                return position
        return None

    def assets(self, refresh: bool = False):
        if self._assets_cache is None or refresh:
            request = GetAssetsRequest(
                status=AssetStatus.ACTIVE,
                asset_class=AssetClass.US_EQUITY,
            )
            assets = self.client.get_all_assets(request)
            self._assets_cache = {
                asset.symbol: asset for asset in assets
                if getattr(asset, "tradable", False)
                and getattr(asset, "status", None) == AssetStatus.ACTIVE
                and asset.symbol.isascii()
                and asset.symbol.replace(".", "").replace("-", "").isalnum()
            }
        return self._assets_cache

    def submit_delta(self, symbol: str, delta_quantity: float, *, allow_short: bool):
        """Submit an order to move a symbol toward its target quantity.

        Reversals are flattened first; the next scan may open the opposite side.
        """
        if abs(delta_quantity) < 1e-6:
            return None
        asset = self.assets().get(symbol)
        if asset is None or not getattr(asset, "tradable", False):
            print(f"SKIP {symbol}: asset is not currently tradable.")
            return None

        position = self.position(symbol)
        current_qty = float(position.qty) if position is not None else 0.0
        delta = float(delta_quantity)
        # Don't submit an order that crosses through zero in one transaction.
        if current_qty != 0 and current_qty * (current_qty + delta) < 0:
            delta = -current_qty
        if delta < 0 and current_qty <= 0 and not allow_short:
            print(f"SKIP {symbol}: shorting disabled.")
            return None
        if delta < 0 and current_qty <= 0 and not getattr(asset, "shortable", False):
            print(f"SKIP {symbol}: asset is not shortable.")
            return None

        fractional = bool(getattr(asset, "fractionable", False))
        qty = Decimal(str(abs(delta)))
        if not fractional:
            qty = qty.quantize(Decimal("1"), rounding=ROUND_DOWN)
        else:
            qty = qty.quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
        if qty <= 0:
            return None

        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        return self.client.submit_order(order_data=order)

    def close_position(self, symbol: str):
        if self.position(symbol) is None:
            return None
        return self.client.close_position(symbol)
<<<<<<< HEAD
=======
>>>>>>> origin/master
>>>>>>> 7ce9a3260bf1e51b5cc30c7b002019690ed69fb8
