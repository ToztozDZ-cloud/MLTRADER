from __future__ import annotations

from decimal import Decimal, ROUND_DOWN
import math
from typing import Any

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, AssetStatus, AssetClass, QueryOrderStatus
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest, GetOrdersRequest

from trading.config import PAPER, require_paper_credentials


class PaperBroker:
    """Alpaca broker wrapper that refuses to operate outside paper trading."""

    def __init__(self) -> None:
        if not PAPER:
            raise RuntimeError("This program is paper-trading only.")
        api_key, api_secret = require_paper_credentials()
        self.client = TradingClient(api_key, api_secret, paper=True)
        self._assets_cache: dict[str, Any] | None = None

    def account(self):
        return self.client.get_account()

    def clock(self):
        """Return Alpaca's exchange clock, including next close/open."""
        return self.client.get_clock()

    def market_is_open(self) -> bool:
        return bool(self.clock().is_open)

    def positions(self):
        return self.client.get_all_positions()

    def open_orders(self):
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
        return list(self.client.get_orders(filter=request))

    def open_order_symbols(self) -> set[str]:
        return {order.symbol for order in self.open_orders()}

    def cancel_open_orders(self):
        """Request cancellation of all open orders; fills may still race cancellation."""
        return self.client.cancel_orders()

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

    def submit_delta(
        self,
        symbol: str,
        delta_quantity: float,
        *,
        allow_short: bool,
        client_order_id: str,
        price: float,
        max_position_fraction: float,
        max_gross_exposure: float,
        max_positions: int,
        reference_prices: dict[str, float] | None = None,
    ):
        """Submit an order to move a symbol toward its target quantity.

        Reversals are flattened first; the next scan may open the opposite side.
        """
        delta = float(delta_quantity)
        price = float(price)
        if not math.isfinite(delta) or not math.isfinite(price) or price <= 0.0:
            raise ValueError("Order quantity and reference price must be finite; price must be positive")
        if abs(delta) < 1e-6:
            return None
        if not 0.0 < max_position_fraction <= 1.0:
            raise ValueError("max_position_fraction must be in (0, 1]")
        if not 0.0 < max_gross_exposure <= 1.0:
            raise ValueError("max_gross_exposure must be in (0, 1]")
        if max_positions < 1 or not client_order_id:
            raise ValueError("max_positions and client_order_id are required")
        asset = self.assets().get(symbol)
        if asset is None or not getattr(asset, "tradable", False):
            print(f"SKIP {symbol}: asset is not currently tradable.")
            return None

        positions = self.positions()
        position_by_symbol = {item.symbol: item for item in positions}
        open_orders = self.open_orders()
        if any(order.symbol == symbol for order in open_orders):
            print(f"SKIP {symbol}: unresolved order already exists.")
            return None
        position = position_by_symbol.get(symbol)
        current_qty = float(position.qty) if position is not None else 0.0
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
        signed_order_qty = float(qty) if delta > 0 else -float(qty)
        account = self.account()
        equity = float(account.equity)
        if not math.isfinite(equity) or equity <= 0.0:
            raise RuntimeError("Broker account equity is invalid; refusing to submit order")
        projected_qty = current_qty + signed_order_qty
        if not allow_short and projected_qty < -1e-8:
            raise RuntimeError("Order would create a short position")
        target_notional = abs(projected_qty * price)

        marks = dict(reference_prices or {})
        marks[symbol] = price
        exposure_by_symbol = {
            item.symbol: abs(float(item.market_value))
            for item in positions
        }
        for pending in open_orders:
            pending_symbol = pending.symbol
            current = position_by_symbol.get(pending_symbol)
            current_position_qty = float(current.qty) if current is not None else 0.0
            remaining_qty = max(
                0.0,
                float(pending.qty) - float(getattr(pending, "filled_qty", 0.0) or 0.0),
            )
            pending_delta = remaining_qty if pending.side == OrderSide.BUY else -remaining_qty
            mark = marks.get(pending_symbol)
            if mark is None and current is not None:
                mark = abs(float(current.market_value) / current_position_qty) if current_position_qty else float(current.avg_entry_price)
            if mark is None or not math.isfinite(float(mark)) or float(mark) <= 0.0:
                exposure_by_symbol[pending_symbol] = max(
                    exposure_by_symbol.get(pending_symbol, 0.0),
                    equity * max_position_fraction,
                )
            else:
                pending_value = abs((current_position_qty + pending_delta) * float(mark))
                exposure_by_symbol[pending_symbol] = max(
                    exposure_by_symbol.get(pending_symbol, 0.0), pending_value
                )

        gross_before = sum(exposure_by_symbol.values())
        current_symbol_notional = exposure_by_symbol.get(symbol, 0.0)
        current_position_count = sum(value > 1e-6 for value in exposure_by_symbol.values())
        exposure_by_symbol[symbol] = target_notional
        gross_exposure = sum(exposure_by_symbol.values())
        open_position_count = sum(value > 1e-6 for value in exposure_by_symbol.values())
        if (
            target_notional > equity * max_position_fraction + 1e-6
            and target_notional > current_symbol_notional + 1e-6
        ):
            raise RuntimeError("Order would exceed the per-symbol exposure limit")
        if (
            gross_exposure > equity * max_gross_exposure + 1e-6
            and gross_exposure > gross_before + 1e-6
        ):
            raise RuntimeError("Order would exceed the gross exposure limit")
        if open_position_count > max_positions and open_position_count > current_position_count:
            raise RuntimeError("Order would exceed the maximum open position count")

        buying_power = float(getattr(account, "buying_power", equity))
        if not math.isfinite(buying_power) or buying_power < 0.0:
            raise RuntimeError("Broker buying power is invalid; refusing to submit order")
        if side == OrderSide.BUY and float(qty) * price > buying_power + 1e-6:
            raise RuntimeError("Order exceeds broker buying power")

        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        return self.client.submit_order(order_data=order)

    def close_position(self, symbol: str):
        if self.position(symbol) is None:
            return None
        return self.client.close_position(symbol)
