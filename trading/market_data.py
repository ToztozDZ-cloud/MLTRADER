from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from trading.config import API_KEY, API_SECRET, LOOKBACK, TIMEFRAME_MINUTES


class MarketData:
    """Batched multi-symbol Alpaca market data for the dynamic universe scanner."""

    def __init__(self) -> None:
        self.client = StockHistoricalDataClient(API_KEY, API_SECRET)

    @staticmethod
    def _timeframe():
        return TimeFrame(TIMEFRAME_MINUTES, TimeFrameUnit.Minute)

    def get_recent_bars(self, symbols: str | list[str], limit: int = LOOKBACK):
        symbol_list = [symbols] if isinstance(symbols, str) else list(dict.fromkeys(symbols))
        if not symbol_list:
            return {}
        end = datetime.now(timezone.utc)
        # A multiplier allows for overnight/weekend closures and market holidays.
        start = end - timedelta(minutes=max(limit * TIMEFRAME_MINUTES * 4, 5 * 24 * 60))
        request = StockBarsRequest(
            symbol_or_symbols=symbol_list,
            timeframe=self._timeframe(),
            start=start,
            end=end,
            limit=limit,
            feed="iex",
        )
        response = self.client.get_stock_bars(request)
        frame = response.df
        if frame.empty:
            return {}
        if not isinstance(frame.index, pd.MultiIndex):
            symbol = symbol_list[0]
            return {symbol: frame.sort_index().tail(limit)}
        output = {}
        # Alpaca's multi-symbol bars index is (symbol, timestamp).
        names = list(frame.index.names)
        symbol_level = names.index("symbol") if "symbol" in names else 0
        for symbol in frame.index.get_level_values(symbol_level).unique():
            try:
                one = frame.xs(symbol, level=symbol_level).sort_index().tail(limit)
                if not one.empty:
                    output[str(symbol)] = one
            except KeyError:
                continue
        return output

    def get_recent_bars_batched(
        self, symbols: list[str], limit: int = LOOKBACK, batch_size: int = 100
    ) -> dict[str, pd.DataFrame]:
        result: dict[str, pd.DataFrame] = {}
        batch_size = max(1, int(batch_size))
        for offset in range(0, len(symbols), batch_size):
            batch = symbols[offset:offset + batch_size]
            try:
                result.update(self.get_recent_bars(batch, limit=limit))
            except Exception as exc:
                print(f"Market-data batch {offset // batch_size + 1} failed: {exc}")
        return result
