from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
<<<<<<< HEAD

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import (
    TimeFrame,
    TimeFrameUnit,
)

from trading.config import (
    API_KEY,
    API_SECRET,
    SYMBOL,
    TIMEFRAME_MINUTES,
)


class MarketData:

    def __init__(self) -> None:

        self.client = (
            StockHistoricalDataClient(
                API_KEY,
                API_SECRET,
            )
        )

    def get_recent_bars(
        self,
        limit: int = 100,
    ) -> pd.DataFrame:

        end = datetime.now(
            timezone.utc,
        )

        # Request enough history to obtain
        # the requested number of bars.

        minutes = (
            limit
            * TIMEFRAME_MINUTES
            * 2
        )

        start = (
            end
            - timedelta(
                minutes=minutes,
            )
        )

        request = StockBarsRequest(
            symbol_or_symbols=SYMBOL,
            timeframe=TimeFrame(
                TIMEFRAME_MINUTES,
                TimeFrameUnit.Minute,
            ),
            start=start,
            end=end,
            limit=limit,
        )

        bars = self.client.get_stock_bars(
            request,
        )

        df = bars.df

        if df.empty:
            raise RuntimeError(
                "No market data received."
            )

        # If Alpaca returns a multi-index,
        # remove the symbol level.

        if isinstance(
            df.index,
            pd.MultiIndex,
        ):

            df = (
                df.reset_index(
                    level="symbol",
                    drop=True,
                )
            )

        return df.sort_index()
=======
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from trading.config import API_KEY, API_SECRET, LOOKBACK, TIMEFRAME_MINUTES


class MarketData:
    """Batched Alpaca market data. Each request is deliberately kept small."""

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
        # 5 calendar days includes weekends/market closures and comfortably
        # covers a 64-bar five-minute window under normal market conditions.
        start = end - timedelta(days=5)
        request = StockBarsRequest(
            symbol_or_symbols=symbol_list,
            timeframe=self._timeframe(),
            start=start,
            end=end,
            limit=min(10000, max(int(limit), int(limit) * len(symbol_list))),
            feed="iex",
        )
        response = self.client.get_stock_bars(request)
        frame = response.df
        if frame.empty:
            return {}

        if not isinstance(frame.index, pd.MultiIndex):
            return {symbol_list[0]: frame.sort_index().tail(limit)}

        names = list(frame.index.names)
        symbol_level = names.index("symbol") if "symbol" in names else 0
        output: dict[str, pd.DataFrame] = {}
        for symbol in frame.index.get_level_values(symbol_level).unique():
            try:
                one = frame.xs(symbol, level=symbol_level).sort_index().tail(limit)
                if not one.empty:
                    output[str(symbol)] = one
            except KeyError:
                continue
        return output

    def get_recent_bars_batched(
        self, symbols: list[str], limit: int = LOOKBACK, batch_size: int = 50
    ) -> dict[str, pd.DataFrame]:
        result: dict[str, pd.DataFrame] = {}
        batch_size = max(1, min(int(batch_size), 100))
        for offset in range(0, len(symbols), batch_size):
            batch = symbols[offset:offset + batch_size]
            try:
                result.update(self.get_recent_bars(batch, limit=limit))
            except Exception as exc:
                print(f"Market-data batch {offset // batch_size + 1} failed: {exc}")
        return result
>>>>>>> origin/master
