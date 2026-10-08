from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

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