from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from environment.data import validate_ohlcv
from trading.config import LOOKBACK, TIMEFRAME_MINUTES, require_paper_credentials


class MarketData:
    """Batched Alpaca market data. Each request is deliberately kept small."""

    def __init__(self) -> None:
        api_key, api_secret = require_paper_credentials()
        self.client = StockHistoricalDataClient(api_key, api_secret)

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
            return {symbol_list[0]: self._completed_bars(frame).tail(limit)}

        names = list(frame.index.names)
        symbol_level = names.index("symbol") if "symbol" in names else 0
        output: dict[str, pd.DataFrame] = {}
        for symbol in frame.index.get_level_values(symbol_level).unique():
            try:
                one = self._completed_bars(
                    frame.xs(symbol, level=symbol_level)
                ).tail(limit)
                if not one.empty:
                    output[str(symbol)] = one
            except KeyError:
                continue
        return output

    @staticmethod
    def _completed_bars(frame: pd.DataFrame) -> pd.DataFrame:
        """Exclude the currently forming five-minute candle."""
        result = frame.sort_index().copy()
        if result.empty:
            return result
        index = pd.to_datetime(result.index, utc=True, errors="coerce")
        cutoff = pd.Timestamp.now(tz="UTC").floor(f"{TIMEFRAME_MINUTES}min")
        result = result.loc[index < cutoff].copy()
        result["timestamp"] = index[index < cutoff]
        checked = validate_ohlcv(
            result,
            source="Alpaca paper market data",
            timezone="America/New_York",
            regular_session_only=True,
            timeframe=f"{TIMEFRAME_MINUTES}m",
            require_timestamp=True,
        )
        return checked.set_index("timestamp")

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
