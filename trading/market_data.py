from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from environment.compat import suppress_dependency_warnings

suppress_dependency_warnings()

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from environment.data import validate_ohlcv
from trading.config import (
    get_lookback,
    get_timeframe_minutes,
    require_paper_credentials,
    resolve_market_data_feed,
)


@dataclass(frozen=True)
class MarketDataBatchFailure:
    batch_number: int
    symbols: tuple[str, ...]
    error_type: str
    message: str


@dataclass
class MarketDataBatchResult:
    bars: dict[str, pd.DataFrame]
    provider: str
    feed: str
    data_identity: dict[str, str]
    failures: list[MarketDataBatchFailure]


class MarketData:
    """Batched Alpaca market data. Each request is deliberately kept small."""

    def __init__(self) -> None:
        api_key, api_secret = require_paper_credentials()
        self.client = StockHistoricalDataClient(api_key, api_secret)
        self.feed_name, self.feed = resolve_market_data_feed()

    @staticmethod
    def _timeframe():
        return TimeFrame(get_timeframe_minutes(), TimeFrameUnit.Minute)

    def get_recent_bars(self, symbols: str | list[str], limit: int | None = None):
        if limit is None:
            limit = get_lookback()
        requested_symbols = [symbols] if isinstance(symbols, str) else symbols
        symbol_list = list(
            dict.fromkeys(str(symbol).strip().upper() for symbol in requested_symbols)
        )
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
            feed=self.feed,
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
        cutoff = pd.Timestamp.now(tz="UTC").floor(f"{get_timeframe_minutes()}min")
        result = result.loc[index < cutoff].copy()
        result["timestamp"] = index[index < cutoff]
        checked = validate_ohlcv(
            result,
            source="Alpaca paper market data",
            timezone="America/New_York",
            regular_session_only=True,
            timeframe=f"{get_timeframe_minutes()}m",
            require_timestamp=True,
            on_bar_interval_error="warn",
        )
        return checked.set_index("timestamp")

    def get_recent_bars_batched(
        self, symbols: list[str], limit: int | None = None, batch_size: int = 50
    ) -> MarketDataBatchResult:
        if limit is None:
            limit = get_lookback()
        unique_symbols = list(
            dict.fromkeys(str(symbol).strip().upper() for symbol in symbols)
        )
        bars_by_symbol: dict[str, pd.DataFrame] = {}
        failures: list[MarketDataBatchFailure] = []
        batch_size = max(1, min(int(batch_size), 100))
        for offset in range(0, len(unique_symbols), batch_size):
            batch = unique_symbols[offset:offset + batch_size]
            batch_number = offset // batch_size + 1
            try:
                batch_bars = self.get_recent_bars(batch, limit=limit)
            except Exception as exc:
                failures.append(
                    MarketDataBatchFailure(
                        batch_number=batch_number,
                        symbols=tuple(batch),
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                continue

            bars_by_symbol.update(batch_bars)
            for symbol in batch:
                if symbol not in batch_bars:
                    failures.append(
                        MarketDataBatchFailure(
                            batch_number=batch_number,
                            symbols=(symbol,),
                            error_type="NoDataReturned",
                            message="Alpaca returned no completed bars for this symbol.",
                        )
                    )

        identities = {
            symbol: hashlib.sha256(
                pd.util.hash_pandas_object(frame, index=True).values.tobytes()
            ).hexdigest()
            for symbol, frame in bars_by_symbol.items()
        }
        return MarketDataBatchResult(
            bars=bars_by_symbol,
            provider="alpaca",
            feed=self.feed_name,
            data_identity=identities,
            failures=failures,
        )
