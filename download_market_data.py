from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

from environment.data import validate_ohlcv


DEFAULT_SYMBOLS = ("SPY", "QQQ", "AAPL", "MSFT")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download timestamped 5-minute US stock bars from Alpaca into data/raw/."
    )
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD, UTC).")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD, UTC); defaults to now.")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument(
        "--feed",
        choices=("iex", "sip"),
        default="iex",
        help="IEX is generally available on basic plans; SIP may require a subscription.",
    )
    args = parser.parse_args()

    load_dotenv()
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret or key.startswith("your_") or secret.startswith("your_"):
        parser.error(
            "Missing Alpaca credentials. Copy .env.example to .env and set "
            "APCA_API_KEY_ID and APCA_API_SECRET_KEY."
        )

    try:
        start = pd.Timestamp(args.start, tz="UTC").to_pydatetime()
        end = (
            pd.Timestamp(args.end, tz="UTC").to_pydatetime()
            if args.end
            else datetime.now(timezone.utc)
        )
    except (TypeError, ValueError) as exc:
        parser.error(f"Invalid date: {exc}")

    if start >= end:
        parser.error("--start must be earlier than --end")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = StockHistoricalDataClient(key, secret)
    feed = DataFeed.IEX if args.feed == "iex" else DataFeed.SIP

    request = StockBarsRequest(
        symbol_or_symbols=[s.upper() for s in args.symbols],
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=feed,
        adjustment="raw",
    )
    bars = client.get_stock_bars(request).df
    if bars.empty:
        print("Alpaca returned no bars. Check credentials, date range, feed access, and symbols.")
        return 1

    written = 0
    for symbol in [s.upper() for s in args.symbols]:
        try:
            frame = bars.xs(symbol, level="symbol").reset_index()
        except (KeyError, ValueError):
            print(f"{symbol}: no bars returned; skipped")
            continue

        completed_cutoff = min(
            pd.Timestamp(end).floor("5min"),
            pd.Timestamp.now(tz="UTC").floor("5min"),
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.loc[frame["timestamp"] < completed_cutoff]
        try:
            frame = validate_ohlcv(
                frame,
                source=f"Alpaca {symbol} download",
                timezone="America/New_York",
                regular_session_only=True,
                timeframe="5m",
                require_timestamp=True,
            )
        except ValueError as exc:
            print(f"{symbol}: invalid downloaded bars; skipped: {exc}")
            continue
        if frame.empty:
            print(f"{symbol}: no completed regular-session bars; skipped")
            continue
        frame["timestamp"] = frame["timestamp"].dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        path = output_dir / f"{symbol}_5m.csv"
        frame.to_csv(path, index=False)
        print(f"{symbol}: saved {len(frame):,} bars -> {path}")
        written += 1

    print(f"Done. Wrote {written} CSV file(s) to {output_dir.resolve()}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
