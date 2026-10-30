from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from environment.compat import suppress_dependency_warnings

suppress_dependency_warnings()

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

from environment.data import validate_ohlcv
from trading.config import resolve_market_data_feed


DEFAULT_SYMBOLS = ("SPY", "QQQ", "AAPL", "MSFT")


def normalize_symbols(symbols: list[str]) -> list[str]:
    return list(
        dict.fromkeys(symbol.strip().upper() for symbol in symbols if symbol.strip())
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        default=None,
        help="Override the feed in config/live.yaml (IEX is generally available; SIP may require a subscription).",
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

    symbols = normalize_symbols(args.symbols)
    if not symbols:
        parser.error("--symbols must contain at least one non-empty symbol.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = StockHistoricalDataClient(key, secret)
    feed_name, feed = resolve_market_data_feed(args.feed)

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=feed,
        adjustment="raw",
    )
    report = {
        "provider": "alpaca",
        "feed": feed_name,
        "timeframe": "5m",
        "adjustment": "raw",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "datasets": [],
        "failures": [],
    }

    def save_report() -> Path:
        report_path = output_dir / "market_data_manifest.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report_path

    try:
        bars = client.get_stock_bars(request).df
    except Exception as exc:
        report["failures"].extend(
            {
                "symbol": symbol,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            for symbol in symbols
        )
        manifest_path = save_report()
        print(f"Market-data manifest: {manifest_path.resolve()}")
        print(f"Market-data report: {json.dumps(report, sort_keys=True)}")
        return 1

    if bars.empty:
        report["failures"].extend(
            {
                "symbol": symbol,
                "error_type": "NoDataReturned",
                "message": "Alpaca returned no bars for this symbol.",
            }
            for symbol in symbols
        )
        manifest_path = save_report()
        print(f"Market-data manifest: {manifest_path.resolve()}")
        print(f"Market-data failures: {json.dumps(report['failures'], sort_keys=True)}")
        return 1

    written = 0
    for symbol in symbols:
        try:
            frame = bars.xs(symbol, level="symbol").reset_index()
        except (KeyError, ValueError):
            report["failures"].append(
                {
                    "symbol": symbol,
                    "error_type": "NoDataReturned",
                    "message": "Alpaca returned no bars for this symbol.",
                }
            )
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
            report["failures"].append(
                {
                    "symbol": symbol,
                    "error_type": "InvalidBars",
                    "message": str(exc),
                }
            )
            continue
        if frame.empty:
            report["failures"].append(
                {
                    "symbol": symbol,
                    "error_type": "NoCompletedBars",
                    "message": "No completed regular-session bars were returned.",
                }
            )
            continue
        frame["timestamp"] = frame["timestamp"].dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        path = output_dir / f"{symbol}_5m.csv"
        frame.to_csv(path, index=False)
        digest = file_sha256(path)
        report["datasets"].append(
            {
                "symbol": symbol,
                "path": path.name,
                "bars": len(frame),
                "sha256": digest,
                "identity": f"alpaca:{feed_name}:{symbol}:5m:{digest}",
            }
        )
        print(f"{symbol}: saved {len(frame):,} bars -> {path}")
        written += 1

    manifest_path = save_report()
    print(f"Done. Wrote {written} CSV file(s) to {output_dir.resolve()}")
    print(f"Market-data manifest: {manifest_path.resolve()}")
    if report["failures"]:
        print(f"Market-data failures: {json.dumps(report['failures'], sort_keys=True)}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
