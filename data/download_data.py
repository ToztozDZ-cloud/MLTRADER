from __future__ import annotations

from pathlib import Path

import yfinance as yf


SYMBOLS = [
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
]


def download_symbol(
    symbol: str,
    start: str = "2005-01-01",
) -> None:
    output_directory = Path(
        "data/raw"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Downloading {symbol}..."
    )

    data = yf.download(
        symbol,
        start=start,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        print(
            f"No data received for {symbol}"
        )
        return

    # Flatten yfinance MultiIndex columns.
    if hasattr(data.columns, "levels"):
        data.columns = [
            column[0].lower()
            if isinstance(column, tuple)
            else str(column).lower()
            for column in data.columns
        ]
    else:
        data.columns = [
            str(column).lower()
            for column in data.columns
        ]

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    available = [
        column
        for column in required
        if column in data.columns
    ]

    data = data[available].copy()

    data.reset_index(
        inplace=True
    )

    data["Date"] = data["Date"].astype(str)

    path = (
        output_directory
        / f"{symbol}.csv"
    )

    data.to_csv(
        path,
        index=False,
    )

    print(
        f"Saved {len(data)} rows -> {path}"
    )


def main() -> None:
    for symbol in SYMBOLS:
        download_symbol(symbol)


if __name__ == "__main__":
    main()