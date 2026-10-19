from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evolve import load_market_data


def _valid_daily_frame(rows: int = 110) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = np.arange(rows, dtype=float) + 100.0
    return pd.DataFrame(
        {
            "Date": dates[::-1].astype(str),
            " Open ": np.full(rows, 100.0),
            "HIGH": close[::-1] + 1.0,
            "Low": np.full(rows, 98.0),
            "Close": close[::-1],
            "Volume": np.full(rows, 1000.0),
        }
    )


def test_load_market_data_normalizes_sorts_and_preserves_instrument(tmp_path) -> None:
    path = tmp_path / "bars.csv"
    _valid_daily_frame().to_csv(path, index=False)

    cleaned = load_market_data(path, regular_session_only=False)

    assert list(cleaned.columns) == ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
    assert cleaned["timestamp"].is_monotonic_increasing
    assert cleaned["close"].iloc[0] == 100.0
    assert set(cleaned["symbol"]) == {"BARS"}
    assert np.isfinite(cleaned[["open", "high", "low", "close", "volume"]].to_numpy()).all()


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("Close", np.inf, "close must be finite"),
        ("Volume", -1.0, "volume must be non-negative"),
        ("HIGH", 90.0, "inconsistent OHLC"),
    ],
)
def test_load_market_data_rejects_invalid_rows(tmp_path, column, value, message) -> None:
    frame = _valid_daily_frame()
    frame.loc[0, column] = value
    path = tmp_path / "malformed.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        load_market_data(path, regular_session_only=False)


def test_load_market_data_rejects_too_few_valid_rows(tmp_path) -> None:
    frame = _valid_daily_frame(99)
    path = tmp_path / "too_short.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="at least 100 valid rows"):
        load_market_data(path, regular_session_only=False)


def test_load_market_data_requires_timestamps_for_chronological_use(tmp_path) -> None:
    frame = _valid_daily_frame().drop(columns="Date")
    path = tmp_path / "untimestamped.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="timestamp.*required"):
        load_market_data(path, regular_session_only=False)
