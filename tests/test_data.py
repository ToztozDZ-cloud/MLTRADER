from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evolve import load_market_data


def test_load_market_data_sorts_and_filters_invalid_rows(tmp_path) -> None:
    rows = 110
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    frame = pd.DataFrame(
        {
            "Date": dates[::-1].astype(str),
            " Open ": np.full(rows, 100.0),
            "HIGH": np.full(rows, 102.0),
            "Low": np.full(rows, 98.0),
            "Close": np.arange(rows, dtype=float) + 100.0,
            "Volume": np.full(rows, 1000.0),
        }
    )
    frame.loc[0, "Close"] = np.inf
    frame.loc[1, "Volume"] = -1.0
    frame.loc[2, "High"] = 90.0

    path = tmp_path / "bars.csv"
    frame.to_csv(path, index=False)

    cleaned = load_market_data(path)

    assert len(cleaned) == rows - 3
    assert list(cleaned.columns) == ["open", "high", "low", "close", "volume"]
    assert cleaned["close"].iloc[0] == 209.0
    assert np.isfinite(cleaned.to_numpy()).all()
    assert (cleaned[["open", "high", "low", "close"]] > 0).all().all()
    assert (cleaned["volume"] >= 0).all()


def test_load_market_data_rejects_too_few_valid_rows(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "open": [100.0] * 99,
            "high": [101.0] * 99,
            "low": [99.0] * 99,
            "close": [100.0] * 99,
            "volume": [1000.0] * 99,
        }
    )
    path = tmp_path / "too_short.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="at least 100 valid rows"):
        load_market_data(path)
