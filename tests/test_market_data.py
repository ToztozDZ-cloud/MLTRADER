from __future__ import annotations

import pandas as pd
import pytest

from environment.data import load_ohlcv_csv, validate_ohlcv


def _bars(timestamps, *, symbols=None) -> pd.DataFrame:
    count = len(timestamps)
    data = {
        "timestamp": timestamps,
        "Open": [100.0] * count,
        "HIGH": [101.0] * count,
        "low": [99.0] * count,
        "Close": [100.0] * count,
        "Volume": [1000.0] * count,
    }
    if symbols is not None:
        data["symbol"] = symbols
    return pd.DataFrame(data)


def test_unsorted_bars_are_sorted_and_timezone_is_normalized() -> None:
    stamps = pd.to_datetime(
        ["2024-03-11 09:35", "2024-03-11 09:30"],
    ).tz_localize("America/New_York")
    result = validate_ohlcv(_bars(stamps), timezone="America/New_York")
    assert result["timestamp"].is_monotonic_increasing
    assert result["timestamp"].iloc[0] == pd.Timestamp("2024-03-11 13:30", tz="UTC")


def test_duplicate_timestamps_are_rejected() -> None:
    stamps = pd.to_datetime(["2024-01-02 10:00", "2024-01-02 10:00"], utc=True)
    with pytest.raises(ValueError, match="duplicate timestamp"):
        validate_ohlcv(_bars(stamps), timezone="UTC")


def test_invalid_ohlc_and_negative_volume_are_rejected() -> None:
    frame = _bars(pd.to_datetime(["2024-01-02"], utc=True))
    frame.loc[0, "HIGH"] = 98.0
    with pytest.raises(ValueError, match="inconsistent OHLC"):
        validate_ohlcv(frame)
    frame.loc[0, "HIGH"] = 101.0
    frame.loc[0, "Volume"] = -1.0
    with pytest.raises(ValueError, match="volume must be non-negative"):
        validate_ohlcv(frame)


def test_missing_intraday_interval_is_rejected() -> None:
    stamps = pd.to_datetime(
        ["2024-01-02 10:00", "2024-01-02 10:05", "2024-01-02 10:15"],
        utc=True,
    )
    with pytest.raises(ValueError, match="irregular 5m interval"):
        validate_ohlcv(_bars(stamps), timeframe="5m", timezone="UTC")


def test_two_hour_bars_are_checked_against_their_configured_interval() -> None:
    stamps = pd.to_datetime(
        ["2024-01-02 09:30", "2024-01-02 11:30"],
    ).tz_localize("America/New_York")

    result = validate_ohlcv(
        _bars(stamps),
        timeframe="2h",
        timezone="America/New_York",
    )

    assert len(result) == 2


def test_regular_session_filter_excludes_premarket_and_weekends() -> None:
    stamps = pd.to_datetime(
        [
            "2024-01-05 08:00-05:00",
            "2024-01-05 09:30-05:00",
            "2024-01-05 15:55-05:00",
            "2024-01-06 10:00-05:00",
        ],
        utc=True,
    )
    result = validate_ohlcv(
        _bars(stamps), timezone="America/New_York", regular_session_only=True
    )
    assert len(result) == 2
    local = result["timestamp"].dt.tz_convert("America/New_York")
    assert (local.dt.dayofweek < 5).all()
    assert (local.dt.hour * 60 + local.dt.minute >= 570).all()


def test_ambiguous_naive_dst_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid timestamp"):
        validate_ohlcv(
            _bars(["2024-11-03 01:30"]),
            timezone="America/New_York",
        )


def test_nonexistent_naive_dst_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid timestamp"):
        validate_ohlcv(
            _bars(["2024-03-10 02:30"]),
            timezone="America/New_York",
        )


def test_mixed_instruments_in_one_dataset_are_rejected() -> None:
    stamps = pd.date_range("2024-01-02", periods=2, freq="D", tz="UTC")
    with pytest.raises(ValueError, match="one instrument per dataset"):
        validate_ohlcv(_bars(stamps, symbols=["AAA", "BBB"]))


def test_csv_loader_uses_filename_as_symbol_identity(tmp_path) -> None:
    path = tmp_path / "SPY_5m.csv"
    stamps = pd.date_range(
        "2024-01-02 09:30", periods=3, freq="5min", tz="America/New_York"
    )
    _bars(stamps).to_csv(path, index=False)
    result = load_ohlcv_csv(path, timezone="America/New_York", timeframe="5m")
    assert set(result["symbol"]) == {"SPY_5M"}
