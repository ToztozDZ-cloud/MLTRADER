from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
TIMESTAMP_ALIASES = ("timestamp", "datetime", "date", "time")


def _source_name(source: str | Path | None) -> str:
    return str(source) if source is not None else "market data"


def _parse_timestamps(
    values: pd.Series,
    *,
    timezone: str,
    source: str,
) -> pd.Series:
    parsed: list[pd.Timestamp] = []
    for row_number, value in enumerate(values, start=2):
        try:
            stamp = pd.Timestamp(value)
            if pd.isna(stamp):
                raise ValueError("timestamp is missing or invalid")
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize(
                    timezone,
                    ambiguous="raise",
                    nonexistent="raise",
                )
            parsed.append(stamp.tz_convert("UTC"))
        except Exception as exc:
            raise ValueError(
                f"{source}: invalid timestamp at row {row_number}: {value!r} ({exc})"
            ) from exc
    return pd.Series(pd.DatetimeIndex(parsed), index=values.index, name="timestamp")


def _timeframe_minutes(timeframe: str) -> int | None:
    value = str(timeframe).strip().lower()
    if value.endswith(("m", "h")) and value[:-1].isdigit():
        minutes = int(value[:-1]) * (60 if value.endswith("h") else 1)
        if minutes <= 0:
            raise ValueError("timeframe must be positive")
        return minutes
    if value in {"1d", "d", "day", "daily"}:
        return None
    raise ValueError(f"Unsupported timeframe {timeframe!r}; use e.g. '5m' or '1d'.")


def validate_ohlcv(
    data: pd.DataFrame,
    *,
    source: str | Path | None = None,
    timezone: str = "UTC",
    regular_session_only: bool = False,
    timeframe: str | None = None,
    require_timestamp: bool = False,
) -> pd.DataFrame:
    """Normalize and validate one instrument's OHLCV bars without fabricating data."""
    label = _source_name(source)
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError(f"{label}: market data is empty or not a DataFrame")

    frame = data.copy()
    normalized_names = [str(column).strip().lower() for column in frame.columns]
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError(f"{label}: column names are duplicated after normalization")
    frame.columns = normalized_names

    timestamp_column = next(
        (name for name in TIMESTAMP_ALIASES if name in frame.columns), None
    )
    if timestamp_column is None and isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.reset_index()
        frame = frame.rename(columns={frame.columns[0]: "timestamp"})
        timestamp_column = "timestamp"
    if require_timestamp and timestamp_column is None:
        raise ValueError(f"{label}: a timestamp/datetime/date/time column is required")

    missing = [column for column in OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{label}: missing required OHLCV fields: {', '.join(missing)}"
        )

    result = pd.DataFrame(index=frame.index)
    if timestamp_column is not None:
        result["timestamp"] = _parse_timestamps(
            frame[timestamp_column], timezone=timezone, source=label
        )

    for column in OHLCV_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce").astype("float64")
        invalid = ~np.isfinite(values.to_numpy())
        if invalid.any():
            row_number = int(np.flatnonzero(invalid)[0]) + 2
            raise ValueError(f"{label}: {column} must be finite at row {row_number}")
        result[column] = values

    if "symbol" in frame.columns:
        symbols = frame["symbol"].astype("string").str.strip().str.upper()
        if symbols.isna().any() or (symbols == "").any():
            raise ValueError(f"{label}: symbol values must be present on every row")
        unique_symbols = symbols.unique()
        if len(unique_symbols) != 1:
            raise ValueError(
                f"{label}: expected one instrument per dataset; found {len(unique_symbols)}"
            )
        result["symbol"] = str(unique_symbols[0])

    prices = result[["open", "high", "low", "close"]]
    invalid_price = (prices <= 0).any(axis=1)
    if invalid_price.any():
        position = int(np.flatnonzero(invalid_price.to_numpy())[0])
        raise ValueError(f"{label}: OHLC prices must be positive at row {position + 2}")

    invalid_volume = result["volume"] < 0
    if invalid_volume.any():
        position = int(np.flatnonzero(invalid_volume.to_numpy())[0])
        raise ValueError(f"{label}: volume must be non-negative at row {position + 2}")

    invalid_ohlc = (result["high"] < result[["open", "close", "low"]].max(axis=1)) | (
        result["low"] > result[["open", "close", "high"]].min(axis=1)
    )
    if invalid_ohlc.any():
        position = int(np.flatnonzero(invalid_ohlc.to_numpy())[0])
        raise ValueError(
            f"{label}: inconsistent OHLC relationship at row {position + 2}"
        )

    if timestamp_column is not None:
        result = result.sort_values("timestamp", kind="stable")
        duplicates = result["timestamp"].duplicated(keep=False)
        if duplicates.any():
            duplicate = result.loc[duplicates, "timestamp"].iloc[0]
            raise ValueError(f"{label}: duplicate timestamp {duplicate.isoformat()}")

        is_intraday = bool(
            timeframe
            and str(timeframe).strip().lower() not in {"1d", "d", "day", "daily"}
        )
        if not is_intraday and len(result) > 1:
            median_interval = result["timestamp"].diff().dropna().median()
            is_intraday = median_interval < pd.Timedelta(days=1)
        if regular_session_only and is_intraday:
            local = result["timestamp"].dt.tz_convert(timezone)
            minute_of_day = local.dt.hour * 60 + local.dt.minute
            in_session = (
                (local.dt.dayofweek < 5)
                & (minute_of_day >= 9 * 60 + 30)
                & (minute_of_day < 16 * 60)
            )
            result = result.loc[in_session].copy()

        if timeframe is not None and str(timeframe).lower() not in {
            "1d",
            "d",
            "day",
            "daily",
        }:
            expected_minutes = _timeframe_minutes(timeframe)
            assert expected_minutes is not None
            if len(result) > 1:
                local = result["timestamp"].dt.tz_convert(timezone)
                deltas = result["timestamp"].diff()
                same_session_day = local.dt.date.eq(local.dt.date.shift())
                within_session = same_session_day
                irregular = within_session & deltas.ne(
                    pd.Timedelta(minutes=expected_minutes)
                )
                if irregular.any():
                    position = int(np.flatnonzero(irregular.to_numpy())[0])
                    raise ValueError(
                        f"{label}: irregular {timeframe} interval before row {position + 2}; "
                        f"found {deltas.iloc[position]}"
                    )
                observed = deltas.loc[deltas <= pd.Timedelta(minutes=expected_minutes)]
                if observed.empty:
                    raise ValueError(
                        f"{label}: no intraday intervals found for {timeframe} data"
                    )
                median_minutes = observed.median().total_seconds() / 60.0
                if (
                    not expected_minutes * 0.5
                    <= median_minutes
                    <= expected_minutes * 1.5
                ):
                    raise ValueError(
                        f"{label}: observed {median_minutes:g}-minute bars, expected {timeframe}"
                    )
        elif timeframe is not None and len(result) > 1:
            median_interval = result["timestamp"].diff().dropna().median()
            if median_interval < pd.Timedelta(hours=12):
                raise ValueError(
                    f"{label}: intraday bars do not match configured daily timeframe"
                )

    result = result.reset_index(drop=True)
    return result


def load_ohlcv_csv(
    path: str | Path,
    *,
    timezone: str = "America/New_York",
    regular_session_only: bool = False,
    timeframe: str | None = None,
    require_timestamp: bool = False,
    minimum_rows: int = 1,
) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Market data file not found: {source}")
    try:
        frame = pd.read_csv(source)
    except Exception as exc:
        raise ValueError(f"Could not read market data CSV {source}: {exc}") from exc
    result = validate_ohlcv(
        frame,
        source=source,
        timezone=timezone,
        regular_session_only=regular_session_only,
        timeframe=timeframe,
        require_timestamp=require_timestamp,
    )
    if "symbol" not in result.columns:
        result["symbol"] = source.stem.upper()
    if len(result) < minimum_rows:
        raise ValueError(
            f"{source}: expected at least {minimum_rows} valid rows after filtering; "
            f"found {len(result)}"
        )
    return result
