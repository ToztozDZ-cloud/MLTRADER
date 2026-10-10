from __future__ import annotations

import pandas as pd

from evolve import load_market_data


def test_intraday_loader_preserves_timestamps_and_filters_regular_session(tmp_path) -> None:
    # Include premarket and after-hours bars for three weekdays.
    stamps = []
    for day in ("2024-01-08", "2024-01-09", "2024-01-10"):
        stamps.extend(pd.date_range(
            f"{day} 08:00", f"{day} 17:00", freq="5min",
            tz="America/New_York", inclusive="left",
        ))
    frame = pd.DataFrame({
        "timestamp": [stamp.isoformat() for stamp in stamps],
        "open": [100.0] * len(stamps),
        "high": [101.0] * len(stamps),
        "low": [99.0] * len(stamps),
        "close": [100.0] * len(stamps),
        "volume": [1000.0] * len(stamps),
    })
    path = tmp_path / "intraday.csv"
    frame.to_csv(path, index=False)

    cleaned = load_market_data(path)

    local = cleaned["timestamp"].dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    assert len(cleaned) == 78 * 3
    assert (minute >= 570).all()
    assert (minute < 960).all()
    assert cleaned["timestamp"].is_monotonic_increasing
