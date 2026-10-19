from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .data import load_ohlcv_csv, validate_ohlcv


class MarketEnvironment:
    """
    Sequential historical market-data environment.

    The environment exposes only information available up to the
    current timestep. Agents interact with this environment through
    observations and actions.
    """

    REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}

    def __init__(
        self,
        data: pd.DataFrame,
        initial_cash: float = 100_000.0,
    ) -> None:
        self.data = self._validate_data(data)
        self.initial_cash = float(initial_cash)

        self.current_step = 0
        self.done = False

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        initial_cash: float = 100_000.0,
    ) -> "MarketEnvironment":
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Market data not found: {path}")

        data = load_ohlcv_csv(path, require_timestamp=False)

        return cls(
            data=data,
            initial_cash=initial_cash,
        )

    @staticmethod
    def _validate_data(data: pd.DataFrame) -> pd.DataFrame:
        return validate_ohlcv(data, timezone="UTC", regular_session_only=False)

    @property
    def current_price(self) -> float:
        return float(self.data.iloc[self.current_step]["close"])

    @property
    def current_bar(self) -> pd.Series:
        return self.data.iloc[self.current_step]

    @property
    def steps_remaining(self) -> int:
        return len(self.data) - self.current_step - 1

    def reset(self, start_step: int = 0) -> pd.Series:
        if not 0 <= start_step < len(self.data):
            raise ValueError(
                f"start_step must be between 0 and {len(self.data) - 1}"
            )

        self.current_step = start_step
        self.done = False

        return self.current_bar

    def step(self) -> Optional[pd.Series]:
        if self.done:
            return None

        next_step = self.current_step + 1

        if next_step >= len(self.data):
            self.done = True
            return None

        self.current_step = next_step

        return self.current_bar

    def get_history(self, lookback: int) -> pd.DataFrame:
        if lookback <= 0:
            raise ValueError("lookback must be greater than zero.")

        start = max(0, self.current_step - lookback + 1)

        return self.data.iloc[start : self.current_step + 1].copy()

    def get_observation_window(self, lookback: int) -> pd.DataFrame:
        """
        Return the historical window available to the agent.

        No future rows are included.
        """
        return self.get_history(lookback)

    def __len__(self) -> int:
        return len(self.data)
