from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrainingEpisode:
    data: pd.DataFrame
    instrument: str
    start_index: int
    end_index: int
    start_timestamp: str | None
    end_timestamp: str | None
    seed: int
    regime: str | None


class TrainingEpisodeSampler:
    """Sample bounded, reproducible windows from training data only."""

    def __init__(
        self,
        *,
        lookback: int,
        episode_bars: int = 0,
        randomize_start: bool = True,
        randomize_instrument: bool = True,
        randomize_market_regime: bool = False,
    ) -> None:
        if lookback < 1:
            raise ValueError("lookback must be positive")
        self.lookback = int(lookback)
        self.episode_bars = int(episode_bars)
        self.randomize_start = bool(randomize_start)
        self.randomize_instrument = bool(randomize_instrument)
        self.randomize_market_regime = bool(randomize_market_regime)

    def sample(
        self,
        datasets: list[pd.DataFrame],
        *,
        episodes: int,
        rng: np.random.Generator,
    ) -> list[TrainingEpisode]:
        if not datasets:
            raise ValueError("At least one training dataset is required")
        if episodes < 1:
            raise ValueError("episodes must be at least one")

        minimum_available = min(len(frame) - self.lookback for frame in datasets)
        if minimum_available < 1:
            raise ValueError(
                "Each dataset must contain lookback history and at least one future bar"
            )
        episode_bars = self.episode_bars
        if episode_bars <= 0:
            episode_bars = max(1, minimum_available // 2)
        if episode_bars > minimum_available:
            raise ValueError(
                f"episode_bars={episode_bars} exceeds the shortest dataset capacity "
                f"({minimum_available} bars after lookback)"
            )

        candidates = [self._candidate_starts(frame, episode_bars) for frame in datasets]
        sampled: list[TrainingEpisode] = []
        for episode_index in range(episodes):
            if self.randomize_instrument:
                dataset_index = int(rng.integers(0, len(datasets)))
            else:
                dataset_index = episode_index % len(datasets)
            frame = datasets[dataset_index]
            starts = candidates[dataset_index]
            regime: str | None = None
            if self.randomize_market_regime:
                start, regime = self._sample_regime_start(frame, starts, rng)
            elif self.randomize_start:
                start = int(rng.choice(starts))
            else:
                candidate_index = min(
                    len(starts) - 1,
                    episode_index * len(starts) // episodes,
                )
                start = int(starts[candidate_index])

            end = start + episode_bars
            history_start = start - self.lookback + 1
            window = frame.iloc[history_start : end + 1].copy().reset_index(drop=True)
            symbol = (
                str(frame["symbol"].iloc[0])
                if "symbol" in frame
                else f"dataset_{dataset_index}"
            )
            sampled.append(
                TrainingEpisode(
                    data=window,
                    instrument=symbol,
                    start_index=start,
                    end_index=end,
                    start_timestamp=self._timestamp(frame, start),
                    end_timestamp=self._timestamp(frame, end),
                    seed=int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)),
                    regime=regime,
                )
            )
        return sampled

    def _candidate_starts(self, frame: pd.DataFrame, episode_bars: int) -> np.ndarray:
        upper_exclusive = len(frame) - episode_bars
        starts: np.ndarray = np.arange(
            self.lookback - 1, upper_exclusive, dtype=np.int64
        )
        if starts.size == 0:
            raise ValueError("No valid episode window fits the dataset boundaries")
        return starts

    @staticmethod
    def _timestamp(frame: pd.DataFrame, index: int) -> str | None:
        if "timestamp" not in frame:
            return None
        return pd.Timestamp(frame.iloc[index]["timestamp"]).isoformat()

    @staticmethod
    def _sample_regime_start(
        frame: pd.DataFrame,
        starts: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[int, str]:
        closes = frame["close"].to_numpy(dtype=np.float64)
        volatilities = []
        for start in starts:
            first = max(1, int(start) - 19)
            trailing_returns = np.diff(np.log(closes[first : int(start) + 1]))
            volatilities.append(
                float(np.std(trailing_returns)) if trailing_returns.size else 0.0
            )
        volatility: np.ndarray = np.asarray(volatilities, dtype=np.float64)
        low, high = np.quantile(volatility, [1 / 3, 2 / 3])
        labels = np.where(volatility <= low, 0, np.where(volatility <= high, 1, 2))
        names = ("low_volatility", "medium_volatility", "high_volatility")
        available = np.unique(labels)
        selected_label = int(rng.choice(available))
        matching = starts[labels == selected_label]
        return int(rng.choice(matching)), names[selected_label]
