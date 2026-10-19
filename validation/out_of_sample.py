from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from agents.agent import Agent
from backtest.engine import BacktestEngine


@dataclass(frozen=True)
class ChronologicalPartitions:
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int
    embargo_bars: int


class OutOfSampleValidator:
    """
    Runs the frozen champion on data that was not used for evolution.
    """

    def __init__(
        self,
        engine: BacktestEngine,
    ) -> None:
        self.engine = engine

    def validate(
        self,
        agent: Agent,
        market_data: pd.DataFrame,
    ) -> dict:
        return self.engine.run(
            agent,
            market_data,
        )

    def validate_holdout(
        self,
        agent: Agent,
        market_data: pd.DataFrame,
        *,
        train_fraction: float = 0.70,
        validation_fraction: float = 0.15,
        test_fraction: float = 0.15,
        embargo_bars: int | None = None,
    ) -> dict:
        lookback = self.engine.evaluator.lookback
        gap = lookback if embargo_bars is None else int(embargo_bars)
        partitions = self.split(
            len(market_data),
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            embargo_bars=gap,
        )
        if partitions.test_start < lookback - 1:
            raise ValueError("Holdout test partition lacks preceding lookback history")
        history_start = partitions.test_start - lookback + 1
        test_data = (
            market_data.iloc[history_start : partitions.test_end]
            .copy()
            .reset_index(drop=True)
        )
        result = self.engine.run(
            agent,
            test_data,
            start_step=lookback - 1,
        )
        metadata = asdict(partitions)
        for key in (
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
            "test_start",
            "test_end",
        ):
            metadata[f"{key}_timestamp"] = self._timestamp(market_data, metadata[key])
        return {
            **result,
            "partitions": metadata,
            "evaluated_configurations": 1,
            "evaluated_folds": 1,
            "holdout_bars": partitions.test_end - partitions.test_start,
        }

    @staticmethod
    def split(
        row_count: int,
        *,
        train_fraction: float,
        validation_fraction: float,
        test_fraction: float,
        embargo_bars: int,
    ) -> ChronologicalPartitions:
        fractions = (train_fraction, validation_fraction, test_fraction)
        if any(not np.isfinite(value) or value <= 0.0 for value in fractions):
            raise ValueError("Train, validation, and test fractions must be positive")
        if not np.isclose(sum(fractions), 1.0, atol=1e-8):
            raise ValueError("Train, validation, and test fractions must sum to 1")
        if row_count < 1 or embargo_bars < 0:
            raise ValueError("row_count must be positive and embargo_bars non-negative")
        train_end = int(row_count * train_fraction)
        validation_start = train_end + embargo_bars
        validation_end = int(row_count * (train_fraction + validation_fraction))
        test_start = validation_end + embargo_bars
        if (
            train_end <= 0
            or validation_end <= validation_start
            or row_count <= test_start
        ):
            raise ValueError(
                "Dataset is too short for the requested chronological partitions and embargo"
            )
        return ChronologicalPartitions(
            train_start=0,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
            test_start=test_start,
            test_end=row_count,
            embargo_bars=embargo_bars,
        )

    @staticmethod
    def _timestamp(data: pd.DataFrame, index: int) -> str | None:
        if "timestamp" not in data or not 0 <= index < len(data):
            return None
        return pd.Timestamp(data.iloc[index]["timestamp"]).isoformat()
