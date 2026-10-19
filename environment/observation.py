from __future__ import annotations

import numpy as np
import pandas as pd

from .market import MarketEnvironment
from .portfolio import Portfolio


class ObservationBuilder:
    """
    Converts raw market/environment state into the numerical
    observation supplied to the agent.

    No future market data is included.
    """

    MARKET_COLUMNS = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    PORTFOLIO_FEATURES = (
        "cash_over_equity",
        "signed_market_value_over_equity",
        "position_sign",
        "unrealized_pnl_over_equity",
    )
    SCHEMA_VERSION = 1

    def __init__(self, lookback: int = 64) -> None:
        if lookback <= 0:
            raise ValueError("lookback must be greater than zero.")

        self.lookback = lookback

    def build(
        self,
        market: MarketEnvironment,
        portfolio: Portfolio,
    ) -> np.ndarray:
        history = market.get_observation_window(self.lookback)

        portfolio_state = self._portfolio_state(
            market=market,
            portfolio=portfolio,
        )
        return self.build_from_history(history, portfolio_state, self.lookback)

    @classmethod
    def build_from_history(
        cls,
        history: pd.DataFrame,
        portfolio_state: np.ndarray,
        lookback: int,
    ) -> np.ndarray:
        if lookback <= 0 or len(history) != lookback:
            raise ValueError(f"Expected exactly {lookback} bars of observation history")
        missing = set(cls.MARKET_COLUMNS) - set(history.columns)
        if missing:
            raise ValueError(f"Observation history is missing fields: {sorted(missing)}")
        values = history[cls.MARKET_COLUMNS].to_numpy(dtype=np.float32)
        values = cls._normalize_market_data(values)
        state = np.asarray(portfolio_state, dtype=np.float32).reshape(-1)
        if state.shape != (len(cls.PORTFOLIO_FEATURES),):
            raise ValueError("Portfolio state does not match the observation schema")
        observation = np.concatenate([values.flatten(), state]).astype(np.float32)
        if not np.isfinite(observation).all():
            raise ValueError("Observation contains non-finite values")
        return observation

    @classmethod
    def input_size(cls, lookback: int) -> int:
        return int(lookback) * len(cls.MARKET_COLUMNS) + len(cls.PORTFOLIO_FEATURES)

    @classmethod
    def schema(cls, lookback: int) -> dict[str, object]:
        return {
            "version": cls.SCHEMA_VERSION,
            "market_features": list(cls.MARKET_COLUMNS),
            "market_feature_order": "oldest_to_newest",
            "lookback": int(lookback),
            "portfolio_features": list(cls.PORTFOLIO_FEATURES),
            "input_size": cls.input_size(lookback),
            "action_semantics": "signed_target_fraction",
        }

    @staticmethod
    def _normalize_market_data(
        values: np.ndarray,
    ) -> np.ndarray:
        """
        Relative normalization prevents raw price scale from
        dominating the neural network.
        """
        if len(values) == 0:
            return values

        reference_price = float(values[-1, 3])

        if reference_price <= 0:
            reference_price = 1.0

        normalized = values.copy()

        normalized[:, 0:4] = (
            normalized[:, 0:4] / reference_price
        ) - 1.0

        volume = normalized[:, 4]

        volume_reference = np.mean(volume)

        if volume_reference > 0:
            normalized[:, 4] = (
                volume / volume_reference
            ) - 1.0
        else:
            normalized[:, 4] = 0.0

        return normalized

    @staticmethod
    def _portfolio_state(
        market: MarketEnvironment,
        portfolio: Portfolio,
    ) -> np.ndarray:
        price = market.current_price
        equity = portfolio.equity(price)

        if equity <= 0:
            equity = 1.0
        if portfolio.cash is None:
            raise RuntimeError("portfolio cash has not been initialized")

        return np.array(
            [
                portfolio.cash / equity,
                portfolio.market_value(price) / equity,
                portfolio.position / max(
                    1.0,
                    abs(portfolio.position),
                ),
                portfolio.unrealized_pnl(price) / equity,
            ],
            dtype=np.float32,
        )
