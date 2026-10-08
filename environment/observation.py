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

        values = history[self.MARKET_COLUMNS].to_numpy(
            dtype=np.float32
        )

        values = self._normalize_market_data(values)

        portfolio_state = self._portfolio_state(
            market=market,
            portfolio=portfolio,
        )

        return np.concatenate(
            [
                values.flatten(),
                portfolio_state,
            ]
        ).astype(np.float32)

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

        reference_price = values[-1, 3]

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