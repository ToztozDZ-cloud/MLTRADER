from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from agents.serialization import AgentSerializer
from trading.alpaca_broker import PaperBroker
from trading.config import DECISION_INTERVAL_SECONDS, LOG_PATH, MODEL_PATH, SYMBOL
from trading.market_data import MarketData
from trading.trade_logger import TradeLogger

LOOKBACK = 64
FEATURE_COUNT = LOOKBACK * 5 + 4


class PaperTrader:
    """Paper-trade the trained agent's signed target exposure.

    Output -1 targets a full short, 0 targets flat, +1 targets a full long.
    Short orders require the Alpaca paper account and symbol to be shortable.
    """

    def __init__(self) -> None:
        print(f"Loading trained agent from {MODEL_PATH}...")
        self.agent = AgentSerializer.load(MODEL_PATH)
        self.agent.reset()
        if self.agent.genome.input_size != FEATURE_COUNT:
            raise ValueError(
                f"Model expects {self.agent.genome.input_size} inputs; this trader "
                f"builds {FEATURE_COUNT} (64 OHLCV bars plus 4 portfolio features). "
                "Train/export with the same ObservationBuilder and lookback."
            )
        self.market = MarketData()
        self.broker = PaperBroker()
        self.logger = TradeLogger(LOG_PATH)
        self.last_bar_time = None
        print(f"Agent loaded. Symbol={SYMBOL}; lookback={LOOKBACK}.")

    @staticmethod
    def _bars(bars: pd.DataFrame) -> pd.DataFrame:
        frame = bars.copy()
        if isinstance(frame.index, pd.MultiIndex):
            frame = frame.reset_index(level="symbol", drop=True)
        frame.columns = [str(c).lower() for c in frame.columns]
        columns = ["open", "high", "low", "close", "volume"]
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValueError(f"Market data missing columns: {missing}")
        frame = frame[columns].apply(pd.to_numeric, errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
        if len(frame) < LOOKBACK:
            raise RuntimeError(f"Need {LOOKBACK} valid bars; received {len(frame)}.")
        frame = frame.iloc[-LOOKBACK:]
        if (frame["close"] <= 0).any():
            raise ValueError("Market data contains a non-positive close.")
        return frame

    def build_observation(self, bars: pd.DataFrame, account, position) -> np.ndarray:
        """Mirror environment/observation.py's OHLCV and portfolio features."""
        frame = self._bars(bars)
        values = frame[["open", "high", "low", "close", "volume"]].to_numpy(
            dtype=np.float32
        )
        reference_price = float(values[-1, 3])
        if not math.isfinite(reference_price) or reference_price <= 0:
            raise ValueError(f"Invalid reference price: {reference_price}")

        normalized = values.copy()
        normalized[:, :4] = normalized[:, :4] / reference_price - 1.0
        volume_mean = float(np.mean(normalized[:, 4]))
        normalized[:, 4] = (
            normalized[:, 4] / volume_mean - 1.0 if volume_mean > 0 else 0.0
        )

        price = reference_price
        equity = float(account.equity)
        cash = float(account.cash)
        quantity = float(position.qty) if position is not None else 0.0
        avg_entry = float(position.avg_entry_price) if position is not None else 0.0
        if not math.isfinite(equity) or equity <= 0:
            raise RuntimeError(f"Invalid account equity: {equity}")

        portfolio_state = np.asarray(
            [
                cash / equity,
                quantity * price / equity,
                quantity / max(1.0, abs(quantity)),
                ((price - avg_entry) * quantity / equity) if quantity else 0.0,
            ],
            dtype=np.float32,
        )
        observation = np.concatenate([normalized.flatten(), portfolio_state]).astype(
            np.float32
        )
        if observation.shape != (FEATURE_COUNT,) or not np.isfinite(observation).all():
            raise ValueError(
                f"Invalid observation: shape={observation.shape}, "
                f"expected=({FEATURE_COUNT},), finite={np.isfinite(observation).all()}."
            )
        return observation

    def get_current_quantity(self) -> float:
        position = self.broker.position()
        return float(position.qty) if position is not None else 0.0

    def execute_target(self, target: float, price: float):
        if not math.isfinite(target):
            raise ValueError("Agent output is not finite.")
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"Invalid market price: {price}")
        target = float(np.clip(target, -1.0, 1.0))
        account = self.broker.account()
        equity = float(account.equity)
        if not math.isfinite(equity) or equity <= 0:
            raise RuntimeError(f"Invalid account equity: {equity}")

        target_quantity = equity * target / price
        delta = target_quantity - self.get_current_quantity()
        if abs(delta) < 1e-6:
            return None
        # Positive delta buys/reduces a short; negative delta sells/reduces a long
        # or opens/increases a short, if Alpaca allows the short sale.
        return self.broker.buy(delta) if delta > 0 else self.broker.sell(abs(delta))

    def step(self) -> None:
        bars = self._bars(self.market.get_recent_bars(LOOKBACK))
        bar_time = bars.index[-1]
        if bar_time == self.last_bar_time:
            return

        price = float(bars["close"].iloc[-1])
        account = self.broker.account()
        position = self.broker.position()
        observation = self.build_observation(bars, account, position)
        output = float(np.asarray(self.agent.act(observation)).reshape(-1)[0])
        if not math.isfinite(output):
            raise RuntimeError("Agent produced a non-finite action.")
        target = float(np.clip(output, -1.0, 1.0))

        print(f"\nTime: {bar_time}\nPrice: {price:.4f}\nAgent action: {output:+.6f}")
        order = self.execute_target(target, price)
        self.last_bar_time = bar_time

        updated_account = self.broker.account()
        updated_position = self.broker.position()
        quantity = float(updated_position.qty) if updated_position is not None else 0.0
        order_id = str(order.id) if order is not None else None
        if order is not None:
            print(f"ORDER: {order_id}")
        print(
            f"Target exposure: {target:+.2%}\nPosition quantity: {quantity}"
            f"\nEquity: ${float(updated_account.equity):,.2f}"
        )
        self.logger.log(
            symbol=SYMBOL,
            price=price,
            agent_output=output,
            target_position=target,
            current_position=quantity,
            equity=float(updated_account.equity),
            order_id=order_id,
        )

    def run(self) -> None:
        print("\n" + "=" * 44 + "\n       MLTRADER PAPER TRADER\n" + "=" * 44)
        print(f"Symbol: {SYMBOL} | PAPER TRADING | Long / flat / short actions enabled")
        while True:
            try:
                self.step()
            except KeyboardInterrupt:
                print("\nPaper trader stopped.")
                break
            except Exception as exc:
                print(f"\nERROR: {type(exc).__name__}: {exc}")
            time.sleep(DECISION_INTERVAL_SECONDS)


if __name__ == "__main__":
    PaperTrader().run()
