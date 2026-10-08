from __future__ import annotations

import math
import time

import numpy as np

from trading.alpaca_broker import (
    PaperBroker,
)
from trading.config import (
    DECISION_INTERVAL_SECONDS,
    LOG_PATH,
    MODEL_PATH,
    SYMBOL,
)
from trading.market_data import (
    MarketData,
)
from trading.model_store import (
    load_agent,
)
from trading.trade_logger import (
    TradeLogger,
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

LOOKBACK = 50

# The agent output is expected to represent
# a target exposure:
#
#   -1.0 = fully short
#    0.0 = flat
#   +1.0 = fully long
#
# IMPORTANT:
# This assumption must match your backtest.
#
# We will verify this against your existing
# fitness code before you actually run it.


class PaperTrader:

    def __init__(self) -> None:

        print(
            "Loading trained agent..."
        )

        self.agent = load_agent(
            MODEL_PATH,
        )

        print(
            "Agent loaded."
        )

        self.market = MarketData()

        self.broker = PaperBroker()

        self.logger = TradeLogger(
            LOG_PATH,
        )

        self.last_bar_time = None

    # -----------------------------------------------------
    # Convert market data into agent observation
    # -----------------------------------------------------

    def build_observation(
        self,
        bars,
    ) -> np.ndarray:

        closes = (
            bars["close"]
            .astype(float)
            .to_numpy()
        )

        if len(closes) < LOOKBACK:
            raise RuntimeError(
                "Not enough market data."
            )

        recent = closes[
            -LOOKBACK:
        ]

        # Basic normalized return representation.
        #
        # WARNING:
        # This MUST eventually be replaced with
        # the EXACT SAME observation construction
        # used during training.

        returns = np.diff(
            recent
        )

        last_price = recent[-1]

        if (
            last_price == 0
            or not math.isfinite(
                last_price
            )
        ):
            raise RuntimeError(
                "Invalid market price."
            )

        returns = (
            returns / last_price
        )

        return returns.astype(
            np.float32
        )

    # -----------------------------------------------------
    # Current position
    # -----------------------------------------------------

    def get_current_quantity(
        self,
    ) -> float:

        position = (
            self.broker.position()
        )

        if position is None:
            return 0.0

        return float(
            position.qty
        )

    # -----------------------------------------------------
    # Account equity
    # -----------------------------------------------------

    def get_equity(
        self,
    ) -> float:

        account = (
            self.broker.account()
        )

        return float(
            account.equity
        )

    # -----------------------------------------------------
    # Execute target position
    # -----------------------------------------------------

    def execute_target(
        self,
        target: float,
        price: float,
    ):

        account = (
            self.broker.account()
        )

        equity = float(
            account.equity
        )

        if equity <= 0:
            raise RuntimeError(
                "Invalid account equity."
            )

        # Clamp only because the broker
        # needs a valid target range.
        #
        # This does NOT impose risk management.
        #
        # It defines the action space.

        target = max(
            -1.0,
            min(
                1.0,
                target,
            ),
        )

        target_dollars = (
            equity * target
        )

        target_quantity = (
            target_dollars / price
        )

        current_quantity = (
            self.get_current_quantity()
        )

        difference = (
            target_quantity
            - current_quantity
        )

        # Ignore tiny changes.

        if abs(difference) < 0.001:
            return None

        if difference > 0:

            return self.broker.buy(
                difference
            )

        return self.broker.sell(
            abs(difference)
        )

    # -----------------------------------------------------
    # One decision
    # -----------------------------------------------------

    def step(self) -> None:

        bars = (
            self.market.get_recent_bars(
                LOOKBACK,
            )
        )

        latest_time = bars.index[-1]

        # Only process each completed bar once.

        if (
            latest_time
            == self.last_bar_time
        ):
            return

        self.last_bar_time = (
            latest_time
        )

        price = float(
            bars["close"].iloc[-1]
        )

        observation = (
            self.build_observation(
                bars,
            )
        )

        # ---------------------------------------------
        # THE AGENT MAKES THE DECISION
        # ---------------------------------------------

        output = self.agent.act(
            observation
        )

        output = float(
            np.asarray(
                output
            ).reshape(-1)[0]
        )

        if not math.isfinite(output):
            raise RuntimeError(
                "Agent produced an invalid output."
            )

        print(
            f"\n"
            f"Time: {latest_time}\n"
            f"Price: {price:.2f}\n"
            f"Agent output: {output:.6f}"
        )

        # ---------------------------------------------
        # EXECUTION
        # ---------------------------------------------

        order = (
            self.execute_target(
                output,
                price,
            )
        )

        equity = (
            self.get_equity()
        )

        current_position = (
            self.get_current_quantity()
        )

        order_id = None

        if order is not None:

            order_id = str(
                order.id
            )

            print(
                f"ORDER: {order_id}"
            )

        print(
            f"Position: "
            f"{current_position}"
        )

        print(
            f"Equity: "
            f"${equity:.2f}"
        )

        self.logger.log(
            symbol=SYMBOL,
            price=price,
            agent_output=output,
            target_position=output,
            current_position=current_position,
            equity=equity,
            order_id=order_id,
        )

    # -----------------------------------------------------
    # Main loop
    # -----------------------------------------------------

    def run(self) -> None:

        print(
            "\n"
            "====================================\n"
            "       MLTRADER PAPER TRADER\n"
            "====================================\n"
        )

        print(
            f"Symbol: {SYMBOL}"
        )

        print(
            "MODE: PAPER TRADING"
        )

        print(
            "Live trading is disabled."
        )

        while True:

            try:

                self.step()

            except KeyboardInterrupt:

                print(
                    "\nPaper trader stopped."
                )

                break

            except Exception as exc:

                print(
                    f"\nERROR: {exc}"
                )

            time.sleep(
                DECISION_INTERVAL_SECONDS
            )


if __name__ == "__main__":

    trader = PaperTrader()

    trader.run()