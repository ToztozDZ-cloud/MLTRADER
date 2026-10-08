from __future__ import annotations

import csv
import os
from datetime import datetime, timezone


class TradeLogger:

    def __init__(
        self,
        path: str,
    ) -> None:

        self.path = path

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        if not os.path.exists(path):

            with open(
                path,
                "w",
                newline="",
            ) as file:

                writer = csv.writer(file)

                writer.writerow(
                    [
                        "timestamp",
                        "symbol",
                        "price",
                        "agent_output",
                        "target_position",
                        "current_position",
                        "equity",
                        "order_id",
                    ]
                )

    def log(
        self,
        symbol: str,
        price: float,
        agent_output: float,
        target_position: float,
        current_position: float,
        equity: float,
        order_id: str | None,
    ) -> None:

        with open(
            self.path,
            "a",
            newline="",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                    symbol,

                    price,

                    agent_output,

                    target_position,

                    current_position,

                    equity,

                    order_id,
                ]
            )