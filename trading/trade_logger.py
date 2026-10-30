from __future__ import annotations

import csv
import os
from datetime import datetime, timezone


class TradeLogger:
    """Append trade decisions to a UTF-8 CSV with a stable header.

    Existing files are left intact. A missing or zero-byte file receives one
    header; later entries are appended without repeating it.
    """

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

        if not os.path.exists(path) or os.path.getsize(path) == 0:

            with open(
                path,
                "w",
                newline="",
                encoding="utf-8",
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
            encoding="utf-8",
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