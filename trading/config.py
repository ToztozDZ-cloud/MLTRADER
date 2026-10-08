from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()


API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

if not API_KEY:
    raise RuntimeError(
        "Missing APCA_API_KEY_ID in .env"
    )

if not API_SECRET:
    raise RuntimeError(
        "Missing APCA_API_SECRET_KEY in .env"
    )


# ---------------------------------------------------------
# PAPER TRADING ONLY
# ---------------------------------------------------------

PAPER = True

PAPER_API_URL = (
    "https://paper-api.alpaca.markets"
)


# ---------------------------------------------------------
# TRADING CONFIGURATION
# ---------------------------------------------------------

SYMBOL = "SPY"

TIMEFRAME_MINUTES = 5

MODEL_PATH = "models/champions/best_agent.json"

LOG_PATH = "logs/paper_trades.csv"


# How frequently we evaluate the agent.
DECISION_INTERVAL_SECONDS = (
    TIMEFRAME_MINUTES * 60
)