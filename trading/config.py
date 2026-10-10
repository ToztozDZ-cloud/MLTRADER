from __future__ import annotations

import os
<<<<<<< HEAD
=======
<<<<<<< HEAD

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
=======
>>>>>>> 7ce9a3260bf1e51b5cc30c7b002019690ed69fb8
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
if not API_KEY:
    raise RuntimeError("Missing APCA_API_KEY_ID in .env")
if not API_SECRET:
    raise RuntimeError("Missing APCA_API_SECRET_KEY in .env")

# This project intentionally supports paper trading only.
PAPER = True
PAPER_API_URL = "https://paper-api.alpaca.markets"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_CONFIG_PATH = PROJECT_ROOT / "config" / "trading_universe.yaml"
MODEL_PATH = str(PROJECT_ROOT / "models" / "champions" / "best_agent.json")
LOG_PATH = str(PROJECT_ROOT / "logs" / "paper_trades.csv")

TIMEFRAME_MINUTES = 5
DECISION_INTERVAL_SECONDS = TIMEFRAME_MINUTES * 60
LOOKBACK = 64

def load_universe_config() -> dict:
    with UNIVERSE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid trading universe config: {UNIVERSE_CONFIG_PATH}")
    return config.get("universe", {})
<<<<<<< HEAD
=======
>>>>>>> origin/master
>>>>>>> 7ce9a3260bf1e51b5cc30c7b002019690ed69fb8
