from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

# This project intentionally supports paper trading only.
PAPER = True
PAPER_API_URL = "https://paper-api.alpaca.markets"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_CONFIG_PATH = PROJECT_ROOT / "config" / "trading_universe.yaml"
PAPER_RUNTIME_CONFIG_PATH = PROJECT_ROOT / "config" / "live.yaml"
MODEL_PATH = str(PROJECT_ROOT / "models" / "champions" / "best_agent.json")
LOG_PATH = str(PROJECT_ROOT / "logs" / "paper_trades.csv")

def validate_paper_runtime_config(config: dict) -> dict:
    if not isinstance(config, dict) or not isinstance(config.get("paper"), dict):
        raise ValueError(f"Invalid paper runtime config: {PAPER_RUNTIME_CONFIG_PATH}")
    settings = config["paper"]
    if settings.get("mode") != "paper":
        raise RuntimeError("Only paper trading is supported; runtime mode must be 'paper'.")
    broker = settings.get("broker", {})
    if broker.get("provider") != "alpaca" or broker.get("endpoint") != "paper":
        raise RuntimeError("Paper runtime must use Alpaca's paper endpoint.")
    market_data = settings.get("market_data", {})
    timeframe = str(market_data.get("timeframe", "")).strip().lower()
    if not timeframe.endswith("m") or not timeframe[:-1].isdigit():
        raise ValueError("Paper market-data timeframe must be configured in minutes, e.g. 5m.")
    if int(timeframe[:-1]) <= 0:
        raise ValueError("Paper market-data timeframe must be positive.")
    if int(market_data.get("lookback", 0)) <= 0:
        raise ValueError("Paper market-data lookback must be positive.")
    return settings


def _load_paper_runtime_config() -> dict:
    with PAPER_RUNTIME_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return validate_paper_runtime_config(config)


PAPER_RUNTIME_CONFIG = _load_paper_runtime_config()
TIMEFRAME_MINUTES = int(PAPER_RUNTIME_CONFIG["market_data"]["timeframe"][:-1])
DECISION_INTERVAL_SECONDS = TIMEFRAME_MINUTES * 60
LOOKBACK = int(PAPER_RUNTIME_CONFIG["market_data"]["lookback"])

def require_paper_credentials() -> tuple[str, str]:
    if not API_KEY or not API_SECRET:
        raise RuntimeError("Missing Alpaca paper credentials in .env")
    return API_KEY, API_SECRET


def load_universe_config() -> dict:
    with UNIVERSE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid trading universe config: {UNIVERSE_CONFIG_PATH}")
    return config.get("universe", {})
