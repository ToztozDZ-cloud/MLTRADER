from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from environment.compat import suppress_dependency_warnings

suppress_dependency_warnings()

from alpaca.data.enums import DataFeed

load_dotenv()

API_KEY = None
API_SECRET = None

# This project intentionally supports paper trading only.
PAPER = True
PAPER_API_URL = "https://paper-api.alpaca.markets"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_CONFIG_PATH = PROJECT_ROOT / "config" / "trading_universe.yaml"
PAPER_RUNTIME_CONFIG_PATH = PROJECT_ROOT / "config" / "live.yaml"
MODEL_PATH = str(PROJECT_ROOT / "models" / "champions" / "best_agent.json")
LOG_PATH = str(PROJECT_ROOT / "logs" / "paper_trades.csv")

MARKET_DATA_FEEDS = {
    "iex": DataFeed.IEX,
    "sip": DataFeed.SIP,
}


def resolve_market_data_feed(feed: str | None = None) -> tuple[str, DataFeed]:
    """Resolve an optional override against the shared paper market-data setting."""
    configured_feed = (
        feed
        if feed is not None
        else PAPER_RUNTIME_CONFIG["market_data"].get("feed", "iex")
    )
    name = str(configured_feed).strip().lower()
    try:
        return name, MARKET_DATA_FEEDS[name]
    except KeyError as exc:
        supported = ", ".join(MARKET_DATA_FEEDS)
        raise ValueError(
            f"Unsupported Alpaca market-data feed {configured_feed!r}; use {supported}."
        ) from exc


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
    resolve_market_data_feed(str(market_data.get("feed", "iex")))
    return settings


def _load_paper_runtime_config() -> dict:
    with PAPER_RUNTIME_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return validate_paper_runtime_config(config)


class _LazyPaperRuntimeConfig(dict):
    def _ensure_loaded(self) -> None:
        if not getattr(self, "_loaded", False):
            self.update(_load_paper_runtime_config())
            self._loaded = True

    def __getitem__(self, key):
        self._ensure_loaded()
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._ensure_loaded()
        return super().get(key, default)

    def __contains__(self, key):
        self._ensure_loaded()
        return super().__contains__(key)


PAPER_RUNTIME_CONFIG = _LazyPaperRuntimeConfig()

# Compatibility names are replaced by these lazy accessors; they intentionally
# do not load YAML or environment configuration until runtime code requests it.
LOOKBACK = None
TIMEFRAME_MINUTES = None
DECISION_INTERVAL_SECONDS = None


def load_paper_runtime_config(*, reload: bool = False) -> dict:
    if reload:
        PAPER_RUNTIME_CONFIG.clear()
        PAPER_RUNTIME_CONFIG._loaded = False
    PAPER_RUNTIME_CONFIG._ensure_loaded()
    return dict(PAPER_RUNTIME_CONFIG)


def get_timeframe_minutes() -> int:
    settings = load_paper_runtime_config()["market_data"]
    return int(settings["timeframe"][:-1])


def get_decision_interval_seconds() -> int:
    return get_timeframe_minutes() * 60


def get_lookback() -> int:
    return int(load_paper_runtime_config()["market_data"]["lookback"])


def require_paper_credentials() -> tuple[str, str]:
    api_key = os.getenv("APCA_API_KEY_ID") or API_KEY
    api_secret = os.getenv("APCA_API_SECRET_KEY") or API_SECRET
    if not api_key or not api_secret:
        raise RuntimeError("Missing Alpaca paper credentials in .env")
    return api_key, api_secret


def load_universe_config() -> dict:
    with UNIVERSE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid trading universe config: {UNIVERSE_CONFIG_PATH}")
    return config.get("universe", {})
