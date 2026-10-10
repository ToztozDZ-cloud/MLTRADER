from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    assert isinstance(value, dict), f"Expected a YAML mapping in {path}"
    return value


def test_training_and_market_configs_are_compatible() -> None:
    training = _load_yaml(PROJECT_ROOT / "config" / "training.yaml")
    market = _load_yaml(PROJECT_ROOT / "config" / "market.yaml")

    training_settings = training["training"]
    market_settings = market["market"]

    assert int(training_settings["population_size"]) >= 2
    assert int(training_settings["generations"]) >= 1
    assert int(market_settings["data"]["lookback"]) > 0
    assert 0.0 < float(market_settings["trading"]["max_position_fraction"]) <= 1.0
    assert float(market_settings["trading"]["max_leverage"]) >= 1.0


def test_trading_universe_limits_are_valid() -> None:
    config = _load_yaml(PROJECT_ROOT / "config" / "trading_universe.yaml")
    universe = config["universe"]

    assert int(universe["max_scan_symbols"]) >= 1
    assert int(universe["bars_batch_size"]) >= 1
    assert int(universe["max_positions"]) >= 1
    assert 0.0 < float(universe["max_position_fraction"]) <= 1.0
    assert 0.0 < float(universe["max_gross_exposure"]) <= 1.0
    assert float(universe["min_price"]) > 0.0
