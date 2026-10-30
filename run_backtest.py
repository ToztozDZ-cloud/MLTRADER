from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.serialization import AgentSerializer
from backtest.benchmarks import compare_strategies
from backtest.engine import BacktestEngine
from environment.observation import ObservationBuilder
from evolve import (
    DEFAULT_MARKET_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    build_costs,
    build_backtest_engine,
    load_market_data,
    load_yaml,
)


def build_engine(market_config: dict, training_config: dict) -> BacktestEngine:
    return build_backtest_engine(market_config, training_config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a validated MLTRADER champion backtest."
    )
    parser.add_argument(
        "--data", required=True, type=Path, help="One-instrument OHLCV CSV file."
    )
    parser.add_argument(
        "--model", type=Path, default=Path("models/champions/best_agent.json")
    )
    parser.add_argument("--market-config", type=Path, default=DEFAULT_MARKET_CONFIG)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument(
        "--benchmarks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare against cash, buy-and-hold, and a moving-average baseline.",
    )
    args = parser.parse_args()

    market_config = load_yaml(args.market_config)
    training_config = load_yaml(args.training_config)
    market = market_config.get("market", {})
    data_settings = market.get("data", {})
    timeframe = str(data_settings.get("timeframe", "1d"))
    data = load_market_data(
        args.data,
        timeframe=timeframe,
        exchange_timezone=str(data_settings.get("timezone", "America/New_York")),
        regular_session_only=bool(data_settings.get("regular_session_only", True)),
    )
    engine = build_engine(market_config, training_config)
    expected_schema = ObservationBuilder.schema(engine.evaluator.lookback)
    agent = AgentSerializer.load(
        args.model,
        expected_observation=expected_schema,
        expected_timeframe=engine.evaluator.timeframe,
        require_metadata=True,
    )
    result = engine.run(agent, data)
    report = {
        "instrument": str(data["symbol"].iloc[0])
        if "symbol" in data
        else args.data.stem,
        "start": data.iloc[engine.evaluator.lookback - 1].get("timestamp"),
        "end": data.iloc[-1].get("timestamp"),
        "rows": len(data),
        "metrics": result["metrics"],
    }
    if args.benchmarks:
        report["benchmarks"] = compare_strategies(agent, data, engine)
    print(json.dumps(report, indent=2, default=str, allow_nan=False))


if __name__ == "__main__":
    main()
