from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


from agents.serialization import AgentSerializer
from backtest.benchmarks import compare_strategies
from backtest.engine import BacktestEngine
from environment.observation import ObservationBuilder
from evolve import (
    DEFAULT_MARKET_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    build_backtest_engine,
    build_costs,
    load_market_data,
    load_yaml,
)
from validation.out_of_sample import OutOfSampleValidator
from validation.retraining import train_new_agent
from validation.robustness import RobustnessValidator
from validation.walk_forward import WalkForwardValidator


def _build_engine(market_config: dict, training_config: dict) -> BacktestEngine:
    return build_backtest_engine(market_config, training_config)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_data(paths: list[Path], market_config: dict):
    settings = market_config.get("market", {}).get("data", {})
    timeframe = str(settings.get("timeframe", "1d"))
    timezone = str(settings.get("timezone", "America/New_York"))
    regular_session = bool(settings.get("regular_session_only", True))
    return [
        (
            path,
            load_market_data(
                path,
                timeframe=timeframe,
                exchange_timezone=timezone,
                regular_session_only=regular_session,
            ),
        )
        for path in paths
    ]


def validate_mode_options(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    holdout_only = (
        args.train_size is not None
        or args.validation_size is not None
        or args.step_size is not None
        or args.minimum_history is not None
        or args.generations is not None
    )
    if args.mode == "holdout" and holdout_only:
        parser.error(
            "walk-forward options --train-size, --validation-size, --step-size, "
            "--minimum-history, and --generations are only valid with --mode walk-forward"
        )

    if args.mode == "walk-forward" and (
        args.train_fraction != 0.70
        or args.validation_fraction != 0.15
        or args.test_fraction != 0.15
    ):
        parser.error(
            "holdout options --train-fraction, --validation-fraction, and "
            "--test-fraction are only valid with --mode holdout"
        )

    if args.mode == "holdout":
        fractions = (args.train_fraction, args.validation_fraction, args.test_fraction)
        if any(value < 0 or value > 1 for value in fractions) or sum(fractions) != 1.0:
            parser.error("holdout fractions must be between 0 and 1 and sum to 1.0")

    if args.mode == "walk-forward":
        for name, value in (
            ("--train-size", args.train_size),
            ("--validation-size", args.validation_size),
            ("--step-size", args.step_size),
            ("--minimum-history", args.minimum_history),
            ("--generations", args.generations),
        ):
            if value is not None and value < 1:
                parser.error(f"{name} must be at least 1")
        if args.embargo_bars is not None and args.embargo_bars < 0:
            parser.error("--embargo-bars must be non-negative")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run isolated holdout evaluation or retraining walk-forward validation."
    )
    parser.add_argument("--data", required=True, nargs="+", type=Path)
    parser.add_argument(
        "--mode", choices=("holdout", "walk-forward"), default="holdout"
    )
    parser.add_argument(
        "--model", type=Path, default=Path("models/champions/best_agent.json")
    )
    parser.add_argument("--market-config", type=Path, default=DEFAULT_MARKET_CONFIG)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--validation-size", type=int, default=None)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument(
        "--window-mode", choices=("rolling", "expanding"), default="rolling"
    )
    parser.add_argument("--minimum-history", type=int, default=None)
    parser.add_argument("--embargo-bars", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    args = parser.parse_args()
    validate_mode_options(args, parser)

    training_config = load_yaml(args.training_config)
    market_config = load_yaml(args.market_config)
    training = training_config["training"]
    market = market_config.get("market", {})
    lookback = int(market.get("data", {}).get("lookback", 64))
    seed = int(args.seed if args.seed is not None else training.get("seed", 0))
    engine = _build_engine(market_config, training_config)
    datasets = _load_data(args.data, market_config)
    reports = []

    if args.mode == "holdout":
        agent = AgentSerializer.load(
            args.model,
            expected_observation=ObservationBuilder.schema(lookback),
            expected_timeframe=engine.evaluator.timeframe,
            require_metadata=True,
        )
        trained_hashes = set(
            agent.metadata.get("training", {}).get("dataset_hashes", [])
        )
        validator = OutOfSampleValidator(engine)
        risk = market.get("trading", {})
        base_costs = build_costs(market_config)
        robustness = RobustnessValidator(
            lookback=lookback,
            initial_cash=float(market.get("initial_cash", 100_000.0)),
            allow_short=bool(risk.get("allow_short", False)),
            max_position_fraction=float(risk.get("max_position_fraction", 1.0)),
            max_leverage=float(risk.get("max_leverage", 1.0)),
            timeframe=str(market.get("data", {}).get("timeframe", "1d")),
            base_costs=base_costs,
            execution_delay_bars=int(
                market.get("execution", {}).get("execution_delay_bars", 0)
            ),
        )
        for path, data in datasets:
            if _file_sha256(path) in trained_hashes:
                raise ValueError(
                    f"{path}: model metadata says this exact file was used for training; "
                    "it cannot serve as an independent holdout. Supply untouched test data."
                )
            result = validator.validate_holdout(
                agent,
                data,
                train_fraction=args.train_fraction,
                validation_fraction=args.validation_fraction,
                test_fraction=args.test_fraction,
                embargo_bars=args.embargo_bars,
            )
            partitions = result["partitions"]
            history_start = partitions["test_start"] - lookback + 1
            test_data = (
                data.iloc[history_start : partitions["test_end"]]
                .copy()
                .reset_index(drop=True)
            )
            benchmarks = compare_strategies(
                agent,
                test_data,
                engine,
                start_step=lookback - 1,
            )
            robust = robustness.validate(agent, test_data, start_step=lookback - 1)
            reports.append(
                {
                    "instrument": str(data["symbol"].iloc[0])
                    if "symbol" in data
                    else path.stem,
                    "source": str(path),
                    "holdout": {
                        "metrics": result["metrics"],
                        "partitions": partitions,
                        "bars": result["holdout_bars"],
                    },
                    "benchmarks_same_period_and_costs": benchmarks,
                    "robustness": [
                        {"scenario": item.scenario, "metrics": item.metrics}
                        for item in robust
                    ],
                    "evaluated_configurations": 1,
                    "evaluated_folds": 1,
                }
            )
        report = {
            "mode": "isolated_chronological_holdout",
            "seed": seed,
            "results": reports,
            "warning": (
                "The holdout must not guide tuning or model selection. Repeated experiments can "
                "indirectly overfit it; this CLI cannot detect analyst-driven reuse."
            ),
        }
    else:
        window_results = []
        for path, data in datasets:
            train_size = args.train_size or max(lookback + 2, int(len(data) * 0.60))
            validation_size = args.validation_size or max(2, int(len(data) * 0.15))
            walk_forward_validator = WalkForwardValidator(
                engine,
                train_size=train_size,
                validation_size=validation_size,
                step_size=args.step_size,
                expanding=args.window_mode == "expanding",
                minimum_history=args.minimum_history,
                embargo_bars=(
                    lookback if args.embargo_bars is None else args.embargo_bars
                ),
                seed=seed,
            )

            def train_agent(training_data, fold_seed):
                return train_new_agent(
                    training_data,
                    training_config,
                    market_config,
                    fold_seed,
                    generations=args.generations,
                )

            folds = walk_forward_validator.validate(data, train_agent)
            window_results.append(
                {
                    "instrument": str(data["symbol"].iloc[0])
                    if "symbol" in data
                    else path.stem,
                    "source": str(path),
                    "summary": walk_forward_validator.summarize(folds),
                    "training_configurations_per_fold": 1,
                }
            )
        report = {
            "mode": "retraining_walk_forward",
            "seed": seed,
            "window_mode": args.window_mode,
            "results": window_results,
        }
    print(json.dumps(report, indent=2, default=str, allow_nan=False))


if __name__ == "__main__":
    main()
