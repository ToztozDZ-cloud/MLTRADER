from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from environment.costs import TradingCosts
from environment.data import load_ohlcv_csv
from environment.observation import ObservationBuilder
from evolution.evolution import EvolutionEngine
from evolution.population import Population
from training.evaluator import AgentEvaluator
from training.fitness import FitnessCalculator, periods_per_year_for_timeframe
from training.trainer import Trainer


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_TRAINING_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "training.yaml"
)

DEFAULT_MARKET_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "market.yaml"
)

CHAMPION_DIR = (
    PROJECT_ROOT
    / "models"
    / "champions"
)

CHAMPION_FILENAME = "best_agent.json"

CHAMPION_PATH = (
    CHAMPION_DIR
    / CHAMPION_FILENAME
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "archives"
)


def load_yaml(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            f"Configuration must be a mapping: {path}"
        )

    return data


def load_market_data(
    path: Path,
    *,
    regular_session_only: bool = True,
    exchange_timezone: str = "America/New_York",
    timeframe: str | None = None,
) -> pd.DataFrame:
    """Compatibility wrapper over the shared OHLCV validation pipeline."""
    return load_ohlcv_csv(
        path,
        timezone=exchange_timezone,
        regular_session_only=regular_session_only,
        timeframe=timeframe,
        require_timestamp=True,
        minimum_rows=100,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_population(
    training_config: dict,
    input_size: int,
    output_size: int,
) -> Population:
    training = training_config["training"]

    population_size = int(
        training["population_size"]
    )

    if population_size < 2:
        raise ValueError(
            "population_size must be at least 2."
        )

    return Population(
        size=population_size,
        input_size=input_size,
        output_size=output_size,
    )


def build_evolution_engine(
    training_config: dict,
    population: Population,
) -> EvolutionEngine:
    training = training_config["training"]

    evolution_config = training[
        "evolution"
    ]

    mutation_config = evolution_config.get(
        "mutation",
        {},
    )

    crossover_config = evolution_config.get(
        "crossover",
        {},
    )

    diversity_config = training.get(
        "diversity",
        {},
    )
    diversity_enabled = bool(diversity_config.get("enabled", True))

    mutation_enabled = bool(
        mutation_config.get(
            "enabled",
            True,
        )
    )

    mutation_rate = float(
        mutation_config.get(
            "rate",
            0.10,
        )
    )

    mutation_sigma = float(
        mutation_config.get(
            "weight_sigma",
            0.50,
        )
    )

    add_node_rate = float(
        mutation_config.get(
            "add_node_rate",
            0.02,
        )
    )

    add_connection_rate = float(
        mutation_config.get(
            "add_connection_rate",
            0.05,
        )
    )

    remove_node_rate = float(
        mutation_config.get(
            "remove_node_rate",
            0.01,
        )
    )

    remove_connection_rate = float(
        mutation_config.get(
            "remove_connection_rate",
            0.01,
        )
    )

    if not mutation_enabled:
        mutation_rate = 0.0
        add_node_rate = 0.0
        add_connection_rate = 0.0
        remove_node_rate = 0.0
        remove_connection_rate = 0.0

    return EvolutionEngine(
        population=population,
        elitism=float(
            evolution_config.get(
                "elitism",
                0.10,
            )
        ),
        survival_rate=float(
            evolution_config.get(
                "survival_rate",
                0.20,
            )
        ),
        mutation_rate=mutation_rate,
        mutation_sigma=mutation_sigma,
        crossover_rate=float(
            crossover_config.get("rate", 0.75)
            if bool(crossover_config.get("enabled", True))
            else 0.0
        ),
        immigrant_rate=float(
            diversity_config.get("random_immigrant_rate", 0.05)
            if diversity_enabled
            else 0.0
        ),
        compatibility_threshold=float(
            diversity_config.get(
                "compatibility_threshold",
                3.0,
            )
        ),
        add_node_rate=add_node_rate,
        add_connection_rate=add_connection_rate,
        remove_node_rate=remove_node_rate,
        remove_connection_rate=remove_connection_rate,
        diversity_enabled=diversity_enabled,
    )


def build_fitness_calculator(
    training_config: dict,
    timeframe: str = "1d",
) -> FitnessCalculator:
    fitness_config = (
        training_config["training"].get(
            "fitness",
            {},
        )
    )

    return FitnessCalculator(
        periods_per_year=periods_per_year_for_timeframe(timeframe),
        return_weight=float(
            fitness_config.get(
                "return_weight",
                1.0,
            )
        ),
        sharpe_weight=float(
            fitness_config.get(
                "sharpe_weight",
                0.5,
            )
        ),
        drawdown_penalty=float(
            fitness_config.get(
                "drawdown_penalty",
                1.0,
            )
        ),
        volatility_penalty=float(
            fitness_config.get(
                "volatility_penalty",
                0.25,
            )
        ),
        turnover_penalty=float(
            fitness_config.get(
                "turnover_penalty",
                0.10,
            )
        ),
        max_drawdown_limit=float(fitness_config.get("max_drawdown_limit", 0.50)),
        drawdown_excess_penalty=float(fitness_config.get("drawdown_excess_penalty", 2.0)),
        sharpe_cap=float(fitness_config.get("sharpe_cap", 5.0)),
    )


def build_costs(
    market_config: dict,
) -> TradingCosts:
    costs_config = (
        market_config
        .get("market", {})
        .get("costs", {})
    )

    return TradingCosts(
        commission_rate=float(
            costs_config.get(
                "commission_rate",
                0.0005,
            )
        ),
        spread_rate=float(
            costs_config.get(
                "spread_rate",
                0.0002,
            )
        ),
        slippage_rate=float(
            costs_config.get(
                "slippage_rate",
                0.0005,
            )
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the MLTRADER evolutionary "
            "trading system."
        )
    )

    parser.add_argument(
        "--data",
        required=True,
        nargs="+",
        type=Path,
        help="One or more OHLCV CSV files, ideally one per instrument.",
    )

    parser.add_argument(
        "--training-config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG,
        help=(
            "Path to training YAML configuration."
        ),
    )

    parser.add_argument(
        "--market-config",
        type=Path,
        default=DEFAULT_MARKET_CONFIG,
        help=(
            "Path to market YAML configuration."
        ),
    )

    parser.add_argument(
        "--reset-champion",
        action="store_true",
        help=(
            "Delete the persistent champion before "
            "starting training."
        ),
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help=(
            "Override the number of generations from "
            "the training configuration."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "xpu"),
        default="auto",
        help="Evaluation device; xpu selects Intel Arc GPUs.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the configured reproducibility seed.")
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--champion-dir", type=Path, default=CHAMPION_DIR)

    args = parser.parse_args()

    if (
        args.generations is not None
        and args.generations < 1
    ):
        parser.error(
            "--generations must be at least 1."
        )

    training_config = load_yaml(
        args.training_config
    )

    market_config = load_yaml(
        args.market_config
    )

    market_data_settings = market_config.get("market", {}).get("data", {})
    configured_timeframe = str(market_data_settings.get("timeframe", "1d"))
    configured_timezone = str(market_data_settings.get("timezone", "America/New_York"))
    regular_session_only = bool(market_data_settings.get("regular_session_only", True))
    market_datasets = [
        load_market_data(
            path,
            timeframe=configured_timeframe,
            exchange_timezone=configured_timezone,
            regular_session_only=regular_session_only,
        )
        for path in args.data
    ]
    market_data = market_datasets[0] if len(market_datasets) == 1 else market_datasets

    evaluation_config = training_config.get("training", {}).get("evaluation", {})
    seed = int(args.seed if args.seed is not None else training_config["training"].get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if AgentEvaluator._xpu_available():
        torch.xpu.manual_seed_all(seed)

    training_section = training_config[
        "training"
    ]

    parallel_config = training_section.get("parallel", {})
    if bool(parallel_config.get("enabled", False)):
        raise ValueError(
            "Parallel population evaluation is not supported by the current CPU reference evaluator; "
            "set training.parallel.enabled to false."
        )

    market_section = market_config.get(
        "market",
        {},
    )

    data_config = market_section.get(
        "data",
        {},
    )

    trading_config = market_section.get(
        "trading",
        {},
    )

    execution_config = market_section.get(
        "execution",
        {},
    )

    lookback = int(
        data_config.get(
            "lookback",
            64,
        )
    )

    input_size = ObservationBuilder.input_size(lookback)

    output_size = 1

    champion_path = args.champion_dir / CHAMPION_FILENAME
    if args.reset_champion:
        if champion_path.exists():
            champion_path.unlink()

            print(
                "Persistent champion removed:"
            )
            print(
                f"  {champion_path}"
            )
        else:
            print(
                "No persistent champion found. "
                "Starting from a fresh population."
            )

    population = build_population(
        training_config=training_config,
        input_size=input_size,
        output_size=output_size,
    )

    evolution = build_evolution_engine(
        training_config=training_config,
        population=population,
    )

    costs = build_costs(
        market_config
    )

    fitness_calculator = (
        build_fitness_calculator(
            training_config,
            timeframe=str(data_config.get("timeframe", "1d")),
        )
    )

    initial_cash = float(
        training_section.get(
            "initial_capital",
            market_section.get(
                "initial_cash",
                100000.0,
            ),
        )
    )

    evaluator = AgentEvaluator(
        lookback=lookback,
        initial_cash=initial_cash,
        timeframe=str(data_config.get("timeframe", "1d")),
        exchange_timezone=str(data_config.get("timezone", "America/New_York")),
        flatten_before_close_minutes=int(data_config.get("flatten_before_close_minutes", 10)),
        execution_delay_bars=int(execution_config.get("execution_delay_bars", 0)),
        costs=costs,
        allow_short=bool(
            trading_config.get(
                "allow_short",
                False,
            )
        ),
        max_position_fraction=float(
            trading_config.get(
                "max_position_fraction",
                1.0,
            )
        ),
        max_leverage=float(
            trading_config.get(
                "max_leverage",
                1.0,
            )
        ),
        execute_on_next_bar=bool(
            execution_config.get(
                "execute_on_next_bar",
                True,
            )
        ),
        fitness_calculator=fitness_calculator,
        device=args.device,
    )

    if args.generations is not None:
        generations = args.generations
    else:
        generations = int(
            training_section[
                "generations"
            ]
        )

    trainer = Trainer(
        population=population,
        evolution=evolution,
        evaluator=evaluator,
        generations=generations,
        checkpoint_dir=args.checkpoint_dir,
        champion_dir=args.champion_dir,
        champion_filename=CHAMPION_FILENAME,
        episodes_per_agent=int(evaluation_config.get("episodes_per_agent", 1)),
        episode_bars=int(evaluation_config.get("episode_bars", 0)),
        randomize_start=bool(evaluation_config.get("randomize_start", False)),
        randomize_instrument=bool(evaluation_config.get("randomize_instrument", False)),
        randomize_market_regime=bool(evaluation_config.get("randomize_market_regime", False)),
        seed=seed,
        model_metadata={
            "dataset_hashes": [file_sha256(path) for path in args.data],
            "training_config_sha256": hashlib.sha256(
                yaml.safe_dump(training_config, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "market_config_sha256": hashlib.sha256(
                yaml.safe_dump(market_config, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "timeframe": configured_timeframe,
            "instruments": [
                str(frame["symbol"].iloc[0]) if "symbol" in frame else str(path.stem)
                for frame, path in zip(market_datasets, args.data)
            ],
        },
        checkpoint_every=int(training_section.get("checkpoints", {}).get("save_every", 1)),
        keep_champions=bool(training_section.get("checkpoints", {}).get("keep_champions", True)),
        keep_archives=bool(training_section.get("checkpoints", {}).get("keep_archives", True)),
    )

    population_size = int(
        training_section[
            "population_size"
        ]
    )

    champion_exists = champion_path.exists()

    print()
    print("=" * 72)
    print("MLTRADER EVOLUTIONARY TRAINING")
    print("=" * 72)
    print(
        f"Market data:        {args.data}"
    )
    print(
        f"Market rows:        {[len(frame) for frame in market_datasets]}"
    )
    print(
        f"Lookback:           {lookback}"
    )
    print(
        f"Input size:         {input_size}"
    )
    print(
        f"Output size:        {output_size}"
    )
    print(
        f"Population:         {population_size}"
    )
    print(
        f"Generations:        {generations}"
    )
    print(f"Random seed:        {seed}")
    print(f"Episodes per agent: {int(evaluation_config.get('episodes_per_agent', 1))}")
    print(
        "Initial capital:    "
        f"${initial_cash:,.2f}"
    )
    print(
        "Next-bar execution: "
        f"{execution_config.get('execute_on_next_bar', True)}"
    )
    print(
        "Allow short:        "
        f"{trading_config.get('allow_short', False)}"
    )
    print(
        "Max position:       "
        f"{trading_config.get('max_position_fraction', 1.0)}"
    )
    print(
        "Max leverage:       "
        f"{trading_config.get('max_leverage', 1.0)}"
    )
    print(f"Evaluation device:  {evaluator.device}")
    if evaluator.using_xpu:
        print(f"GPU:                {torch.xpu.get_device_name(evaluator.device)}")
    elif evaluator.using_cuda:
        print(f"GPU:                {torch.cuda.get_device_name(evaluator.device)}")

    print(
        "Persistent champion: "
        f"{champion_exists}"
    )
    print(
        f"Champion path:      {champion_path}"
    )
    print("=" * 72)
    print()

    best_agent = trainer.train(
        market_data
    )

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)
    print(
        "Best fitness:       "
        f"{best_agent.fitness:.6f}"
    )
    print(
        "Champion saved to:  "
        f"{champion_path}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
