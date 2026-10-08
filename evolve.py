from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from environment.costs import TradingCosts
from evolution.evolution import EvolutionEngine
from evolution.population import Population
from training.evaluator import AgentEvaluator
from training.fitness import FitnessCalculator
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
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Market data file not found: {path}"
        )

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError(
            f"Market data is empty: {path}"
        )

    required_columns = {
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    columns = {
        column.lower()
        for column in data.columns
    }

    missing = required_columns - columns

    if missing:
        raise ValueError(
            "Market data is missing required "
            f"columns: {sorted(missing)}"
        )

    rename_map = {}

    for column in data.columns:
        lower = column.lower()

        if lower in required_columns:
            rename_map[column] = lower

    data = data.rename(
        columns=rename_map
    )

    data = data[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].copy()

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna().reset_index(
        drop=True
    )

    if len(data) < 100:
        raise ValueError(
            "Market data must contain at least "
            f"100 valid rows. Found {len(data)}."
        )

    return data


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
            crossover_config.get(
                "rate",
                0.75,
            )
        ),
        immigrant_rate=float(
            diversity_config.get(
                "random_immigrant_rate",
                0.05,
            )
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
    )


def build_fitness_calculator(
    training_config: dict,
) -> FitnessCalculator:
    fitness_config = (
        training_config["training"].get(
            "fitness",
            {},
        )
    )

    return FitnessCalculator(
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
        type=Path,
        help="Path to OHLCV CSV market data.",
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

    market_data = load_market_data(
        args.data
    )

    training_section = training_config[
        "training"
    ]

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

    input_size = (
        lookback * 5
    ) + 4

    output_size = 1

    if args.reset_champion:
        if CHAMPION_PATH.exists():
            CHAMPION_PATH.unlink()

            print(
                "Persistent champion removed:"
            )
            print(
                f"  {CHAMPION_PATH}"
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
            training_config
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
        checkpoint_dir=CHECKPOINT_DIR,
        champion_dir=CHAMPION_DIR,
        champion_filename=CHAMPION_FILENAME,
    )

    population_size = int(
        training_section[
            "population_size"
        ]
    )

    champion_exists = (
        CHAMPION_PATH.exists()
    )

    print()
    print("=" * 72)
    print("MLTRADER EVOLUTIONARY TRAINING")
    print("=" * 72)
    print(
        f"Market data:        {args.data}"
    )
    print(
        f"Market rows:        {len(market_data)}"
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
    print(
        "CUDA available:     "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():
        print(
            "GPU:                "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        "Persistent champion: "
        f"{champion_exists}"
    )
    print(
        f"Champion path:      {CHAMPION_PATH}"
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
        f"{CHAMPION_PATH}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()