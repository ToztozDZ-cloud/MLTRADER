from __future__ import annotations

import random
import sys
from contextlib import redirect_stdout
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from agents.agent import Agent
from environment.costs import TradingCosts
from environment.observation import ObservationBuilder
from evolution.evolution import EvolutionEngine
from training.evaluator import AgentEvaluator
from training.trainer import Trainer
from evolve import (
    build_evolution_engine,
    build_fitness_calculator,
    build_population,
    build_costs,
)


def train_new_agent(
    training_data: pd.DataFrame,
    training_config: dict,
    market_config: dict,
    seed: int,
    *,
    generations: int | None = None,
) -> Agent:
    """Train one fresh evolutionary population using only the supplied window."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    training = training_config["training"]
    market = market_config.get("market", {})
    data_settings = market.get("data", {})
    trading = market.get("trading", {})
    execution = market.get("execution", {})
    lookback = int(data_settings.get("lookback", 64))
    population = build_population(
        training_config,
        input_size=ObservationBuilder.input_size(lookback),
        output_size=1,
    )
    evolution: EvolutionEngine = build_evolution_engine(training_config, population)
    fitness = build_fitness_calculator(
        training_config,
        timeframe=str(data_settings.get("timeframe", "1d")),
    )
    costs: TradingCosts = build_costs(market_config)
    evaluator = AgentEvaluator(
        lookback=lookback,
        initial_cash=float(
            training.get("initial_capital", market.get("initial_cash", 100_000.0))
        ),
        costs=costs,
        allow_short=bool(trading.get("allow_short", False)),
        max_position_fraction=float(trading.get("max_position_fraction", 1.0)),
        max_leverage=float(trading.get("max_leverage", 1.0)),
        execute_on_next_bar=bool(execution.get("execute_on_next_bar", True)),
        execution_delay_bars=int(execution.get("execution_delay_bars", 0)),
        fitness_calculator=fitness,
        timeframe=str(data_settings.get("timeframe", "1d")),
        exchange_timezone=str(data_settings.get("timezone", "America/New_York")),
        flatten_before_close_minutes=int(
            data_settings.get("flatten_before_close_minutes", 10)
        ),
    )
    evaluation = training.get("evaluation", {})
    with (
        TemporaryDirectory(prefix="mltrader-fold-") as output_dir,
        redirect_stdout(sys.stderr),
    ):
        trainer = Trainer(
            population,
            evolution,
            evaluator,
            generations=int(
                generations
                if generations is not None
                else training.get("generations", 1)
            ),
            checkpoint_dir=f"{output_dir}/archives",
            champion_dir=f"{output_dir}/champions",
            episodes_per_agent=int(evaluation.get("episodes_per_agent", 1)),
            episode_bars=int(evaluation.get("episode_bars", 0)),
            randomize_start=bool(evaluation.get("randomize_start", False)),
            randomize_instrument=False,
            randomize_market_regime=bool(
                evaluation.get("randomize_market_regime", False)
            ),
            seed=seed,
            resume_champion=False,
            model_metadata={"validation_fold_seed": seed},
        )
        return trainer.train(training_data)
