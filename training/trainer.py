from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from agents.agent import Agent
from agents.serialization import AgentSerializer
from evolution.evolution import EvolutionEngine
from evolution.population import Population
from .evaluator import AgentEvaluator, EvaluationResult
from .episodes import TrainingEpisodeSampler
from environment.observation import ObservationBuilder


class Trainer:
    def __init__(
        self,
        population: Population,
        evolution: EvolutionEngine,
        evaluator: AgentEvaluator,
        generations: int,
        checkpoint_dir: str | Path = "models/archives",
        champion_dir: str | Path = "models/champions",
        champion_filename: str = "best_agent.json",
        episodes_per_agent: int = 1,
        episode_bars: int = 0,
        randomize_start: bool = False,
        randomize_instrument: bool = False,
        randomize_market_regime: bool = False,
        seed: int = 0,
        resume_champion: bool = True,
        model_metadata: dict | None = None,
        checkpoint_every: int = 1,
        keep_champions: bool = True,
        keep_archives: bool = True,
    ) -> None:
        self.population = population
        self.evolution = evolution
        self.evaluator = evaluator
        self.generations = int(generations)
        self.episodes_per_agent = int(episodes_per_agent)
        self.seed = int(seed)
        self.resume_champion = bool(resume_champion)
        self.model_metadata = dict(model_metadata or {})
        self.checkpoint_every = int(checkpoint_every)
        self.keep_champions = bool(keep_champions)
        self.keep_archives = bool(keep_archives)
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least one")
        if self.episodes_per_agent < 1:
            raise ValueError("episodes_per_agent must be at least one")
        self.episode_sampler = TrainingEpisodeSampler(
            lookback=self.evaluator.lookback,
            episode_bars=episode_bars,
            randomize_start=randomize_start,
            randomize_instrument=randomize_instrument,
            randomize_market_regime=randomize_market_regime,
        )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.champion_dir = Path(champion_dir)
        self.champion_path = (
            self.champion_dir / champion_filename
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.champion_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.loaded_champion: Agent | None = None

        print(
            "Trainer initialized"
        )

        print(
            f"Population size: "
            f"{len(self.population.agents)}"
        )

        print(
            f"Generations: "
            f"{self.generations}"
        )

        print(
            f"Champion path: "
            f"{self.champion_path}"
        )

        if getattr(self.evaluator, "using_accelerator", False):
            if self.evaluator.using_xpu:
                gpu_name = torch.xpu.get_device_name(self.evaluator.device)
                gpu_memory = (
                    torch.xpu.get_device_properties(self.evaluator.device).total_memory
                    / (1024 ** 3)
                )
            else:
                gpu_name = torch.cuda.get_device_name(self.evaluator.device)
                gpu_memory = (
                    torch.cuda.get_device_properties(self.evaluator.device).total_memory
                    / (1024 ** 3)
                )
            print(f"Accelerated population evaluation: {self.evaluator.device}")
            print(f"GPU: {gpu_name}")
            print(f"GPU memory: {gpu_memory:.2f} GB")
        else:
            print("Using CPU reference evaluation.")

    def train(
        self,
        market_data: pd.DataFrame | list[pd.DataFrame],
    ) -> Agent:
        if self.generations <= 0:
            raise ValueError(
                "generations must be greater than zero."
            )

        if self.resume_champion:
            self._load_previous_champion()

        start_time = time.perf_counter()

        best_agent: Agent | None = None

        for generation in range(
            self.generations
        ):
            generation_start = time.perf_counter()

            print(
                ""
            )

            print(
                f"Generation "
                f"{generation:04d}"
            )

            results = self._evaluate_population(
                market_data
            )

            ranked = sorted(
                zip(
                    self.population.agents,
                    results,
                ),
                key=lambda item: item[1].fitness if np.isfinite(item[1].fitness) else float("-inf"),
                reverse=True,
            )
            if not ranked or not np.isfinite(ranked[0][1].fitness):
                raise RuntimeError(
                    f"Generation {generation} produced no finite training fitness values"
                )

            self.population.agents = [
                agent
                for agent, _ in ranked
            ]

            for agent, result in ranked:
                agent.fitness = result.fitness

            generation_best = (
                self.population.agents[0]
            )

            generation_best_result = ranked[0][1]

            best_agent = generation_best.clone()

            print(
                f"Best fitness: "
                f"{generation_best_result.fitness:.6f}"
            )

            print(
                f"Final equity: "
                f"${generation_best_result.final_equity:,.2f}"
            )

            print(
                f"Return: "
                f"{generation_best_result.total_return:.2%}"
            )

            print(
                f"Max drawdown: "
                f"{generation_best_result.max_drawdown:.2%}"
            )

            print(
                f"Turnover: "
                f"{generation_best_result.turnover:.4f}"
            )

            if self.keep_champions:
                self._save_champion(generation_best)
            if self.keep_archives and (
                (generation + 1) % self.checkpoint_every == 0
                or generation == self.generations - 1
            ):
                self._save_generation_checkpoint(generation_best, generation)

            generation_elapsed = (
                time.perf_counter()
                - generation_start
            )

            total_elapsed = (
                time.perf_counter()
                - start_time
            )

            completed = generation + 1

            average_generation_time = (
                total_elapsed / completed
            )

            remaining = (
                self.generations
                - completed
            )

            eta_seconds = (
                average_generation_time
                * remaining
            )

            print(
                f"Generation time: "
                f"{self._format_duration(generation_elapsed)}"
            )

            print(
                f"Total elapsed: "
                f"{self._format_duration(total_elapsed)}"
            )

            print(
                f"ETA: "
                f"{self._format_duration(eta_seconds)}"
            )

            if generation >= self.generations - 1:
                continue

            self.evolution.evolve()

        if best_agent is None:
            raise RuntimeError(
                "Training completed without producing "
                "a best agent."
            )

        self._save_champion(
            best_agent
        )

        return best_agent

    def _load_previous_champion(
        self,
    ) -> None:
        if not self.champion_path.exists():
            print(
                ""
            )

            print(
                "No previous champion found."
            )

            print(
                "Training will start from a "
                "new population."
            )

            return

        print(
            ""
        )

        print(
            "Previous champion found:"
        )

        print(
            f"  {self.champion_path}"
        )

        try:
            champion = AgentSerializer.load(
                self.champion_path,
                expected_observation=ObservationBuilder.schema(self.evaluator.lookback),
                expected_timeframe=self.model_metadata.get("timeframe"),
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the previous "
                f"champion from "
                f"{self.champion_path}: {exc}"
            ) from exc

        if not isinstance(
            champion,
            Agent,
        ):
            raise RuntimeError(
                "Champion file did not contain "
                "a valid Agent."
            )

        expected_inputs = ObservationBuilder.input_size(self.evaluator.lookback)
        if (
            champion.genome.input_size != expected_inputs
            or champion.genome.output_size != 1
        ):
            raise RuntimeError(
                "Previous champion architecture is incompatible with the configured "
                f"observation/action schema (expected {expected_inputs} inputs and one output)."
            )

        champion.fitness = float("-inf")
        champion.reset()

        self._insert_champion(
            champion
        )

        self.loaded_champion = champion

        print(
            "Previous champion loaded "
            "into the new population."
        )

    def _insert_champion(
        self,
        champion: Agent,
    ) -> None:
        if not self.population.agents:
            raise RuntimeError(
                "Population is empty; cannot "
                "insert champion."
            )

        champion_clone = champion.clone()

        champion_clone.fitness = float(
            "-inf"
        )

        champion_clone.reset()

        self.population.agents[0] = (
            champion_clone
        )

    def _save_champion(
        self,
        agent: Agent,
    ) -> None:
        champion = agent.clone()
        metadata = {
            "observation": ObservationBuilder.schema(self.evaluator.lookback),
            "training_seed": self.seed,
            "training": self.model_metadata,
        }

        AgentSerializer.save(
            champion,
            self.champion_path,
            metadata=metadata,
        )

        print(
            "Champion saved:"
        )

        print(
            f"  {self.champion_path}"
        )

    def _save_generation_checkpoint(
        self,
        agent: Agent,
        generation: int,
    ) -> None:
        checkpoint_path = (
            self.checkpoint_dir
            / f"generation_{generation:04d}_best.json"
        )

        checkpoint = agent.clone()
        metadata = {
            "observation": ObservationBuilder.schema(self.evaluator.lookback),
            "training_seed": self.seed,
            "training": self.model_metadata,
        }

        AgentSerializer.save(
            checkpoint,
            checkpoint_path,
            metadata=metadata,
        )

    def _evaluate_population(
        self,
        market_data: pd.DataFrame | list[pd.DataFrame],
    ):
        agents = self.population.agents
        datasets = market_data if isinstance(market_data, list) else [market_data]
        if not datasets:
            raise ValueError("At least one market dataset is required.")

        # Use the CPU reference evaluator for training until the vectorized GPU
        # simulator reproduces the exact ObservationBuilder portfolio state and
        # execution accounting. A faster path with different inputs can select
        # a different champion, so correctness takes priority here.

        episode_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.population.generation])
        )
        episodes = self.episode_sampler.sample(
            datasets,
            episodes=self.episodes_per_agent,
            rng=episode_rng,
        )
        total = len(agents)
        per_agent_results: list[list[EvaluationResult]] = [
            [] for _ in agents
        ]
        if getattr(self.evaluator, "using_accelerator", False):
            for episode in episodes:
                batch_results = self.evaluator.evaluate_population(
                    agents,
                    episode.data,
                    start_step=self.evaluator.lookback - 1,
                ).results
                for index, result in enumerate(batch_results):
                    per_agent_results[index].append(result)
        else:
            for index, agent in enumerate(agents):
                per_agent_results[index] = [
                    self.evaluator.evaluate(
                        agent,
                        episode.data,
                        start_step=self.evaluator.lookback - 1,
                    )
                    for episode in episodes
                ]

        results = []
        for index, (agent, per_dataset) in enumerate(
            zip(agents, per_agent_results),
            start=1,
        ):
            count = len(per_dataset)
            fitness_values: np.ndarray = np.asarray(
                [item.fitness for item in per_dataset], dtype=np.float64
            )
            if not np.isfinite(fitness_values).all():
                aggregate_fitness = float("-inf")
            else:
                tail_count = max(1, int(np.ceil(count * 0.25)))
                lower_tail = float(np.mean(np.sort(fitness_values)[:tail_count]))
                aggregate_fitness = float(0.5 * np.mean(fitness_values) + 0.5 * lower_tail)
            curve_length = len(per_dataset[0].equity_curve)
            mean_curve = [
                float(np.mean([item.equity_curve[j] for item in per_dataset]))
                for j in range(curve_length)
            ]
            aggregate = EvaluationResult(
                fitness=aggregate_fitness,
                final_equity=float(np.mean([item.final_equity for item in per_dataset])),
                total_return=float(np.mean([item.total_return for item in per_dataset])),
                max_drawdown=max(item.max_drawdown for item in per_dataset),
                turnover=float(np.mean([item.turnover for item in per_dataset])),
                equity_curve=mean_curve,
                trade_count=sum(item.trade_count for item in per_dataset),
                transaction_costs=float(np.mean([item.transaction_costs for item in per_dataset])),
                average_gross_exposure=float(np.mean([item.average_gross_exposure for item in per_dataset])),
                max_gross_exposure=max(item.max_gross_exposure for item in per_dataset),
            )
            agent.fitness = aggregate.fitness
            results.append(aggregate)
            self._log_episode_results(index, episodes, per_dataset, aggregate)
            print(
                f"\rEvaluating agents: {index}/{total} "
                f"({len(episodes)} episodes; fitness={aggregate.fitness:.5f})",
                end="",
                flush=True,
            )
        print()
        return results

    def _log_episode_results(self, agent_index, episodes, results, aggregate) -> None:
        def json_number(value):
            return float(value) if np.isfinite(value) else None

        record = {
            "generation": self.population.generation,
            "agent_index": agent_index - 1,
            "seed": self.seed,
            "episode_count": len(episodes),
            "episodes": [
                {
                    "seed": episode.seed,
                    "instrument": episode.instrument,
                    "start": episode.start_timestamp,
                    "end": episode.end_timestamp,
                    "start_index": episode.start_index,
                    "end_index": episode.end_index,
                    "regime": episode.regime,
                    "fitness": json_number(result.fitness),
                    "total_return": json_number(result.total_return),
                    "max_drawdown": json_number(result.max_drawdown),
                    "turnover": json_number(result.turnover),
                }
                for episode, result in zip(episodes, results)
            ],
            "aggregate": {
                "fitness": json_number(aggregate.fitness),
                "mean_fitness": json_number(np.mean([result.fitness for result in results])),
                "lower_tail_fitness": float(
                    np.mean(
                        [value for value in sorted(result.fitness for result in results)[:max(1, int(np.ceil(len(results) * 0.25)))] if np.isfinite(value)]
                    )
                ) if any(np.isfinite(result.fitness) for result in results) else None,
                "total_return": json_number(aggregate.total_return),
                "max_drawdown": json_number(aggregate.max_drawdown),
            },
        }
        path = self.checkpoint_dir / "training_episodes.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")

    @staticmethod
    def _format_duration(
        seconds: float,
    ) -> str:
        seconds = max(
            0.0,
            float(seconds),
        )

        if seconds < 60:
            return (
                f"{seconds:.1f}s"
            )

        minutes, seconds = divmod(
            int(seconds),
            60,
        )

        if minutes < 60:
            return (
                f"{minutes}m "
                f"{seconds:02d}s"
            )

        hours, minutes = divmod(
            minutes,
            60,
        )

        return (
            f"{hours}h "
            f"{minutes:02d}m "
            f"{seconds:02d}s"
        )
