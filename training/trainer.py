from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch

from agents.agent import Agent
from agents.serialization import AgentSerializer
from evolution.evolution import EvolutionEngine
from evolution.population import Population
from .evaluator import AgentEvaluator


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
    ) -> None:
        self.population = population
        self.evolution = evolution
        self.evaluator = evaluator
        self.generations = int(generations)

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

        if self.evaluator.using_cuda:
            print(
                "GPU evaluation: enabled"
            )
            print(
                f"CUDA device: "
                f"{self.evaluator.device}"
            )

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(
                    self.evaluator.device
                )

                gpu_memory = (
                    torch.cuda.get_device_properties(
                        self.evaluator.device
                    ).total_memory
                    / (1024 ** 3)
                )

                print(
                    f"GPU: {gpu_name}"
                )

                print(
                    f"GPU memory: "
                    f"{gpu_memory:.2f} GB"
                )
        else:
            print(
                "GPU evaluation: disabled"
            )

    def train(
        self,
        market_data: pd.DataFrame,
    ) -> Agent:
        if self.generations <= 0:
            raise ValueError(
                "generations must be greater than zero."
            )

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
                key=lambda item: item[1].fitness,
                reverse=True,
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

            if (
                best_agent is None
                or generation_best.fitness
                > best_agent.fitness
            ):
                best_agent = (
                    generation_best.clone()
                )

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

            self._save_champion(
                generation_best
            )

            self._save_generation_checkpoint(
                generation_best,
                generation,
            )

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

            champion = self.evolution.evolve()

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
                self.champion_path
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

        AgentSerializer.save(
            champion,
            self.champion_path,
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

        AgentSerializer.save(
            checkpoint,
            checkpoint_path,
        )

    def _evaluate_population(
        self,
        market_data: pd.DataFrame,
    ):
        agents = self.population.agents

        if (
            self.evaluator.using_cuda
            and len(agents) > 1
        ):
            population_result = (
                self.evaluator.evaluate_population(
                    agents,
                    market_data,
                )
            )

            return population_result.results

        results = []

        total = len(agents)

        for index, agent in enumerate(
            agents,
            start=1,
        ):
            result = self.evaluator.evaluate(
                agent,
                market_data,
            )

            results.append(
                result
            )

            print(
                f"\rEvaluating agents: "
                f"{index}/{total}",
                end="",
                flush=True,
            )

        print()

        return results

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