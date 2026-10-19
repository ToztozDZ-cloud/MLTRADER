from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import platform
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.observation import ObservationBuilder
from evolution.evolution import EvolutionEngine
from evolution.population import Population
from training.episodes import TrainingEpisodeSampler
from training.evaluator import AgentEvaluator
from training.trainer import Trainer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare full-generation GPU runtimes with episode batching."
    )
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--lookback", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--episode-bars", type=int, default=128)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.population < 2 or args.episodes < 1 or args.lookback < 1:
        parser.error("population must be >=2; episodes and lookback must be >=1")
    if args.rows <= args.lookback or args.episode_bars < 1:
        parser.error("rows must exceed lookback and episode-bars must be positive")
    if (
        args.repetitions < 1
        or args.warmups < 0
        or any(size < 1 for size in args.batch_sizes)
    ):
        parser.error(
            "repetitions/batch sizes must be positive and warmups non-negative"
        )
    if not AgentEvaluator._xpu_available():
        parser.error("An available PyTorch XPU device is required for this benchmark")
    return args


def _synthetic_market_data(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.008, size=rows)))
    opening = close * (1.0 + rng.normal(0.0, 0.0005, size=rows))
    high = np.maximum(opening, close) * 1.001
    low = np.minimum(opening, close) * 0.999
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2018-01-01", periods=rows, freq="D", tz="UTC"),
            "open": opening,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(100_000, 1_000_000, size=rows),
        }
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.xpu.manual_seed_all(seed)


def _run_one_generation(
    args: argparse.Namespace, batch_episodes: int
) -> dict[str, Any]:
    _seed_everything(args.seed)
    population = Population(
        size=args.population,
        input_size=ObservationBuilder.input_size(args.lookback),
        output_size=1,
    )
    evaluator = AgentEvaluator(
        lookback=args.lookback,
        timeframe="1d",
        device="xpu",
    )
    evolution = EvolutionEngine(population)
    with tempfile.TemporaryDirectory(prefix="mltrader-generation-batch-") as temp_dir:
        output_dir = Path(temp_dir)
        trainer = Trainer(
            population=population,
            evolution=evolution,
            evaluator=evaluator,
            generations=1,
            checkpoint_dir=output_dir / "checkpoints",
            champion_dir=output_dir / "champions",
            episodes_per_agent=args.episodes,
            accelerator_episodes_per_batch=batch_episodes,
            episode_bars=args.episode_bars,
            randomize_start=True,
            randomize_instrument=False,
            randomize_market_regime=False,
            seed=args.seed,
            resume_champion=False,
            checkpoint_every=1,
            keep_champions=True,
            keep_archives=True,
        )
        generation_metrics: list[dict[str, float | int]] = []
        evaluate_population = trainer._evaluate_population

        def capture_metrics(market_data: pd.DataFrame | list[pd.DataFrame]):
            results = evaluate_population(market_data)
            best = max(results, key=lambda result: result.fitness)
            generation_metrics.append(
                {
                    "fitness": float(best.fitness),
                    "final_equity": float(best.final_equity),
                    "total_return": float(best.total_return),
                    "max_drawdown": float(best.max_drawdown),
                    "trade_count": int(best.trade_count),
                }
            )
            return results

        trainer.__dict__["_evaluate_population"] = capture_metrics
        market_data = _synthetic_market_data(args.rows, args.seed)
        sample_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 0]))
        sampled = trainer.episode_sampler.sample(
            [market_data], episodes=args.episodes, rng=sample_rng
        )
        transitions = sum(item.end_index - item.start_index for item in sampled)
        unique_transitions = len(
            {
                index
                for item in sampled
                for index in range(item.start_index, item.end_index)
            }
        )

        torch.xpu.reset_peak_memory_stats(evaluator.device)
        torch.xpu.synchronize(evaluator.device)
        started = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            champion = trainer.train(market_data)
        torch.xpu.synchronize(evaluator.device)
        elapsed = time.perf_counter() - started

        return {
            "episodes_per_batch": batch_episodes,
            "total_gpu_agent_lanes": args.population
            * min(args.episodes, batch_episodes),
            "market_episode_transitions": transitions,
            "unique_source_transition_bars": unique_transitions,
            "agent_bars": transitions * args.population,
            "full_generation_seconds": elapsed,
            "agent_bars_per_second": transitions * args.population / elapsed,
            "champion_fitness": float(champion.fitness),
            "champion_metrics": generation_metrics[-1],
            "peak_xpu_allocated_bytes": int(
                torch.xpu.max_memory_allocated(evaluator.device)
            ),
            "checkpoint_and_champion_writing_included": True,
        }


def main() -> int:
    args = _parse_args()
    market_data = _synthetic_market_data(args.rows, args.seed)
    sample_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 0]))
    sampler = TrainingEpisodeSampler(
        lookback=args.lookback,
        episode_bars=args.episode_bars,
        randomize_start=True,
        randomize_instrument=False,
        randomize_market_regime=False,
    )
    sampled = sampler.sample([market_data], episodes=args.episodes, rng=sample_rng)
    result_by_size: dict[int, list[dict[str, Any]]] = {
        size: [] for size in args.batch_sizes
    }
    for batch_size in args.batch_sizes:
        for _ in range(args.warmups):
            _run_one_generation(args, batch_size)
    for _ in range(args.repetitions):
        for batch_size in args.batch_sizes:
            result_by_size[batch_size].append(_run_one_generation(args, batch_size))

    summaries = []
    for batch_size, runs in result_by_size.items():
        times = [run["full_generation_seconds"] for run in runs]
        summaries.append(
            {
                "episodes_per_batch": batch_size,
                "lane_batch_size": args.population * min(args.episodes, batch_size),
                "median_generation_seconds": statistics.median(times),
                "min_generation_seconds": min(times),
                "max_generation_seconds": max(times),
                "median_agent_bars_per_second": statistics.median(
                    run["agent_bars_per_second"] for run in runs
                ),
                "median_peak_xpu_allocated_bytes": int(
                    statistics.median(run["peak_xpu_allocated_bytes"] for run in runs)
                ),
                "runs": runs,
            }
        )

    baseline_seconds = summaries[0]["median_generation_seconds"]
    baseline_runs = result_by_size[args.batch_sizes[0]]
    for summary in summaries:
        summary["speedup_vs_first_batch_size"] = (
            baseline_seconds / summary["median_generation_seconds"]
        )
        paired_metrics = []
        for baseline_run, batched_run in zip(
            baseline_runs, result_by_size[summary["episodes_per_batch"]]
        ):
            baseline_metrics = baseline_run["champion_metrics"]
            batched_metrics = batched_run["champion_metrics"]
            differences = {
                metric: abs(
                    float(baseline_metrics[metric]) - float(batched_metrics[metric])
                )
                for metric in (
                    "fitness",
                    "final_equity",
                    "total_return",
                    "max_drawdown",
                )
            }
            paired_metrics.append(
                {
                    "metrics_match_within_tolerance": (
                        differences["fitness"] <= 2e-3
                        and differences["final_equity"] <= 0.1
                        and differences["total_return"] <= 2e-6
                        and differences["max_drawdown"] <= 2e-6
                        and baseline_metrics["trade_count"]
                        == batched_metrics["trade_count"]
                    ),
                    "absolute_differences": differences,
                }
            )
        summary["paired_correctness_checks_vs_first_batch_size"] = paired_metrics
        summary["all_champion_metrics_match"] = all(
            item["metrics_match_within_tolerance"] for item in paired_metrics
        )

    report = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "benchmark": "full one-generation XPU training using deterministic synthetic OHLCV",
        "environment": {
            "python": sys.version,
            "os": platform.platform(),
            "pytorch": torch.__version__,
            "xpu_name": torch.xpu.get_device_name(0),
            "xpu_total_memory_bytes": int(
                torch.xpu.get_device_properties(0).total_memory
            ),
        },
        "configuration": {
            "population": args.population,
            "rows": args.rows,
            "lookback": args.lookback,
            "episodes": args.episodes,
            "episode_bars": args.episode_bars,
            "seed": args.seed,
            "warmups_per_batch_size": args.warmups,
            "repetitions": args.repetitions,
            "market_episode_transitions_per_generation": sum(
                item.end_index - item.start_index for item in sampled
            ),
            "agent_bars_definition": "sum of market transitions evaluated for every agent across sampled episodes",
            "agent_bars_per_generation": args.population
            * sum(item.end_index - item.start_index for item in sampled),
            "unique_source_transition_bars": len(
                {
                    index
                    for item in sampled
                    for index in range(item.start_index, item.end_index)
                }
            ),
        },
        "results": summaries,
        "correctness_note": "Both paths use the same seeded genomes and episode windows; paired champion metrics are checked with explicit floating-point tolerances.",
        "limitations": [
            "Synthetic daily OHLCV is used; results are a throughput comparison, not a trading-performance claim.",
            "Generation timing includes evaluation, aggregation, and checkpoint/champion serialization.",
            "Only PyTorch allocator peak is reported; system-wide GPU utilization telemetry is not available here.",
        ],
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
