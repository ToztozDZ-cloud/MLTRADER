from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Callable


def _evaluate_one(agent, evaluator_factory: Callable, market_data):
    evaluator = evaluator_factory()
    result = evaluator.evaluate(agent, market_data)
    return agent, result


def evaluate_parallel(
    agents,
    evaluator_factory: Callable,
    market_data,
    workers: int = -1,
):
    """
    Evaluate agents in separate processes.

    Each worker creates its own evaluator instance. ``evaluator_factory`` and
    the agent and market-data objects must be picklable for process spawning;
    lambdas, closures, and other local objects are therefore unsupported.
    """

    if workers == 0:
        raise ValueError("workers cannot be zero.")

    if workers < 0:
        executor_workers = None
    else:
        executor_workers = workers

    worker = partial(
        _evaluate_one,
        evaluator_factory=evaluator_factory,
        market_data=market_data,
    )

    with ProcessPoolExecutor(
        max_workers=executor_workers,
    ) as executor:
        results = list(
            executor.map(
                worker,
                agents,
            )
        )

    return results