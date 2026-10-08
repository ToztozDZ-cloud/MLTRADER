from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable


def evaluate_parallel(
    agents,
    evaluator_factory: Callable,
    market_data,
    workers: int = -1,
):
    """
    Evaluate agents in separate processes.

    Each worker receives its own evaluator instance.
    """

    if workers == 0:
        raise ValueError("workers cannot be zero.")

    if workers < 0:
        executor_workers = None
    else:
        executor_workers = workers

    def evaluate_one(agent):
        evaluator = evaluator_factory()

        result = evaluator.evaluate(
            agent,
            market_data,
        )

        return agent, result

    with ProcessPoolExecutor(
        max_workers=executor_workers,
    ) as executor:
        results = list(
            executor.map(
                evaluate_one,
                agents,
            )
        )

    return results