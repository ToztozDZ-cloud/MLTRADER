from __future__ import annotations

import argparse

import pandas as pd

from agents.serialization import AgentSerializer
from backtest.engine import BacktestEngine
from validation.out_of_sample import OutOfSampleValidator
from validation.robustness import RobustnessValidator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an MLTRADER champion."
    )

    parser.add_argument(
        "--data",
        required=True,
    )

    parser.add_argument(
        "--model",
        default="models/champions/best_agent.json",
    )

    args = parser.parse_args()

    data = pd.read_csv(args.data)

    agent = AgentSerializer.load(
        args.model
    )

    engine = BacktestEngine(
        lookback=64,
        initial_cash=100_000.0,
    )

    oos = OutOfSampleValidator(
        engine
    )

    result = oos.validate(
        agent,
        data,
    )

    print()
    print("=" * 50)
    print("OUT-OF-SAMPLE VALIDATION")
    print("=" * 50)

    for key, value in result["metrics"].items():
        print(f"{key:20s}: {value}")

    robustness = RobustnessValidator()

    robustness_results = robustness.validate(
        agent,
        data,
    )

    print()
    print("=" * 50)
    print("ROBUSTNESS")
    print("=" * 50)

    for result in robustness_results:
        print()
        print(result.scenario)

        for key, value in result.metrics.items():
            print(f"  {key:18s}: {value}")


if __name__ == "__main__":
    main()