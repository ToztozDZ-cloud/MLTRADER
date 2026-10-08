from __future__ import annotations

import argparse

import pandas as pd

from agents.serialization import AgentSerializer
from backtest.engine import BacktestEngine
from environment.costs import TradingCosts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an MLTRADER champion backtest."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="CSV market-data file.",
    )

    parser.add_argument(
        "--model",
        default="models/champions/best_agent.json",
        help="Champion model path.",
    )

    args = parser.parse_args()

    data = pd.read_csv(
        args.data
    )

    agent = AgentSerializer.load(
        args.model
    )

    engine = BacktestEngine(
        lookback=64,
        initial_cash=100_000.0,
        costs=TradingCosts(
            commission_rate=0.0005,
            spread_rate=0.0002,
            slippage_rate=0.0005,
        ),
        allow_short=False,
        max_position_fraction=1.0,
        max_leverage=1.0,
        execute_on_next_bar=True,
    )

    result = engine.run(
        agent,
        data,
    )

    print()
    print("=" * 60)
    print("MLTRADER BACKTEST")
    print("=" * 60)

    metrics = result["metrics"]

    print(
        f"{'Total return':20s}: "
        f"{metrics['total_return']:.2%}"
    )

    print(
        f"{'Max drawdown':20s}: "
        f"{metrics['max_drawdown']:.2%}"
    )

    print(
        f"{'Volatility':20s}: "
        f"{metrics['volatility']:.6f}"
    )

    print(
        f"{'Sharpe':20s}: "
        f"{metrics['sharpe']:.4f}"
    )

    print(
        f"{'Calmar':20s}: "
        f"{metrics['calmar']:.4f}"
    )

    print(
        f"{'Turnover':20s}: "
        f"{metrics['turnover']:.4f}"
    )

    print(
        f"{'Final equity':20s}: "
        f"${metrics['final_equity']:,.2f}"
    )

    print(
        f"{'Fitness':20s}: "
        f"{metrics['fitness']:.6f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()