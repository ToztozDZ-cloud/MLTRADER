<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 7ce9a3260bf1e51b5cc30c7b002019690ed69fb8
from __future__ import annotations

import pandas as pd

from agents.agent import Agent
from backtest.engine import BacktestEngine


class OutOfSampleValidator:
    """
    Runs the frozen champion on data that was not used for evolution.
    """

    def __init__(
        self,
        engine: BacktestEngine,
    ) -> None:
        self.engine = engine

    def validate(
        self,
        agent: Agent,
        market_data: pd.DataFrame,
    ) -> dict:
        return self.engine.run(
            agent,
            market_data,
<<<<<<< HEAD
=======
=======
from __future__ import annotations

import pandas as pd

from agents.agent import Agent
from backtest.engine import BacktestEngine


class OutOfSampleValidator:
    """
    Runs the frozen champion on data that was not used for evolution.
    """

    def __init__(
        self,
        engine: BacktestEngine,
    ) -> None:
        self.engine = engine

    def validate(
        self,
        agent: Agent,
        market_data: pd.DataFrame,
    ) -> dict:
        return self.engine.run(
            agent,
            market_data,
>>>>>>> origin/master
>>>>>>> 7ce9a3260bf1e51b5cc30c7b002019690ed69fb8
        )