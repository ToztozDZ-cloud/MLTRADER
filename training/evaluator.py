from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Callable, Sequence, TypedDict

import numpy as np
import pandas as pd
import torch

from agents.agent import Agent
from agents.genome import ConnectionGene, NodeGene
from environment.costs import TradingCosts
from environment.exchange import Exchange
from environment.execution import ExecutionEngine
from environment.market import MarketEnvironment
from environment.observation import ObservationBuilder
from environment.portfolio import Portfolio

from .fitness import FitnessCalculator
from .fitness import periods_per_year_for_timeframe


@dataclass
class EvaluationResult:
    fitness: float
    final_equity: float
    total_return: float
    max_drawdown: float
    turnover: float
    equity_curve: list[float]
    trade_count: int = 0
    transaction_costs: float = 0.0
    average_gross_exposure: float = 0.0
    max_gross_exposure: float = 0.0


@dataclass
class PopulationEvaluationResult:
    results: list[EvaluationResult]


class CompiledLayer(TypedDict):
    nodes: np.ndarray
    matrix: np.ndarray
    activations: np.ndarray


class CompiledAgent(TypedDict):
    node_count: int
    input_indices: list[int]
    bias_indices: list[int]
    output_indices: list[int]
    layers: list[CompiledLayer]


class AgentEvaluator:
    def __init__(
        self,
        lookback: int = 64,
        initial_cash: float = 100_000.0,
        costs: TradingCosts | None = None,
        allow_short: bool = False,
        max_position_fraction: float = 1.0,
        max_leverage: float = 1.0,
        execute_on_next_bar: bool = True,
        execution_delay_bars: int = 0,
        fitness_calculator: FitnessCalculator | None = None,
        device: str | torch.device | None = None,
        timeframe: str = "1d",
        exchange_timezone: str = "America/New_York",
        flatten_before_close_minutes: int = 10,
    ) -> None:
        self.lookback = int(lookback)
        self.initial_cash = float(initial_cash)
        if self.lookback <= 0:
            raise ValueError("lookback must be positive")
        if not np.isfinite(self.initial_cash) or self.initial_cash <= 0.0:
            raise ValueError("initial_cash must be finite and positive")

        self.costs = costs or TradingCosts()

        self.allow_short = bool(
            allow_short
        )

        self.max_position_fraction = float(
            max_position_fraction
        )
        if (
            not np.isfinite(self.max_position_fraction)
            or not 0.0 < self.max_position_fraction <= 1.0
        ):
            raise ValueError("max_position_fraction must be finite and in (0, 1]")

        self.max_leverage = float(
            max_leverage
        )
        if not np.isfinite(self.max_leverage) or self.max_leverage < 0.0:
            raise ValueError("max_leverage must be finite and non-negative")

        self.execute_on_next_bar = bool(
            execute_on_next_bar
        )
        if not self.execute_on_next_bar:
            raise ValueError(
                "Only completed-bar signal with next-bar-open execution is supported; "
                "execute_on_next_bar=False is invalid."
            )
        self.execution_delay_bars = int(execution_delay_bars)
        if self.execution_delay_bars < 0:
            raise ValueError("execution_delay_bars must be non-negative")
        self.timeframe = str(timeframe).lower()
        self.exchange_timezone = str(exchange_timezone)
        self.flatten_before_close_minutes = max(0, int(flatten_before_close_minutes))
        self.timeframe_minutes = None
        if self.timeframe.endswith(("m", "h")) and self.timeframe[:-1].isdigit():
            interval = int(self.timeframe[:-1])
            self.timeframe_minutes = interval * (60 if self.timeframe.endswith("h") else 1)
        self.periods_per_year = periods_per_year_for_timeframe(self.timeframe)

        self.fitness_calculator = (
            fitness_calculator
            or FitnessCalculator(periods_per_year=self.periods_per_year)
        )

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        elif str(device) == "auto":
            if self._xpu_available():
                self.device = torch.device("xpu")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        if self.device.type == "xpu" and not self._xpu_available():
            raise RuntimeError(
                "Intel GPU was requested, but PyTorch XPU is unavailable. "
                "Install the official PyTorch XPU build and verify the Intel GPU driver."
            )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        if self.device.type not in {"cpu", "cuda", "xpu"}:
            raise ValueError(f"Unsupported evaluation device: {self.device.type}")

    @staticmethod
    def _xpu_available() -> bool:
        xpu = getattr(torch, "xpu", None)
        if xpu is None:
            return False
        try:
            return bool(xpu.is_available())
        except (OSError, RuntimeError):
            return False

    @property
    def using_cuda(self) -> bool:
        return self.device.type == "cuda"

    @property
    def using_xpu(self) -> bool:
        return self.device.type == "xpu"

    @property
    def using_accelerator(self) -> bool:
        return self.device.type in {"cuda", "xpu"}

    def _synchronize_device(self) -> None:
        if self.using_cuda:
            torch.cuda.synchronize(self.device)
        elif self.using_xpu:
            torch.xpu.synchronize(self.device)

    # ==========================================================
    # CPU REFERENCE EVALUATION
    # ==========================================================

    def evaluate(
        self,
        agent: Agent,
        market_data: pd.DataFrame,
        start_step: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> EvaluationResult:
        market = MarketEnvironment(
            data=market_data,
            initial_cash=self.initial_cash,
        )

        if len(market) <= self.lookback:
            raise ValueError(
                "Market data must contain more rows "
                "than the lookback window."
            )

        if start_step is None:
            start_step = self.lookback - 1

        if start_step < self.lookback - 1:
            raise ValueError(
                "start_step must be at least "
                "lookback - 1."
            )

        if start_step >= len(market) - 1:
            raise ValueError(
                "Not enough market data after start_step."
            )

        portfolio = Portfolio(
            initial_cash=self.initial_cash
        )

        exchange = Exchange(
            portfolio=portfolio,
            costs=self.costs,
            allow_short=self.allow_short,
            max_leverage=self.max_leverage,
        )

        execution = ExecutionEngine(
            exchange=exchange
        )

        observation_builder = ObservationBuilder(
            lookback=self.lookback
        )

        agent.reset()

        market.reset(start_step)

        equity_curve = [
            portfolio.equity(
                market.current_price
            )
        ]

        turnover = 0.0
        trade_count = 0
        transaction_costs = 0.0
        gross_exposures = []
        invalid_output_seen = False
        pending_targets: dict[int, float] = {}
        total_steps = len(market) - start_step - 1
        completed_steps = 0
        progress_interval = max(1, total_steps // 100)

        while not market.done:
            observation = observation_builder.build(
                market,
                portfolio,
            )

            try:
                action_values = np.asarray(agent.act(observation), dtype=np.float64).reshape(-1)
                if action_values.size != 1 or not np.isfinite(action_values[0]):
                    raise ValueError("agent output must be one finite target-position value")
                target_fraction = float(np.clip(action_values[0], -1.0, 1.0))
            except Exception:
                target_fraction = 0.0
                invalid_agent_output = True
                invalid_output_seen = True
            else:
                invalid_agent_output = False

            if not self.allow_short:
                target_fraction = max(
                    0.0,
                    target_fraction,
                )

            # Intraday positions must be flat before the regular session ends.
            # Signal is computed from the current completed bar; a flatten order
            # is then simulated at the next bar's open, not its close.
            if (
                self.timeframe_minutes is not None
                and self.flatten_before_close_minutes > 0
                and "timestamp" in market.current_bar.index
            ):
                stamp = pd.Timestamp(market.current_bar["timestamp"])
                if stamp.tzinfo is None:
                    stamp = stamp.tz_localize("UTC")
                local_stamp = stamp.tz_convert(self.exchange_timezone)
                minutes = local_stamp.hour * 60 + local_stamp.minute
                close_cutoff = 16 * 60 - self.flatten_before_close_minutes - self.timeframe_minutes
                if minutes >= close_cutoff:
                    target_fraction = 0.0

            target_fraction *= self.max_position_fraction

            current_price = market.current_price

            current_equity = portfolio.equity(current_price)
            if not np.isfinite(current_equity) or current_equity <= 0.0:
                break

            target_value = (
                current_equity
                * target_fraction
            )

            target_position = (
                target_value / current_price
                if current_price > 0.0
                else 0.0
            )

            next_bar = market.step()

            if next_bar is None:
                break

            if not invalid_agent_output:
                pending_targets[
                    market.current_step + self.execution_delay_bars
                ] = target_position
            due_target = pending_targets.pop(market.current_step, None)
            if due_target is not None:
                execution_price = float(next_bar["open"])
                order_result = execution.execute_target_position(
                    target_position=due_target,
                    market_price=execution_price,
                )
                if order_result.executed_quantity != 0.0:
                    trade_count += 1
                    transaction_costs += self.costs.estimated_execution_cost(order_result.notional)
                    turnover += abs(order_result.notional) / max(
                        portfolio.equity(execution_price), 1.0
                    )

            equity_curve.append(
                portfolio.equity(
                    market.current_price
                )
            )
            marked_equity = equity_curve[-1]
            gross_exposures.append(
                abs(portfolio.position * market.current_price) / max(marked_equity, 1.0)
                if np.isfinite(marked_equity) and marked_equity > 0.0
                else float("inf")
            )

            completed_steps += 1
            if progress_callback is not None and (
                completed_steps % progress_interval == 0
                or completed_steps >= total_steps
            ):
                progress_callback(completed_steps, total_steps)

        if portfolio.position != 0.0:
            final_price = market.current_price

            liquidation = (
                execution.execute_target_position(
                    target_position=0.0,
                    market_price=final_price,
                )
            )

            if liquidation.executed_quantity != 0.0:
                trade_count += 1
                transaction_costs += self.costs.estimated_execution_cost(liquidation.notional)

            turnover += (
                abs(liquidation.notional)
                / max(
                    portfolio.equity(
                        final_price
                    ),
                    1.0,
                )
            )

            equity_curve[-1] = (
                portfolio.equity(
                    final_price
                )
            )

        evaluation_result = self._calculate_result(
            equity_curve=equity_curve,
            turnover=turnover,
            trade_count=trade_count,
            transaction_costs=transaction_costs,
            gross_exposures=gross_exposures,
        )
        if invalid_output_seen:
            evaluation_result.fitness = float("-inf")

        agent.fitness = evaluation_result.fitness

        return evaluation_result

    # ==========================================================
    # GPU POPULATION EVALUATION
    # ==========================================================

    def evaluate_population(
        self,
        agents: Sequence[Agent],
        market_data: pd.DataFrame,
        start_step: int | None = None,
    ) -> PopulationEvaluationResult:
        if not agents:
            return PopulationEvaluationResult(
                results=[]
            )

        market_data = MarketEnvironment(
            data=market_data,
            initial_cash=self.initial_cash,
        ).data

        if start_step is None:
            start_step = self.lookback - 1

        if len(market_data) <= self.lookback:
            raise ValueError(
                "Market data must contain more rows "
                "than the lookback window."
            )

        if start_step < self.lookback - 1:
            raise ValueError(
                "start_step must be at least "
                "lookback - 1."
            )

        if start_step >= len(market_data) - 1:
            raise ValueError(
                "Not enough market data after start_step."
            )

        print()
        print(
            f"{self.device.type.upper()} population evaluation | "
            f"{len(agents)} agents | "
            f"{len(market_data)} market bars"
        )

        preprocessing_start = time.perf_counter()

        observations = self._precompute_observations(
            market_data=market_data,
            start_step=start_step,
        )

        preprocessing_elapsed = (
            time.perf_counter()
            - preprocessing_start
        )

        print(
            f"Preprocessing complete | "
            f"{len(observations)} evaluation bars | "
            f"{preprocessing_elapsed:.2f}s"
        )

        prices = self._extract_prices(market_data)
        opens = market_data["open"].to_numpy(dtype=np.float32)
        force_flatten = self._precompute_flatten_mask(market_data)

        observation_tensor = torch.from_numpy(
            observations
        ).to(
            self.device,
            dtype=torch.float32,
        )

        price_tensor = torch.from_numpy(
            prices
        ).to(
            self.device,
            dtype=torch.float32,
        )

        open_tensor = torch.from_numpy(opens).to(
            self.device,
            dtype=torch.float32,
        )
        flatten_tensor = torch.from_numpy(force_flatten).to(self.device)

        self._synchronize_device()

        compiled = self._compile_population(
            agents
        )

        results = self._simulate_population(
            agents=agents,
            observations=observation_tensor,
            prices=price_tensor,
            opens=open_tensor,
            force_flatten=flatten_tensor,
            compiled=compiled,
            start_step=start_step,
        )

        return PopulationEvaluationResult(
            results=results
        )

    # ==========================================================
    # MARKET PREPROCESSING
    # ==========================================================

    def _precompute_observations(
        self,
        market_data: pd.DataFrame,
        start_step: int,
    ) -> np.ndarray:
        data = market_data.copy()

        data.columns = [
            str(column).strip().lower()
            for column in data.columns
        ]

        required = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        missing = [
            column
            for column in required
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing market columns: {missing}"
            )

        values = (
            data[required]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .to_numpy(
                dtype=np.float32
            )
        )

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                "Market data contains non-finite values."
            )

        observations = []

        total_rows = len(values)

        total_steps = max(
            total_rows - 1 - start_step,
            0,
        )

        progress_interval = max(
            1,
            total_steps // 100,
        )

        preprocessing_start = time.perf_counter()

        for index, step in enumerate(
            range(
                start_step,
                total_rows - 1,
            ),
            start=1,
        ):
            window_start = max(
                0,
                step - self.lookback + 1,
            )

            window = values[
                window_start : step + 1
            ]

            if len(window) < self.lookback:
                padding = np.repeat(
                    window[:1],
                    self.lookback - len(window),
                    axis=0,
                )

                window = np.concatenate(
                    [
                        padding,
                        window,
                    ],
                    axis=0,
                )

            window = window[
                -self.lookback :
            ]

            normalized = ObservationBuilder._normalize_market_data(window)

            observation = np.concatenate(
                [
                    normalized.reshape(-1),
                    np.array(
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        dtype=np.float32,
                    ),
                ]
            )

            observations.append(
                observation
            )

            if (
                index == total_steps
                or index % progress_interval == 0
            ):
                elapsed = (
                    time.perf_counter()
                    - preprocessing_start
                )

                rate = (
                    index / elapsed
                    if elapsed > 0.0
                    else 0.0
                )

                remaining = (
                    total_steps - index
                )

                eta = (
                    remaining / rate
                    if rate > 0.0
                    else 0.0
                )

                self._print_progress_bar(
                    prefix="Preprocessing",
                    completed=index,
                    total=total_steps,
                    elapsed=elapsed,
                    rate=rate,
                    eta=eta,
                )

        if total_steps == 0:
            print(
                "Preprocessing | no evaluation bars"
            )

        return np.asarray(
            observations,
            dtype=np.float32,
        )

    def _precompute_flatten_mask(
        self,
        market_data: pd.DataFrame,
    ) -> np.ndarray:
        mask = np.zeros(len(market_data), dtype=np.bool_)
        if (
            self.timeframe_minutes is None
            or self.flatten_before_close_minutes <= 0
            or "timestamp" not in market_data.columns
        ):
            return mask

        cutoff = 16 * 60 - self.flatten_before_close_minutes - self.timeframe_minutes
        for index, value in enumerate(market_data["timestamp"]):
            stamp = pd.Timestamp(value)
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize("UTC")
            local_stamp = stamp.tz_convert(self.exchange_timezone)
            mask[index] = local_stamp.hour * 60 + local_stamp.minute >= cutoff
        return mask

    @staticmethod
    def _extract_prices(
        market_data: pd.DataFrame,
    ) -> np.ndarray:
        columns = {
            str(column).strip().lower(): column
            for column in market_data.columns
        }

        if "close" not in columns:
            raise ValueError(
                "Market data is missing the close column."
            )

        prices = (
            pd.to_numeric(
                market_data[
                    columns["close"]
                ],
                errors="coerce",
            )
            .to_numpy(
                dtype=np.float32
            )
        )

        if not np.all(
            np.isfinite(prices)
        ):
            raise ValueError(
                "Market close prices contain non-finite values."
            )

        return prices

    # ==========================================================
    # PROGRESS DISPLAY
    # ==========================================================

    @staticmethod
    def _print_progress_bar(
        prefix: str,
        completed: int,
        total: int,
        elapsed: float,
        rate: float,
        eta: float,
        width: int = 36,
    ) -> None:
        if total <= 0:
            percentage = 100.0
            completed = 0
            total = 0
        else:
            percentage = (
                completed
                / total
                * 100.0
            )

        filled = (
            int(
                width
                * completed
                / total
            )
            if total > 0
            else width
        )

        filled = max(
            0,
            min(
                width,
                filled,
            ),
        )

        bar = (
            "█" * filled
            + "░" * (width - filled)
        )

        eta_text = (
            AgentEvaluator._format_duration(
                eta
            )
        )

        elapsed_text = (
            AgentEvaluator._format_duration(
                elapsed
            )
        )

        line = (
            f"\r{prefix} |"
            f"{bar}| "
            f"{percentage:6.2f}% | "
            f"{completed:,}/{total:,} | "
            f"{rate:7.2f} bar/s | "
            f"Elapsed {elapsed_text} | "
            f"ETA {eta_text}"
        )

        sys.stdout.write(line)
        sys.stdout.flush()

        if (
            total > 0
            and completed >= total
        ):
            sys.stdout.write("\n")
            sys.stdout.flush()

    @staticmethod
    def _format_duration(
        seconds: float,
    ) -> str:
        seconds = max(
            0.0,
            float(seconds),
        )

        if seconds < 60.0:
            return f"{seconds:.1f}s"

        minutes = int(
            seconds // 60
        )

        remaining_seconds = int(
            seconds % 60
        )

        if minutes < 60:
            return (
                f"{minutes}m "
                f"{remaining_seconds:02d}s"
            )

        hours = int(
            minutes // 60
        )

        remaining_minutes = (
            minutes % 60
        )

        return (
            f"{hours}h "
            f"{remaining_minutes:02d}m"
        )

    # ==========================================================
    # GENOME COMPILATION
    # ==========================================================

    def _compile_population(
        self,
        agents: Sequence[Agent],
    ) -> dict:
        """
        Compile every genome into padded batched topological layers.

        Input and bias nodes are initialized directly in the forward
        pass and are therefore excluded from executable layers.

        Each executable layer contains only hidden/output nodes at
        the same topological depth.
        """

        compiled_agents: list[CompiledAgent] = []

        max_nodes = 0
        max_inputs = 0
        max_biases = 0
        max_outputs = 0
        max_depth = 0

        for agent in agents:
            genome = agent.genome

            order = self._topological_order(
                genome
            )

            node_ids = [
                node_id
                for node_id in order
            ]

            node_index = {
                node_id: index
                for index, node_id in enumerate(
                    node_ids
                )
            }

            node_by_id = {
                node.node_id: node
                for node in genome.nodes
            }

            max_nodes = max(
                max_nodes,
                len(node_ids),
            )

            input_indices = [
                node_index[node_id]
                for node_id in order
                if node_by_id[node_id].node_type
                == "input"
            ]

            bias_indices = [
                node_index[node_id]
                for node_id in order
                if node_by_id[node_id].node_type
                == "bias"
            ]

            output_indices = [
                node_index[node_id]
                for node_id in order
                if node_by_id[node_id].node_type
                == "output"
            ]

            max_inputs = max(
                max_inputs,
                len(input_indices),
            )

            max_biases = max(
                max_biases,
                len(bias_indices),
            )

            max_outputs = max(
                max_outputs,
                len(output_indices),
            )

            incoming: dict[int, list[ConnectionGene]] = {
                node_id: []
                for node_id in order
            }

            for connection in genome.connections:
                if not connection.enabled:
                    continue

                source_id = connection.source
                target_id = connection.target

                if source_id not in node_index:
                    continue

                if target_id not in node_index:
                    continue

                incoming[
                    target_id
                ].append(
                    connection
                )

            depths: dict[int, int] = {}

            for node_id in order:
                node = node_by_id[node_id]

                if node.node_type in (
                    "input",
                    "bias",
                ):
                    depths[node_id] = 0
                    continue

                node_depth = 1

                for connection in incoming[
                    node_id
                ]:
                    source_depth = depths.get(
                        connection.source,
                        0,
                    )

                    node_depth = max(
                        node_depth,
                        source_depth + 1,
                    )

                depths[node_id] = node_depth

            depth_groups: dict[int, list[NodeGene]] = {}

            for node_id in order:
                node = node_by_id[node_id]

                if node.node_type in (
                    "input",
                    "bias",
                ):
                    continue

                depth = depths[node_id]

                depth_groups.setdefault(
                    depth,
                    [],
                ).append(
                    node
                )

            max_depth = max(
                max_depth,
                len(depth_groups),
            )

            layers: list[CompiledLayer] = []

            for depth in sorted(
                depth_groups
            ):
                target_nodes = depth_groups[
                    depth
                ]

                target_indices = [
                    node_index[
                        node.node_id
                    ]
                    for node in target_nodes
                ]

                matrix: np.ndarray = np.zeros(
                    (
                        len(target_nodes),
                        len(node_ids),
                    ),
                    dtype=np.float32,
                )

                activations: np.ndarray = np.zeros(
                    len(target_nodes),
                    dtype=np.int64,
                )

                for target_position, node in enumerate(
                    target_nodes
                ):
                    activation = str(
                        node.activation
                    ).lower()

                    if activation == "tanh":
                        activations[
                            target_position
                        ] = 1

                    elif activation == "relu":
                        activations[
                            target_position
                        ] = 2

                    elif activation == "sigmoid":
                        activations[
                            target_position
                        ] = 3

                    elif activation == "linear":
                        activations[
                            target_position
                        ] = 4

                    else:
                        activations[
                            target_position
                        ] = 1

                    for connection in incoming[
                        node.node_id
                    ]:
                        source_index = node_index.get(
                            connection.source
                        )

                        if source_index is None:
                            continue

                        matrix[
                            target_position,
                            source_index,
                        ] = float(
                            connection.weight
                        )

                layers.append(
                    {
                        "nodes": np.asarray(
                            target_indices,
                            dtype=np.int64,
                        ),
                        "matrix": matrix,
                        "activations": activations,
                    }
                )

            compiled_agents.append(
                {
                    "node_count": len(node_ids),
                    "input_indices": input_indices,
                    "bias_indices": bias_indices,
                    "output_indices": output_indices,
                    "layers": layers,
                }
            )

        population_size = len(
            compiled_agents
        )

        layer_matrices: list[torch.Tensor] = []
        layer_nodes: list[torch.Tensor] = []
        layer_activations: list[torch.Tensor] = []
        layer_masks: list[torch.Tensor] = []

        for depth_index in range(
            max_depth
        ):
            target_count = 0

            for description in compiled_agents:
                agent_layers = description[
                    "layers"
                ]

                if depth_index >= len(agent_layers):
                    continue

                target_count = max(
                    target_count,
                    len(
                        agent_layers[
                            depth_index
                        ][
                            "nodes"
                        ]
                    ),
                )

            layer_matrix_batch = torch.zeros(
                (
                    population_size,
                    target_count,
                    max_nodes,
                ),
                dtype=torch.float32,
                device=self.device,
            )

            layer_node_batch = torch.zeros(
                (
                    population_size,
                    target_count,
                ),
                dtype=torch.long,
                device=self.device,
            )

            layer_activation_batch = torch.zeros(
                (
                    population_size,
                    target_count,
                ),
                dtype=torch.long,
                device=self.device,
            )

            layer_mask_batch = torch.zeros(
                (
                    population_size,
                    target_count,
                ),
                dtype=torch.bool,
                device=self.device,
            )

            for agent_index, description in enumerate(
                compiled_agents
            ):
                agent_layers = description[
                    "layers"
                ]

                if depth_index >= len(agent_layers):
                    continue

                layer = agent_layers[
                    depth_index
                ]

                local_nodes = layer[
                    "nodes"
                ]

                local_matrix = layer[
                    "matrix"
                ]

                local_activations = layer[
                    "activations"
                ]

                local_target_count = len(
                    local_nodes
                )

                local_width = (
                    local_matrix.shape[1]
                )

                if local_width > max_nodes:
                    raise RuntimeError(
                        "Compiled layer matrix is wider "
                        "than population max_nodes: "
                        f"{local_width} > {max_nodes}"
                    )

                layer_matrix_batch[
                    agent_index,
                    :local_target_count,
                    :local_width,
                ] = torch.from_numpy(
                    local_matrix
                ).to(
                    self.device
                )

                layer_node_batch[
                    agent_index,
                    :local_target_count,
                ] = torch.from_numpy(
                    local_nodes
                ).to(
                    self.device
                )

                layer_activation_batch[
                    agent_index,
                    :local_target_count,
                ] = torch.from_numpy(
                    local_activations
                ).to(
                    self.device
                )

                layer_mask_batch[
                    agent_index,
                    :local_target_count,
                ] = True

            layer_matrices.append(
                layer_matrix_batch
            )

            layer_nodes.append(
                layer_node_batch
            )

            layer_activations.append(
                layer_activation_batch
            )

            layer_masks.append(
                layer_mask_batch
            )

        input_tensor = torch.full(
            (
                population_size,
                max_inputs,
            ),
            -1,
            dtype=torch.long,
            device=self.device,
        )

        bias_tensor = torch.full(
            (
                population_size,
                max_biases,
            ),
            -1,
            dtype=torch.long,
            device=self.device,
        )

        output_tensor = torch.full(
            (
                population_size,
                max_outputs,
            ),
            -1,
            dtype=torch.long,
            device=self.device,
        )

        node_counts = torch.zeros(
            population_size,
            dtype=torch.long,
            device=self.device,
        )

        for agent_index, description in enumerate(
            compiled_agents
        ):
            node_counts[
                agent_index
            ] = description[
                "node_count"
            ]

            input_indices = description[
                "input_indices"
            ]

            if input_indices:
                input_tensor[
                    agent_index,
                    :len(input_indices),
                ] = torch.tensor(
                    input_indices,
                    dtype=torch.long,
                    device=self.device,
                )

            bias_indices = description[
                "bias_indices"
            ]

            if bias_indices:
                bias_tensor[
                    agent_index,
                    :len(bias_indices),
                ] = torch.tensor(
                    bias_indices,
                    dtype=torch.long,
                    device=self.device,
                )

            output_indices = description[
                "output_indices"
            ]

            if output_indices:
                output_tensor[
                    agent_index,
                    :len(output_indices),
                ] = torch.tensor(
                    output_indices,
                    dtype=torch.long,
                    device=self.device,
                )

        return {
            "matrices": layer_matrices,
            "nodes": layer_nodes,
            "activations": layer_activations,
            "masks": layer_masks,
            "input_indices": input_tensor,
            "bias_indices": bias_tensor,
            "output_indices": output_tensor,
            "node_counts": node_counts,
            "max_nodes": max_nodes,
            "max_inputs": max_inputs,
            "max_biases": max_biases,
            "max_outputs": max_outputs,
            "max_depth": max_depth,
        }

    def _topological_order(
        self,
        genome,
    ) -> list:
        """
        Return genome node IDs in topological order.

        Only enabled connections participate in the graph.
        Raises RuntimeError if the genome contains a cycle.
        """

        nodes = {
            node.node_id
            for node in genome.nodes
        }

        adjacency: dict[int, list[int]] = {
            node_id: []
            for node_id in nodes
        }

        indegree: dict[int, int] = {
            node_id: 0
            for node_id in nodes
        }

        for connection in genome.connections:
            if not connection.enabled:
                continue

            source = connection.source
            target = connection.target

            if source not in nodes:
                raise RuntimeError(
                    "Connection references missing "
                    f"source node: {source}"
                )

            if target not in nodes:
                raise RuntimeError(
                    "Connection references missing "
                    f"target node: {target}"
                )

            adjacency[source].append(
                target
            )

            indegree[target] += 1

        queue = sorted(
            node_id
            for node_id, degree in indegree.items()
            if degree == 0
        )

        order = []

        while queue:
            node_id = queue.pop(0)

            order.append(
                node_id
            )

            for target in adjacency[node_id]:
                indegree[target] -= 1

                if indegree[target] == 0:
                    queue.append(target)

            queue.sort()

        if len(order) != len(nodes):
            raise RuntimeError(
                "Genome contains a cycle or "
                "invalid dependency graph. "
                f"Resolved {len(order)} of "
                f"{len(nodes)} nodes."
            )

        return order

    # ==========================================================
    # GPU SIMULATION
    # ==========================================================

    def _simulate_population(
        self,
        agents: Sequence[Agent],
        observations: torch.Tensor,
        prices: torch.Tensor,
        opens: torch.Tensor,
        force_flatten: torch.Tensor,
        compiled: dict,
        start_step: int,
    ) -> list[EvaluationResult]:
        population_size = len(
            agents
        )

        cash = torch.full(
            (
                population_size,
            ),
            self.initial_cash,
            dtype=torch.float32,
            device=self.device,
        )

        position = torch.zeros(
            population_size,
            dtype=torch.float32,
            device=self.device,
        )

        average_entry = torch.zeros(
            population_size,
            dtype=torch.float32,
            device=self.device,
        )

        turnover = torch.zeros(
            population_size,
            dtype=torch.float32,
            device=self.device,
        )

        trade_count = torch.zeros(
            population_size,
            dtype=torch.long,
            device=self.device,
        )

        transaction_costs = torch.zeros(
            population_size,
            dtype=torch.float32,
            device=self.device,
        )

        invalid_output_seen = torch.zeros(
            population_size,
            dtype=torch.bool,
            device=self.device,
        )

        initial_equity = cash.clone()

        equity_history = [
            initial_equity.clone()
        ]

        gross_exposure_history = []
        pending_targets: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        steps = observations.shape[0]

        progress_interval = max(
            1,
            steps // 10,
        )

        simulation_start = time.perf_counter()

        self._synchronize_device()

        print(
            f"{self.device.type.upper()} evaluation | "
            f"{population_size} agents | "
            f"{steps:,} bars"
        )

        for relative_step in range(
            steps
        ):
            price_index = (
                start_step
                + relative_step
            )

            current_price = prices[
                price_index
            ]

            equity = (
                cash
                + position
                * current_price
            )

            safe_equity = torch.where(
                equity > 0.0,
                equity,
                torch.ones_like(equity),
            )

            position_fraction = (
                position
                * current_price
                / safe_equity
            )

            cash_fraction = (
                cash
                / safe_equity
            )

            unrealized_fraction = (
                (
                    current_price
                    - average_entry
                )
                * position
                / safe_equity
            )

            batch_observation = (
                observations[
                    relative_step
                ]
                .unsqueeze(0)
                .expand(
                    population_size,
                    -1,
                )
                .clone()
            )

            batch_observation[:, -4] = cash_fraction
            batch_observation[:, -3] = position_fraction
            batch_observation[:, -2] = position / torch.maximum(
                torch.ones_like(position),
                torch.abs(position),
            )
            batch_observation[:, -1] = unrealized_fraction

            actions = self._forward_population(
                batch_observation,
                compiled,
            )

            invalid_output = torch.isnan(actions)
            invalid_output_seen |= invalid_output
            target_fraction = torch.clamp(
                torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0),
                -1.0,
                1.0,
            )

            if not self.allow_short:
                target_fraction = torch.maximum(
                    target_fraction,
                    torch.zeros_like(
                        target_fraction
                    ),
                )

            target_fraction *= (
                self.max_position_fraction
            )

            target_fraction = torch.where(
                force_flatten[price_index],
                torch.zeros_like(target_fraction),
                target_fraction,
            )

            target_position = (
                equity
                * target_fraction
                / torch.clamp(
                    current_price,
                    min=1e-6,
                )
            )

            next_index = price_index + 1
            due_index = next_index + self.execution_delay_bars
            pending_targets[due_index] = (
                target_position,
                ~invalid_output,
            )
            due_target = pending_targets.pop(next_index, None)
            if due_target is not None:
                scheduled_position, scheduled_mask = due_target
                effective_target = torch.where(
                    scheduled_mask,
                    scheduled_position,
                    position,
                )
                average_entry = self._execute_batch(
                    cash=cash,
                    position=position,
                    average_entry=average_entry,
                    turnover=turnover,
                    trade_count=trade_count,
                    transaction_costs=transaction_costs,
                    target_position=effective_target,
                    execution_price=opens[next_index],
                )

            marked_price = prices[next_index]
            marked_equity = cash + position * marked_price
            equity_history.append(marked_equity.clone())
            gross_exposure_history.append(
                torch.where(
                    torch.isfinite(marked_equity) & (marked_equity > 0.0),
                    torch.abs(position * marked_price)
                    / torch.clamp(marked_equity, min=1.0),
                    torch.full_like(marked_equity, float("inf")),
                ).clone()
            )

            completed = (
                relative_step + 1
            )

            if (
                completed == steps
                or completed % progress_interval == 0
            ):
                self._synchronize_device()

                elapsed = (
                    time.perf_counter()
                    - simulation_start
                )

                rate = (
                    completed / elapsed
                    if elapsed > 0.0
                    else 0.0
                )

                remaining = (
                    steps - completed
                )

                eta = (
                    remaining / rate
                    if rate > 0.0
                    else 0.0
                )

                self._print_progress_bar(
                    prefix=f"{self.device.type.upper()} evaluation",
                    completed=completed,
                    total=steps,
                    elapsed=elapsed,
                    rate=rate,
                    eta=eta,
                )

        self._synchronize_device()

        final_price = prices[-1]

        average_entry = (
            self._execute_batch(
                cash=cash,
                position=position,
                average_entry=average_entry,
                turnover=turnover,
                trade_count=trade_count,
                transaction_costs=transaction_costs,
                target_position=torch.zeros_like(
                    position
                ),
                execution_price=final_price,
            )
        )

        self._synchronize_device()

        final_equity = (
            cash
            + position
            * final_price
        )

        if equity_history:
            equity_history[-1] = (
                final_equity.clone()
            )

        equity_matrix = torch.stack(
            equity_history,
            dim=0,
        )

        self._synchronize_device()

        total_elapsed = (
            time.perf_counter()
            - simulation_start
        )

        print(
            f"{self.device.type.upper()} evaluation complete | "
            f"{steps:,} bars | "
            f"{total_elapsed:.2f}s | "
            f"{steps / total_elapsed:.2f} bar/s"
        )

        turnover_cpu = (
            turnover.detach()
            .cpu()
            .numpy()
        )

        trade_count_cpu = trade_count.detach().cpu().numpy()
        transaction_costs_cpu = transaction_costs.detach().cpu().numpy()
        invalid_output_cpu = invalid_output_seen.detach().cpu().numpy()

        equity_cpu = (
            equity_matrix.detach()
            .cpu()
            .numpy()
        )

        exposure_cpu = (
            torch.stack(gross_exposure_history)
            .detach()
            .cpu()
            .numpy()
            if gross_exposure_history
            else np.empty((0, population_size), dtype=np.float32)
        )

        results = []

        for index, agent in enumerate(
            agents
        ):
            result = self._calculate_result(
                equity_curve=[float(value) for value in equity_cpu[:, index]],
                turnover=float(turnover_cpu[index]),
                trade_count=int(trade_count_cpu[index]),
                transaction_costs=float(transaction_costs_cpu[index]),
                gross_exposures=[
                    float(value) for value in exposure_cpu[:, index]
                ],
            )
            if invalid_output_cpu[index]:
                result.fitness = float("-inf")

            agent.fitness = result.fitness

            results.append(
                result
            )

        return results

    # ==========================================================
    # GPU NEURAL NETWORK
    # ==========================================================

    def _forward_population(
        self,
        observations: torch.Tensor,
        compiled: dict,
    ) -> torch.Tensor:
        """
        Batched feed-forward evaluation across the population.
        """

        population_size = (
            observations.shape[0]
        )

        max_nodes = compiled[
            "max_nodes"
        ]

        values = torch.zeros(
            (
                population_size,
                max_nodes,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        # ------------------------------------------------------
        # Inputs
        # ------------------------------------------------------

        input_indices = compiled[
            "input_indices"
        ]

        if input_indices.shape[1] > 0:
            safe_indices = torch.clamp(
                input_indices,
                min=0,
            )

            mask = (
                input_indices >= 0
            )

            input_count = min(
                observations.shape[1],
                input_indices.shape[1],
            )

            input_values = torch.zeros(
                (
                    population_size,
                    input_indices.shape[1],
                ),
                dtype=torch.float32,
                device=self.device,
            )

            input_values[
                :,
                :input_count,
            ] = observations[
                :,
                :input_count,
            ]

            input_values = torch.where(
                mask,
                input_values,
                torch.zeros_like(
                    input_values
                ),
            )

            values.scatter_(
                1,
                safe_indices,
                input_values,
            )

        # ------------------------------------------------------
        # Biases
        # ------------------------------------------------------

        bias_indices = compiled[
            "bias_indices"
        ]

        if bias_indices.shape[1] > 0:
            safe_bias_indices = torch.clamp(
                bias_indices,
                min=0,
            )

            bias_mask = (
                bias_indices >= 0
            )

            bias_values = torch.where(
                bias_mask,
                torch.ones_like(
                    bias_indices,
                    dtype=torch.float32,
                ),
                torch.zeros_like(
                    bias_indices,
                    dtype=torch.float32,
                ),
            )

            values.scatter_(
                1,
                safe_bias_indices,
                bias_values,
            )

        # ------------------------------------------------------
        # Topological layers
        # ------------------------------------------------------

        for matrices, nodes, activations, masks in zip(
            compiled["matrices"],
            compiled["nodes"],
            compiled["activations"],
            compiled["masks"],
        ):
            if matrices.shape[1] == 0:
                continue

            totals = torch.bmm(
                matrices,
                values.unsqueeze(-1),
            ).squeeze(-1)

            result = torch.zeros_like(
                totals
            )

            tanh_result = torch.tanh(
                totals
            )

            relu_result = torch.relu(
                totals
            )

            sigmoid_result = torch.sigmoid(
                torch.clamp(
                    totals,
                    -60.0,
                    60.0,
                )
            )

            linear_result = totals

            result = torch.where(
                activations == 1,
                tanh_result,
                result,
            )

            result = torch.where(
                activations == 2,
                relu_result,
                result,
            )

            result = torch.where(
                activations == 3,
                sigmoid_result,
                result,
            )

            result = torch.where(
                activations == 4,
                linear_result,
                result,
            )

            result = torch.where(
                masks,
                result,
                torch.zeros_like(
                    result
                ),
            )

            safe_nodes = torch.clamp(
                nodes,
                min=0,
            )

            values.scatter_(
                1,
                safe_nodes,
                result,
            )

        # ------------------------------------------------------
        # Output
        # ------------------------------------------------------

        output_indices = compiled[
            "output_indices"
        ]

        if output_indices.shape[1] == 0:
            return torch.zeros(
                population_size,
                dtype=torch.float32,
                device=self.device,
            )

        safe_output_indices = torch.clamp(
            output_indices,
            min=0,
        )

        output_values = values.gather(
            1,
            safe_output_indices,
        )

        output_mask = (
            output_indices >= 0
        )

        output_values = torch.where(
            output_mask,
            output_values,
            torch.zeros_like(
                output_values
            ),
        )

        actions = output_values[
            :,
            0,
        ]

        return actions

    # ==========================================================
    # GPU PORTFOLIO EXECUTION
    # ==========================================================

    def _execute_batch(
        self,
        cash: torch.Tensor,
        position: torch.Tensor,
        average_entry: torch.Tensor,
        turnover: torch.Tensor,
        trade_count: torch.Tensor,
        transaction_costs: torch.Tensor,
        target_position: torch.Tensor,
        execution_price: torch.Tensor,
    ) -> torch.Tensor:
        quantity = (
            target_position
            - position
        )

        unchanged = torch.abs(quantity) <= (
            1e-12 + 1e-12 * torch.abs(position)
        )
        quantity = torch.where(unchanged, torch.zeros_like(quantity), quantity)

        if not self.allow_short:
            quantity = torch.maximum(
                quantity,
                -position,
            )

        equity = (
            cash
            + position
            * execution_price
        )

        max_notional = torch.maximum(
            equity
            * self.max_leverage,
            torch.zeros_like(
                equity
            ),
        )

        safe_price = torch.clamp(
            execution_price,
            min=1e-6,
        )

        allowed_position = (
            max_notional
            / safe_price
        )

        target_after = (
            position
            + quantity
        )

        if self.allow_short:
            target_after = torch.maximum(
                target_after,
                -allowed_position,
            )

            target_after = torch.minimum(
                target_after,
                allowed_position,
            )

        else:
            target_after = torch.maximum(
                target_after,
                torch.zeros_like(
                    target_after
                ),
            )

            target_after = torch.minimum(
                target_after,
                allowed_position,
            )

        quantity = (
            target_after
            - position
        )

        adverse_rate = (
            self.costs.spread_rate / 2.0
            + self.costs.slippage_rate
        )

        execution_price_adjusted = (
            execution_price
            * (
                1.0
                + torch.sign(quantity)
                * adverse_rate
            )
        )

        cost_rate = self.costs.commission_rate

        notional = (
            quantity
            * execution_price_adjusted
        )

        transaction_cost = (
            torch.abs(notional)
            * cost_rate
        )

        buy_mask = (
            quantity > 0.0
        )

        affordable_quantity = (
            cash
            / torch.clamp(
                execution_price_adjusted
                * (1.0 + cost_rate),
                min=1e-6,
            )
        )

        affordable_quantity = torch.maximum(
            affordable_quantity,
            torch.zeros_like(
                affordable_quantity
            ),
        )

        insufficient_cash = (
            buy_mask
            & (
                notional
                + transaction_cost
                > cash
            )
        )

        quantity = torch.where(
            insufficient_cash,
            torch.minimum(
                quantity,
                affordable_quantity,
            ),
            quantity,
        )

        execution_price_adjusted = (
            execution_price
            * (
                1.0
                + torch.sign(quantity)
                * adverse_rate
            )
        )

        notional = (
            quantity
            * execution_price_adjusted
        )

        transaction_cost = (
            torch.abs(notional)
            * cost_rate
        )

        equity_before = (
            cash
            + position
            * execution_price
        )

        cash -= (
            notional
            + transaction_cost
        )

        old_position = position.clone()

        old_abs = torch.abs(
            old_position
        )

        quantity_abs = torch.abs(
            quantity
        )

        new_position = (
            old_position
            + quantity
        )

        adding = (
            (old_position == 0.0)
            | (
                (old_position > 0.0)
                & (quantity > 0.0)
            )
            | (
                (old_position < 0.0)
                & (quantity < 0.0)
            )
        )

        weighted_average = (
            (
                old_abs
                * average_entry
            )
            + (
                quantity_abs
                * execution_price_adjusted
            )
        ) / torch.clamp(
            old_abs + quantity_abs,
            min=1e-8,
        )

        updated_average = torch.where(
            adding,
            weighted_average,
            average_entry,
        )

        crossed = (
            old_position
            * new_position
            < 0.0
        )

        updated_average = torch.where(
            crossed,
            execution_price_adjusted,
            updated_average,
        )

        updated_average = torch.where(
            new_position == 0.0,
            torch.zeros_like(
                updated_average
            ),
            updated_average,
        )

        turnover += (
            torch.abs(notional)
            / torch.clamp(
                equity_before,
                min=1.0,
            )
        )

        trade_count += (quantity != 0.0).to(torch.long)
        transaction_costs += torch.abs(notional) * (
            self.costs.commission_rate
            + self.costs.spread_rate / 2.0
            + self.costs.slippage_rate
        )

        position.copy_(
            new_position
        )

        return updated_average

    # ==========================================================
    # FITNESS
    # ==========================================================

    def _calculate_result(
        self,
        equity_curve: list[float],
        turnover: float,
        trade_count: int = 0,
        transaction_costs: float = 0.0,
        gross_exposures: list[float] | None = None,
    ) -> EvaluationResult:
        equity: np.ndarray = np.asarray(
            equity_curve,
            dtype=np.float64,
        )

        if len(equity) < 2:
            return EvaluationResult(
                fitness=float("-inf"),
                final_equity=float(
                    equity[-1]
                ),
                total_return=0.0,
                max_drawdown=0.0,
                turnover=float(
                    turnover
                ),
                equity_curve=[
                    float(value)
                    for value in equity
                ],
                trade_count=trade_count,
                transaction_costs=transaction_costs,
                average_gross_exposure=0.0,
                max_gross_exposure=0.0,
            )

        initial = equity[0]

        if initial <= 0.0:
            return EvaluationResult(
                fitness=float("-inf"),
                final_equity=float(
                    equity[-1]
                ),
                total_return=0.0,
                max_drawdown=0.0,
                turnover=float(
                    turnover
                ),
                equity_curve=[
                    float(value)
                    for value in equity
                ],
                trade_count=trade_count,
                transaction_costs=transaction_costs,
                average_gross_exposure=0.0,
                max_gross_exposure=0.0,
            )

        total_return = (
            equity[-1]
            / initial
            - 1.0
        )

        running_max = np.maximum.accumulate(
            equity
        )

        drawdowns = (
            equity
            / running_max
            - 1.0
        )

        max_drawdown = abs(
            float(
                np.min(
                    drawdowns
                )
            )
        )

        fitness = self.fitness_calculator.calculate(equity_curve, turnover)
        exposures: np.ndarray = np.asarray(gross_exposures or [], dtype=np.float64)
        finite_exposures = exposures[np.isfinite(exposures)]

        return EvaluationResult(
            fitness=float(
                fitness
            ),
            final_equity=float(
                equity[-1]
            ),
            total_return=float(
                total_return
            ),
            max_drawdown=float(
                max_drawdown
            ),
            turnover=float(
                turnover
            ),
            equity_curve=[
                float(value)
                for value in equity
            ],
            trade_count=int(trade_count),
            transaction_costs=float(transaction_costs),
            average_gross_exposure=(
                float(np.mean(finite_exposures)) if finite_exposures.size else 0.0
            ),
            max_gross_exposure=(
                float(np.max(finite_exposures)) if finite_exposures.size else 0.0
            ),
        )
