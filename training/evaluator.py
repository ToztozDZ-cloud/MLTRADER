from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from agents.agent import Agent
from environment.costs import TradingCosts
from environment.exchange import Exchange
from environment.execution import ExecutionEngine
from environment.market import MarketEnvironment
from environment.observation import ObservationBuilder
from environment.portfolio import Portfolio

from .fitness import FitnessCalculator


@dataclass
class EvaluationResult:
    fitness: float
    final_equity: float
    total_return: float
    max_drawdown: float
    turnover: float
    equity_curve: list[float]


@dataclass
class PopulationEvaluationResult:
    results: list[EvaluationResult]


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
        fitness_calculator: FitnessCalculator | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.lookback = int(lookback)
        self.initial_cash = float(initial_cash)

        self.costs = costs or TradingCosts()

        self.allow_short = bool(
            allow_short
        )

        self.max_position_fraction = float(
            max_position_fraction
        )

        self.max_leverage = float(
            max_leverage
        )

        self.execute_on_next_bar = bool(
            execute_on_next_bar
        )

        self.fitness_calculator = (
            fitness_calculator
            or FitnessCalculator()
        )

        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

    @property
    def using_cuda(self) -> bool:
        return self.device.type == "cuda"

    # ==========================================================
    # CPU REFERENCE EVALUATION
    # ==========================================================

    def evaluate(
        self,
        agent: Agent,
        market_data: pd.DataFrame,
        start_step: int | None = None,
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

        while not market.done:
            observation = observation_builder.build(
                market,
                portfolio,
            )

            action = agent.act(
                observation
            )

            target_fraction = float(
                np.clip(
                    action,
                    -1.0,
                    1.0,
                )
            )

            if not self.allow_short:
                target_fraction = max(
                    0.0,
                    target_fraction,
                )

            target_fraction *= (
                self.max_position_fraction
            )

            current_price = market.current_price

            current_equity = portfolio.equity(
                current_price
            )

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

            execution_price = market.current_price

            if self.execute_on_next_bar:
                result = (
                    execution.execute_target_position(
                        target_position=target_position,
                        market_price=execution_price,
                    )
                )

                turnover += (
                    abs(result.notional)
                    / max(
                        portfolio.equity(
                            execution_price
                        ),
                        1.0,
                    )
                )

            equity_curve.append(
                portfolio.equity(
                    market.current_price
                )
            )

        if portfolio.position != 0.0:
            final_price = market.current_price

            liquidation = (
                execution.execute_target_position(
                    target_position=0.0,
                    market_price=final_price,
                )
            )

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

        result = self._calculate_result(
            equity_curve=equity_curve,
            turnover=turnover,
        )

        agent.fitness = result.fitness

        return result

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
            f"GPU population evaluation | "
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

        prices = self._extract_prices(
            market_data
        )

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

        if self.using_cuda:
            torch.cuda.synchronize(
                self.device
            )

        compiled = self._compile_population(
            agents
        )

        results = self._simulate_population(
            agents=agents,
            observations=observation_tensor,
            prices=price_tensor,
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

            close = window[:, 3]

            safe_close = np.where(
                close > 0.0,
                close,
                1.0,
            )

            normalized = window.copy()

            normalized[:, :4] = (
                normalized[:, :4]
                / safe_close[:, None]
                - 1.0
            )

            volume_scale = max(
                float(
                    np.mean(window[:, 4])
                ),
                1.0,
            )

            normalized[:, 4] = (
                window[:, 4]
                / volume_scale
                - 1.0
            )

            normalized = np.clip(
                normalized,
                -10.0,
                10.0,
            )

            progress = (
                step
                / max(
                    total_rows - 1,
                    1,
                )
            )

            observation = np.concatenate(
                [
                    normalized.reshape(-1),
                    np.array(
                        [
                            0.0,
                            1.0,
                            0.0,
                            progress,
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

        compiled_agents = []

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

            incoming = {
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

            depths = {}

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

            depth_groups = {}

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

            layers = []

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

                matrix = np.zeros(
                    (
                        len(target_nodes),
                        len(node_ids),
                    ),
                    dtype=np.float32,
                )

                activations = np.zeros(
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

        layer_matrices = []
        layer_nodes = []
        layer_activations = []
        layer_masks = []

        for depth_index in range(
            max_depth
        ):
            target_count = 0

            for description in compiled_agents:
                layers = description[
                    "layers"
                ]

                if depth_index >= len(layers):
                    continue

                target_count = max(
                    target_count,
                    len(
                        layers[
                            depth_index
                        ][
                            "nodes"
                        ]
                    ),
                )

            matrices = torch.zeros(
                (
                    population_size,
                    target_count,
                    max_nodes,
                ),
                dtype=torch.float32,
                device=self.device,
            )

            nodes = torch.zeros(
                (
                    population_size,
                    target_count,
                ),
                dtype=torch.long,
                device=self.device,
            )

            activations = torch.zeros(
                (
                    population_size,
                    target_count,
                ),
                dtype=torch.long,
                device=self.device,
            )

            masks = torch.zeros(
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
                layers = description[
                    "layers"
                ]

                if depth_index >= len(layers):
                    continue

                layer = layers[
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

                matrices[
                    agent_index,
                    :local_target_count,
                    :local_width,
                ] = torch.from_numpy(
                    local_matrix
                ).to(
                    self.device
                )

                nodes[
                    agent_index,
                    :local_target_count,
                ] = torch.from_numpy(
                    local_nodes
                ).to(
                    self.device
                )

                activations[
                    agent_index,
                    :local_target_count,
                ] = torch.from_numpy(
                    local_activations
                ).to(
                    self.device
                )

                masks[
                    agent_index,
                    :local_target_count,
                ] = True

            layer_matrices.append(
                matrices
            )

            layer_nodes.append(
                nodes
            )

            layer_activations.append(
                activations
            )

            layer_masks.append(
                masks
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

        adjacency = {
            node_id: []
            for node_id in nodes
        }

        indegree = {
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

        initial_equity = cash.clone()

        peak_equity = initial_equity.clone()

        max_drawdown = torch.zeros(
            population_size,
            dtype=torch.float32,
            device=self.device,
        )

        equity_history = [
            initial_equity.clone()
        ]

        steps = observations.shape[0]

        progress_interval = max(
            1,
            steps // 100,
        )

        simulation_start = time.perf_counter()

        if self.using_cuda:
            torch.cuda.synchronize(
                self.device
            )

        print(
            f"GPU evaluation | "
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

            safe_equity = torch.clamp(
                equity,
                min=1e-6,
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

            progress = (
                float(price_index)
                / max(
                    len(prices) - 1,
                    1,
                )
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

            batch_observation[:, -4] = (
                position_fraction
            )

            batch_observation[:, -3] = (
                cash_fraction
            )

            batch_observation[:, -2] = (
                unrealized_fraction
            )

            batch_observation[:, -1] = (
                progress
            )

            actions = self._forward_population(
                batch_observation,
                compiled,
            )

            target_fraction = torch.clamp(
                actions,
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

            target_position = (
                equity
                * target_fraction
                / torch.clamp(
                    current_price,
                    min=1e-6,
                )
            )

            if (
                self.execute_on_next_bar
                and relative_step + 1 < steps
            ):
                execution_price = prices[
                    price_index + 1
                ]

                average_entry = (
                    self._execute_batch(
                        cash=cash,
                        position=position,
                        average_entry=average_entry,
                        turnover=turnover,
                        target_position=target_position,
                        execution_price=execution_price,
                    )
                )

                marked_equity = (
                    cash
                    + position
                    * execution_price
                )

                peak_equity = torch.maximum(
                    peak_equity,
                    marked_equity,
                )

                drawdown = (
                    marked_equity
                    / torch.clamp(
                        peak_equity,
                        min=1e-6,
                    )
                    - 1.0
                )

                max_drawdown = torch.maximum(
                    max_drawdown,
                    -drawdown,
                )

                equity_history.append(
                    marked_equity.clone()
                )

            completed = (
                relative_step + 1
            )

            if (
                completed == steps
                or completed % progress_interval == 0
            ):
                if self.using_cuda:
                    torch.cuda.synchronize(
                        self.device
                    )

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
                    prefix="GPU evaluation",
                    completed=completed,
                    total=steps,
                    elapsed=elapsed,
                    rate=rate,
                    eta=eta,
                )

        if self.using_cuda:
            torch.cuda.synchronize(
                self.device
            )

        final_price = prices[-1]

        average_entry = (
            self._execute_batch(
                cash=cash,
                position=position,
                average_entry=average_entry,
                turnover=turnover,
                target_position=torch.zeros_like(
                    position
                ),
                execution_price=final_price,
            )
        )

        if self.using_cuda:
            torch.cuda.synchronize(
                self.device
            )

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

        returns = (
            equity_matrix[1:]
            / torch.clamp(
                equity_matrix[:-1],
                min=1e-6,
            )
            - 1.0
        )

        if returns.shape[0] > 0:
            volatility = torch.std(
                returns,
                dim=0,
                unbiased=False,
            )

            mean_return = torch.mean(
                returns,
                dim=0,
            )
        else:
            volatility = torch.zeros(
                population_size,
                dtype=torch.float32,
                device=self.device,
            )

            mean_return = torch.zeros(
                population_size,
                dtype=torch.float32,
                device=self.device,
            )

        sharpe = torch.where(
            volatility > 0.0,
            mean_return
            / volatility
            * np.sqrt(252.0),
            torch.zeros_like(
                volatility
            ),
        )

        total_return = (
            final_equity
            / initial_equity
            - 1.0
        )

        fitness = (
            self.fitness_calculator.return_weight
            * total_return
            + self.fitness_calculator.sharpe_weight
            * sharpe
            - self.fitness_calculator.drawdown_penalty
            * max_drawdown
            - self.fitness_calculator.volatility_penalty
            * volatility
            - self.fitness_calculator.turnover_penalty
            * turnover
        )

        if self.using_cuda:
            torch.cuda.synchronize(
                self.device
            )

        total_elapsed = (
            time.perf_counter()
            - simulation_start
        )

        print(
            f"GPU evaluation complete | "
            f"{steps:,} bars | "
            f"{total_elapsed:.2f}s | "
            f"{steps / total_elapsed:.2f} bar/s"
        )

        fitness_cpu = (
            fitness.detach()
            .cpu()
            .numpy()
        )

        final_equity_cpu = (
            final_equity.detach()
            .cpu()
            .numpy()
        )

        total_return_cpu = (
            total_return.detach()
            .cpu()
            .numpy()
        )

        max_drawdown_cpu = (
            max_drawdown.detach()
            .cpu()
            .numpy()
        )

        turnover_cpu = (
            turnover.detach()
            .cpu()
            .numpy()
        )

        equity_cpu = (
            equity_matrix.detach()
            .cpu()
            .numpy()
        )

        results = []

        for index, agent in enumerate(
            agents
        ):
            result = EvaluationResult(
                fitness=float(
                    fitness_cpu[index]
                ),
                final_equity=float(
                    final_equity_cpu[index]
                ),
                total_return=float(
                    total_return_cpu[index]
                ),
                max_drawdown=float(
                    max_drawdown_cpu[index]
                ),
                turnover=float(
                    turnover_cpu[index]
                ),
                equity_curve=[
                    float(value)
                    for value in equity_cpu[
                        :,
                        index,
                    ]
                ],
            )

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

        return torch.clamp(
            actions,
            -1.0,
            1.0,
        )

    # ==========================================================
    # GPU PORTFOLIO EXECUTION
    # ==========================================================

    def _execute_batch(
        self,
        cash: torch.Tensor,
        position: torch.Tensor,
        average_entry: torch.Tensor,
        turnover: torch.Tensor,
        target_position: torch.Tensor,
        execution_price: torch.Tensor,
    ) -> torch.Tensor:
        quantity = (
            target_position
            - position
        )

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
            self.costs.spread_rate
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

        cost_rate = (
            self.costs.commission_rate
            + self.costs.spread_rate
            + self.costs.slippage_rate
        )

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
            torch.abs(new_position) < 1e-8,
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
    ) -> EvaluationResult:
        equity = np.asarray(
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

        returns = (
            np.diff(equity)
            / equity[:-1]
        )

        volatility = float(
            np.std(
                returns
            )
        )

        if volatility > 0.0:
            sharpe = (
                float(
                    np.mean(
                        returns
                    )
                    / volatility
                )
                * np.sqrt(252.0)
            )
        else:
            sharpe = 0.0

        fitness = (
            self.fitness_calculator.return_weight
            * total_return
            + self.fitness_calculator.sharpe_weight
            * sharpe
            - self.fitness_calculator.drawdown_penalty
            * max_drawdown
            - self.fitness_calculator.volatility_penalty
            * volatility
            - self.fitness_calculator.turnover_penalty
            * turnover
        )

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
        )