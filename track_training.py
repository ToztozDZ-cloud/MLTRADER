from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

# Force this tracker process to use UTF-8 too.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(
        encoding="utf-8",
        errors="replace",
    )

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MaxNLocator
import tkinter as tk


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

EVOLVE_SCRIPT = PROJECT_ROOT / "evolve.py"

DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "SPY.csv"
DEFAULT_TRAINING_CONFIG = PROJECT_ROOT / "config" / "training.yaml"
DEFAULT_MARKET_CONFIG = PROJECT_ROOT / "config" / "market.yaml"

ARCHIVE_DIR = PROJECT_ROOT / "models" / "archives"
RUNS_DIR = ARCHIVE_DIR / "training_runs"

CHAMPION_DIR = PROJECT_ROOT / "models" / "champions"
CHAMPION_PATH = CHAMPION_DIR / "best_agent.json"


# ============================================================================
# REGEX
# ============================================================================

FLOAT_PATTERN = (
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?"
)

GENERATION_RE = re.compile(
    rf"Generation\s+(\d+)"
    rf"(?:\s*\|\s*Best fitness:\s*({FLOAT_PATTERN}))?"
)

BEST_FITNESS_RE = re.compile(
    rf"Best fitness:\s*({FLOAT_PATTERN})"
)

FINAL_EQUITY_RE = re.compile(
    r"Final equity:\s*\$([\d,]+(?:\.\d+)?)"
)

RETURN_RE = re.compile(
    rf"Return:\s*({FLOAT_PATTERN})\s*%"
)

DRAWDOWN_RE = re.compile(
    rf"Max drawdown:\s*({FLOAT_PATTERN})\s*%"
)

TURNOVER_RE = re.compile(
    rf"Turnover:\s*({FLOAT_PATTERN})"
)

GENERATION_TIME_RE = re.compile(
    rf"Generation time:\s*({FLOAT_PATTERN})s"
)


# ============================================================================
# ARGUMENTS
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MLTRADER evolution training while displaying "
            "live return, fitness, and final-equity graphs."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Historical market CSV.",
    )

    parser.add_argument(
        "--training-config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG,
        help="Training YAML configuration.",
    )

    parser.add_argument(
        "--market-config",
        type=Path,
        default=DEFAULT_MARKET_CONFIG,
        help="Market YAML configuration.",
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Override the number of generations.",
    )

    parser.add_argument(
        "--reset-champion",
        action="store_true",
        help="Delete the persistent champion before training.",
    )

    return parser.parse_args()


# ============================================================================
# HELPERS
# ============================================================================

def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    return (PROJECT_ROOT / path).resolve()


def create_run_directory() -> Path:
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = RUNS_DIR / timestamp

    counter = 1

    while run_dir.exists():
        run_dir = (
            RUNS_DIR /
            f"{timestamp}_{counter:02d}"
        )

        counter += 1

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    return run_dir


def clean_champion() -> None:
    if CHAMPION_PATH.exists():
        CHAMPION_PATH.unlink()

        print(
            "Previous champion removed:"
        )

        print(
            f"  {CHAMPION_PATH}"
        )


def build_evolve_command(
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(EVOLVE_SCRIPT),
        "--data",
        str(resolve_path(args.data)),
        "--training-config",
        str(resolve_path(args.training_config)),
        "--market-config",
        str(resolve_path(args.market_config)),
    ]

    if args.generations is not None:
        command.extend(
            [
                "--generations",
                str(args.generations),
            ]
        )

    if args.reset_champion:
        command.append(
            "--reset-champion"
        )

    return command


# ============================================================================
# CSV
# ============================================================================

CSV_FIELDS = [
    "generation",
    "internal_generation",
    "best_fitness",
    "final_equity",
    "return_percent",
    "max_drawdown_percent",
    "turnover",
    "generation_time_seconds",
]


def initialize_history_csv(
    path: Path,
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()


def append_history(
    path: Path,
    metrics: dict,
) -> None:
    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        writer.writerow(
            {
                "generation":
                    metrics["generation"],
                "internal_generation":
                    metrics["internal_generation"],
                "best_fitness":
                    metrics["best_fitness"],
                "final_equity":
                    metrics["final_equity"],
                "return_percent":
                    metrics["return_percent"],
                "max_drawdown_percent":
                    metrics["max_drawdown_percent"],
                "turnover":
                    metrics["turnover"],
                "generation_time_seconds":
                    metrics["generation_time_seconds"],
            }
        )


# ============================================================================
# SNAPSHOTS
# ============================================================================

def save_champion_snapshot(
    run_dir: Path,
    generation: int,
) -> Path | None:
    if not CHAMPION_PATH.exists():
        return None

    snapshot_path = (
        run_dir /
        f"generation_{generation:04d}.json"
    )

    shutil.copy2(
        CHAMPION_PATH,
        snapshot_path,
    )

    return snapshot_path


# ============================================================================
# TRAINING PARSER
# ============================================================================

class TrainingRunner:
    def __init__(
        self,
        command: list[str],
        run_dir: Path,
        history_csv: Path,
        events: Queue,
    ) -> None:
        self.command = command
        self.run_dir = run_dir
        self.history_csv = history_csv
        self.events = events

        self.process: (
            subprocess.Popen[str] | None
        ) = None

        self.current_generation: (
            int | None
        ) = None

        self.metrics = {
            "best_fitness": None,
            "final_equity": None,
            "return_percent": None,
            "max_drawdown_percent": None,
            "turnover": None,
            "generation_time_seconds": None,
        }

    def reset_generation_metrics(
        self,
    ) -> None:
        self.metrics = {
            "best_fitness": None,
            "final_equity": None,
            "return_percent": None,
            "max_drawdown_percent": None,
            "turnover": None,
            "generation_time_seconds": None,
        }

    def start(self) -> None:
        thread = threading.Thread(
            target=self.run,
            name="MLTRADER-Training",
            daemon=True,
        )

        thread.start()

    def run(self) -> None:
        try:
            # ------------------------------------------------------------
            # Force the child evolve.py process to use UTF-8.
            # This fixes Windows cp1252 UnicodeEncodeError crashes.
            # ------------------------------------------------------------

            env = os.environ.copy()

            env["PYTHONIOENCODING"] = (
                "utf-8"
            )

            env["PYTHONUTF8"] = "1"

            self.process = subprocess.Popen(
                self.command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                universal_newlines=True,
                env=env,
            )

            assert self.process.stdout is not None

            for raw_line in self.process.stdout:
                line = raw_line.rstrip()

                print(
                    line,
                    flush=True,
                )

                self.process_line(line)

            return_code = self.process.wait()

            self.events.put(
                (
                    "done",
                    {
                        "return_code":
                            return_code,
                    },
                )
            )

        except Exception as exc:
            self.events.put(
                (
                    "error",
                    {
                        "message":
                            str(exc),
                    },
                )
            )

    def process_line(
        self,
        line: str,
    ) -> None:
        # ------------------------------------------------------------
        # Generation
        # ------------------------------------------------------------

        generation_match = (
            GENERATION_RE.search(line)
        )

        if generation_match:
            generation = int(
                generation_match.group(1)
            )

            if (
                self.current_generation
                is not None
                and generation
                != self.current_generation
            ):
                self.reset_generation_metrics()

            self.current_generation = (
                generation
            )

            fitness_text = (
                generation_match.group(2)
            )

            if fitness_text is not None:
                self.metrics[
                    "best_fitness"
                ] = float(
                    fitness_text
                )

        # ------------------------------------------------------------
        # Best fitness
        # ------------------------------------------------------------

        fitness_match = (
            BEST_FITNESS_RE.search(line)
        )

        if fitness_match:
            self.metrics[
                "best_fitness"
            ] = float(
                fitness_match.group(1)
            )

        # ------------------------------------------------------------
        # Final equity
        # ------------------------------------------------------------

        equity_match = (
            FINAL_EQUITY_RE.search(line)
        )

        if equity_match:
            self.metrics[
                "final_equity"
            ] = float(
                equity_match.group(1)
                .replace(",", "")
            )

        # ------------------------------------------------------------
        # Return
        # ------------------------------------------------------------

        return_match = (
            RETURN_RE.search(line)
        )

        if return_match:
            self.metrics[
                "return_percent"
            ] = float(
                return_match.group(1)
            )

        # ------------------------------------------------------------
        # Drawdown
        # ------------------------------------------------------------

        drawdown_match = (
            DRAWDOWN_RE.search(line)
        )

        if drawdown_match:
            self.metrics[
                "max_drawdown_percent"
            ] = float(
                drawdown_match.group(1)
            )

        # ------------------------------------------------------------
        # Turnover
        # ------------------------------------------------------------

        turnover_match = (
            TURNOVER_RE.search(line)
        )

        if turnover_match:
            self.metrics[
                "turnover"
            ] = float(
                turnover_match.group(1)
            )

        # ------------------------------------------------------------
        # Generation time
        # ------------------------------------------------------------

        generation_time_match = (
            GENERATION_TIME_RE.search(line)
        )

        if generation_time_match:
            self.metrics[
                "generation_time_seconds"
            ] = float(
                generation_time_match.group(1)
            )

            self.finish_generation()

    def finish_generation(self) -> None:
        if self.current_generation is None:
            return

        return_percent = (
            self.metrics["return_percent"]
        )

        if return_percent is None:
            return

        internal_generation = (
            self.current_generation
        )

        # evolve.py starts at generation 0.
        # Display the graph as Generation 1, 2, 3...
        display_generation = (
            internal_generation + 1
        )

        metrics = {
            "generation":
                display_generation,
            "internal_generation":
                internal_generation,
            "best_fitness":
                self.metrics["best_fitness"],
            "final_equity":
                self.metrics["final_equity"],
            "return_percent":
                return_percent,
            "max_drawdown_percent":
                self.metrics[
                    "max_drawdown_percent"
                ],
            "turnover":
                self.metrics["turnover"],
            "generation_time_seconds":
                self.metrics[
                    "generation_time_seconds"
                ],
        }

        append_history(
            self.history_csv,
            metrics,
        )

        snapshot_path = (
            save_champion_snapshot(
                self.run_dir,
                display_generation,
            )
        )

        self.events.put(
            (
                "generation",
                {
                    "metrics":
                        metrics,
                    "snapshot":
                        snapshot_path,
                },
            )
        )


# ============================================================================
# GRAPH WINDOW
# ============================================================================

class TrainingGraph:
    """Live return window plus a second window for fitness and equity."""

    def __init__(
        self,
        root: tk.Tk,
        runner: TrainingRunner,
        events: Queue,
        run_dir: Path,
    ) -> None:
        self.root = root
        self.runner = runner
        self.events = events
        self.run_dir = run_dir

        self.generations: list[int] = []
        self.returns: list[float] = []
        self.fitnesses: list[float] = []
        self.equities: list[float] = []

        self.closed = False
        self.training_finished = False

        # ================================================================
        # WINDOW 1 — RETURN
        # ================================================================

        self.root.title("MLTRADER — Return Curve")
        self.root.geometry("1100x700")
        self.root.minsize(850, 550)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        header = tk.Frame(self.root, padx=12, pady=8)
        header.pack(side=tk.TOP, fill=tk.X)

        self.status_label = tk.Label(
            header,
            text="Starting MLTRADER training...",
            anchor="w",
            font=("Segoe UI", 11, "bold"),
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.latest_label = tk.Label(
            header,
            text="No generation completed yet",
            anchor="e",
            font=("Segoe UI", 10),
        )
        self.latest_label.pack(side=tk.RIGHT)

        self.figure_return, self.axis_return = plt.subplots(
            figsize=(11, 7),
            dpi=100,
        )

        self.axis_return.set_title(
            "MLTRADER Evolution — Return by Generation"
        )
        self.axis_return.set_xlabel("Generation")
        self.axis_return.set_ylabel("Return (%)")
        self.axis_return.grid(True, alpha=0.25)
        self.axis_return.axhline(
            y=0.0,
            linewidth=1.0,
            linestyle="--",
            alpha=0.5,
        )
        self.axis_return.xaxis.set_major_locator(
            MaxNLocator(integer=True, nbins=15)
        )

        self.return_line, = self.axis_return.plot(
            [], [],
            marker="o",
            linewidth=1.5,
            markersize=5,
        )

        self.canvas_return = FigureCanvasTkAgg(
            self.figure_return,
            master=self.root,
        )
        self.canvas_return.get_tk_widget().pack(
            side=tk.TOP,
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=(0, 10),
        )
        self.canvas_return.draw()

        # ================================================================
        # WINDOW 2 — FITNESS + FINAL EQUITY
        # ================================================================

        self.metrics_window = tk.Toplevel(self.root)
        self.metrics_window.title(
            "MLTRADER — Fitness & Final Equity"
        )
        self.metrics_window.geometry("1100x850")
        self.metrics_window.minsize(850, 700)
        self.metrics_window.protocol(
            "WM_DELETE_WINDOW",
            self.on_close,
        )

        metrics_header = tk.Frame(
            self.metrics_window,
            padx=12,
            pady=8,
        )
        metrics_header.pack(
            side=tk.TOP,
            fill=tk.X,
        )

        self.metrics_label = tk.Label(
            metrics_header,
            text="No generation completed yet",
            anchor="w",
            font=("Segoe UI", 10, "bold"),
        )
        self.metrics_label.pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
        )

        self.figure_fitness, self.axis_fitness = plt.subplots(
            figsize=(11, 4),
            dpi=100,
        )

        self.axis_fitness.set_title(
            "MLTRADER Evolution — Best Fitness"
        )
        self.axis_fitness.set_xlabel("Generation")
        self.axis_fitness.set_ylabel("Fitness")
        self.axis_fitness.grid(True, alpha=0.25)
        self.axis_fitness.xaxis.set_major_locator(
            MaxNLocator(integer=True, nbins=15)
        )

        self.fitness_line, = self.axis_fitness.plot(
            [], [],
            marker="o",
            linewidth=1.5,
            markersize=4,
        )

        self.canvas_fitness = FigureCanvasTkAgg(
            self.figure_fitness,
            master=self.metrics_window,
        )
        self.canvas_fitness.get_tk_widget().pack(
            side=tk.TOP,
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=(5, 5),
        )
        self.canvas_fitness.draw()

        self.figure_equity, self.axis_equity = plt.subplots(
            figsize=(11, 4),
            dpi=100,
        )

        self.axis_equity.set_title(
            "MLTRADER Evolution — Final Equity"
        )
        self.axis_equity.set_xlabel("Generation")
        self.axis_equity.set_ylabel("Final Equity (USD)")
        self.axis_equity.grid(True, alpha=0.25)
        self.axis_equity.xaxis.set_major_locator(
            MaxNLocator(integer=True, nbins=15)
        )

        self.equity_line, = self.axis_equity.plot(
            [], [],
            marker="o",
            linewidth=1.5,
            markersize=4,
        )

        self.canvas_equity = FigureCanvasTkAgg(
            self.figure_equity,
            master=self.metrics_window,
        )
        self.canvas_equity.get_tk_widget().pack(
            side=tk.TOP,
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=(5, 10),
        )
        self.canvas_equity.draw()

        self.root.after(200, self.process_events)

    def process_events(self) -> None:
        if self.closed:
            return

        processed = False

        while True:
            try:
                event_type, payload = self.events.get_nowait()
            except Empty:
                break

            processed = True

            if event_type == "generation":
                self.handle_generation(payload)
            elif event_type == "done":
                self.handle_done(payload)
            elif event_type == "error":
                self.handle_error(payload)

        if not self.training_finished:
            self.root.after(200, self.process_events)
        elif processed:
            self.root.after(500, self.process_events)

    def handle_generation(self, payload: dict) -> None:
        metrics = payload["metrics"]

        generation = int(metrics["generation"])
        return_percent = float(metrics["return_percent"])

        fitness = metrics["best_fitness"]
        final_equity = metrics["final_equity"]

        self.generations.append(generation)
        self.returns.append(return_percent)
        self.fitnesses.append(
            float(fitness) if fitness is not None else float("nan")
        )
        self.equities.append(
            float(final_equity)
            if final_equity is not None
            else float("nan")
        )

        # Return
        self.return_line.set_data(
            self.generations,
            self.returns,
        )
        self.axis_return.relim()
        self.axis_return.autoscale_view()

        # Fitness
        self.fitness_line.set_data(
            self.generations,
            self.fitnesses,
        )
        self.axis_fitness.relim()
        self.axis_fitness.autoscale_view()

        # Equity
        self.equity_line.set_data(
            self.generations,
            self.equities,
        )
        self.axis_equity.relim()
        self.axis_equity.autoscale_view()

        self.axis_return.xaxis.set_major_locator(
            MaxNLocator(integer=True, nbins=15)
        )
        self.axis_fitness.xaxis.set_major_locator(
            MaxNLocator(integer=True, nbins=15)
        )
        self.axis_equity.xaxis.set_major_locator(
            MaxNLocator(integer=True, nbins=15)
        )

        fitness_text = (
            "n/a"
            if fitness is None
            else f"{float(fitness):.6f}"
        )

        equity_text = (
            "n/a"
            if final_equity is None
            else f"${float(final_equity):,.2f}"
        )

        self.status_label.config(
            text=(
                "Training in progress — "
                f"Generation {generation}"
            )
        )

        self.latest_label.config(
            text=(
                f"Return: {return_percent:+.4f}%   |   "
                f"Fitness: {fitness_text}   |   "
                f"Equity: {equity_text}"
            )
        )

        self.metrics_label.config(
            text=(
                f"Generation {generation}   |   "
                f"Fitness: {fitness_text}   |   "
                f"Final equity: {equity_text}"
            )
        )

        self.canvas_return.draw_idle()
        self.canvas_fitness.draw_idle()
        self.canvas_equity.draw_idle()

        # Save all three live graphs.
        self.figure_return.savefig(
            self.run_dir / "return_curve.png",
            dpi=120,
            bbox_inches="tight",
        )
        self.figure_fitness.savefig(
            self.run_dir / "fitness_curve.png",
            dpi=120,
            bbox_inches="tight",
        )
        self.figure_equity.savefig(
            self.run_dir / "equity_curve.png",
            dpi=120,
            bbox_inches="tight",
        )

    def handle_done(self, payload: dict) -> None:
        self.training_finished = True
        return_code = payload["return_code"]

        if return_code == 0:
            self.status_label.config(
                text=(
                    "Training complete — "
                    f"{len(self.generations)} generations recorded"
                )
            )

            if self.returns:
                final_return = self.returns[-1]
                final_fitness = self.fitnesses[-1]
                final_equity = self.equities[-1]

                self.latest_label.config(
                    text=(
                        f"Final return: {final_return:+.4f}%   |   "
                        f"Fitness: {final_fitness:.6f}   |   "
                        f"Equity: ${final_equity:,.2f}"
                    )
                )

                self.metrics_label.config(
                    text=(
                        f"Final generation: {self.generations[-1]}   |   "
                        f"Fitness: {final_fitness:.6f}   |   "
                        f"Final equity: ${final_equity:,.2f}"
                    )
                )
        else:
            message = f"Training failed — exit code {return_code}"
            self.status_label.config(text=message)
            self.metrics_label.config(text=message)

        self.canvas_return.draw_idle()
        self.canvas_fitness.draw_idle()
        self.canvas_equity.draw_idle()

    def handle_error(self, payload: dict) -> None:
        self.training_finished = True
        message = f"Training error: {payload['message']}"

        self.status_label.config(text=message)
        self.metrics_label.config(text=message)

    def on_close(self) -> None:
        if self.closed:
            return

        self.closed = True

        process = self.runner.process

        if (
            process is not None
            and process.poll() is None
        ):
            try:
                process.terminate()
            except Exception:
                pass

        self.root.destroy()


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    args = parse_args()

    data_path = resolve_path(
        args.data
    )

    training_config = resolve_path(
        args.training_config
    )

    market_config = resolve_path(
        args.market_config
    )

    # ------------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------------

    if not EVOLVE_SCRIPT.exists():
        print(
            "ERROR: evolve.py not found:"
        )

        print(
            f"  {EVOLVE_SCRIPT}",
            file=sys.stderr,
        )

        return 1

    if not data_path.exists():
        print(
            "ERROR: market data not found:"
        )

        print(
            f"  {data_path}",
            file=sys.stderr,
        )

        return 1

    if not training_config.exists():
        print(
            "ERROR: training config not found:"
        )

        print(
            f"  {training_config}",
            file=sys.stderr,
        )

        return 1

    if not market_config.exists():
        print(
            "ERROR: market config not found:"
        )

        print(
            f"  {market_config}",
            file=sys.stderr,
        )

        return 1

    # ------------------------------------------------------------------------
    # Reset champion
    # ------------------------------------------------------------------------

    if args.reset_champion:
        clean_champion()

    # ------------------------------------------------------------------------
    # Create run directory
    # ------------------------------------------------------------------------

    run_dir = (
        create_run_directory()
    )

    history_csv = (
        run_dir /
        "generation_history.csv"
    )

    initialize_history_csv(
        history_csv
    )

    # ------------------------------------------------------------------------
    # Display configuration
    # ------------------------------------------------------------------------

    print("=" * 72)
    print(
        "MLTRADER LIVE TRAINING TRACKER"
    )
    print("=" * 72)

    print(
        f"Data:\n"
        f"  {data_path}"
    )

    print(
        f"Training config:\n"
        f"  {training_config}"
    )

    print(
        f"Market config:\n"
        f"  {market_config}"
    )

    if args.generations is not None:
        print(
            f"Generation override:\n"
            f"  {args.generations}"
        )

    print(
        f"Run directory:\n"
        f"  {run_dir}"
    )

    print(
        f"History:\n"
        f"  {history_csv}"
    )

    print(
        f"Persistent champion:\n"
        f"  {CHAMPION_PATH}"
    )

    print("=" * 72)
    print()

    # ------------------------------------------------------------------------
    # Build command
    # ------------------------------------------------------------------------

    command = (
        build_evolve_command(args)
    )

    print(
        "Starting training process..."
    )

    print(
        "Command:"
    )

    print(
        " ".join(
            (
                f'"{part}"'
                if " " in part
                else part
            )
            for part in command
        )
    )

    print()

    # ------------------------------------------------------------------------
    # Event queue
    # ------------------------------------------------------------------------

    events: Queue = Queue()

    runner = TrainingRunner(
        command=command,
        run_dir=run_dir,
        history_csv=history_csv,
        events=events,
    )

    # ------------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------------

    root = tk.Tk()

    TrainingGraph(
        root=root,
        runner=runner,
        events=events,
        run_dir=run_dir,
    )

    runner.start()

    root.mainloop()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )