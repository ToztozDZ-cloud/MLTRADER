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
from tkinter import ttk


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

EVOLVE_SCRIPT = PROJECT_ROOT / "evolve.py"

DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "SPY_5m.csv"
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
    rf"Generation time:\s*(?:(?P<hours>\d+)h\s*)?"
    rf"(?:(?P<minutes>\d+)m\s*)?(?P<seconds>{FLOAT_PATTERN})s"
)

GENERATION_COUNT_RE = re.compile(r"Generations:\s*(\d+)")

PROGRESS_RE = re.compile(
    rf"Training progress:\s*(?P<percentage>{FLOAT_PATTERN})%\s*\|\s*"
    r"(?P<completed>[\d,]+)/(?P<total>[\d,]+) bars\s*\|\s*"
    rf"(?P<rate>{FLOAT_PATTERN}) bars/s\s*\|\s*ETA\s*(?P<eta>.+)"
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
        "--device",
        choices=("auto", "cpu", "cuda", "xpu"),
        default="auto",
        help="Evaluation device; xpu selects Intel Arc GPUs.",
    )

    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Warm-start this run from a saved champion without overwriting it.",
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
    run_dir: Path,
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

    command.extend(["--device", args.device])
    command.extend(["--checkpoint-dir", str(run_dir / "checkpoints")])
    command.extend(["--champion-dir", str(run_dir / "champions")])

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
    run_champion_path = run_dir / "champions" / CHAMPION_PATH.name
    if not run_champion_path.exists():
        return None

    snapshot_path = (
        run_dir /
        f"generation_{generation:04d}.json"
    )

    shutil.copy2(
        run_champion_path,
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

        self.metrics: dict[str, float | None] = {
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
        if line:
            self.events.put(("log", {"line": line}))

        progress_match = PROGRESS_RE.search(line)
        if progress_match:
            self.events.put(
                (
                    "progress",
                    {
                        "percentage": float(progress_match.group("percentage")),
                        "completed": int(progress_match.group("completed").replace(",", "")),
                        "total": int(progress_match.group("total").replace(",", "")),
                        "rate": float(progress_match.group("rate")),
                        "eta": progress_match.group("eta").strip(),
                        "generation": self.current_generation,
                    },
                )
            )

        generation_count_match = GENERATION_COUNT_RE.search(line)
        if generation_count_match:
            self.events.put(
                ("system", {"total_generations": int(generation_count_match.group(1))})
            )

        if line.startswith(("GPU:", "GPU memory:", "Accelerated population evaluation:")):
            self.events.put(("system", {"line": line}))

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
            ] = (
                3600.0 * float(generation_time_match.group("hours") or 0)
                + 60.0 * float(generation_time_match.group("minutes") or 0)
                + float(generation_time_match.group("seconds"))
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
        self.drawdowns: list[float] = []
        self.turnovers: list[float] = []

        self.closed = False
        self.training_finished = False
        self.total_generations = 0
        self.generation_durations: list[float] = []
        self.palette = {
            "background": "#080d16",
            "panel": "#111b2b",
            "panel_alt": "#17243a",
            "border": "#26364d",
            "text": "#e7eefb",
            "muted": "#91a2ba",
            "blue": "#45b8ff",
            "cyan": "#67d9e8",
            "green": "#43d6a3",
            "red": "#f27878",
        }
        self.root.title("MLTRADER | Training Dashboard")
        self.root.geometry("1440x960")
        self.root.minsize(1050, 720)
        self.root.configure(bg=self.palette["background"])
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TNotebook", background=self.palette["background"], borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=self.palette["panel"],
            foreground=self.palette["muted"],
            padding=(18, 11),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.palette["panel_alt"])],
            foreground=[("selected", self.palette["text"])],
        )
        style.configure(
            "Training.Horizontal.TProgressbar",
            troughcolor=self.palette["panel_alt"],
            background=self.palette["blue"],
            bordercolor=self.palette["panel_alt"],
            lightcolor=self.palette["blue"],
            darkcolor=self.palette["blue"],
        )
        style.configure(
            "History.Treeview",
            background=self.palette["panel"],
            fieldbackground=self.palette["panel"],
            foreground=self.palette["text"],
            rowheight=30,
            borderwidth=0,
        )
        style.configure(
            "History.Treeview.Heading",
            background=self.palette["panel_alt"],
            foreground=self.palette["muted"],
            font=("Segoe UI", 9, "bold"),
        )

        header = tk.Frame(self.root, bg=self.palette["background"], padx=28, pady=20)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text="MLTRADER",
            bg=self.palette["background"],
            fg=self.palette["blue"],
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")
        self.status_label = tk.Label(
            header,
            text="Starting training...",
            bg=self.palette["background"],
            fg=self.palette["text"],
            font=("Segoe UI", 20, "bold"),
            anchor="w",
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=(4, 0))
        self.latest_label = tk.Label(
            header,
            text="No generation completed",
            bg=self.palette["background"],
            fg=self.palette["muted"],
            font=("Segoe UI", 10),
            anchor="e",
        )
        self.latest_label.pack(side=tk.RIGHT, pady=(14, 0))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        self.overview_tab = tk.Frame(self.notebook, bg=self.palette["background"], padx=14, pady=18)
        self.analytics_tab = tk.Frame(self.notebook, bg=self.palette["background"], padx=14, pady=18)
        self.history_tab = tk.Frame(self.notebook, bg=self.palette["background"], padx=14, pady=18)
        self.system_tab = tk.Frame(self.notebook, bg=self.palette["background"], padx=14, pady=18)
        self.notebook.add(self.overview_tab, text="Overview")
        self.notebook.add(self.analytics_tab, text="Charts")
        self.notebook.add(self.history_tab, text="History")
        self.notebook.add(self.system_tab, text="System")

        self.kpi_values: dict[str, tk.Label] = {}
        kpi_row = tk.Frame(self.overview_tab, bg=self.palette["background"])
        kpi_row.pack(fill=tk.X, pady=(0, 14))
        for key, title, initial in (
            ("fitness", "FITNESS DE L'AGENT", "—"),
            ("return", "RETURN", "—"),
            ("equity", "END CAPITAL", "—"),
            ("drawdown", "MAX DRAWDOWN", "—"),
        ):
            card = self._card(kpi_row)
            card.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
            tk.Label(card, text=title, bg=self.palette["panel"], fg=self.palette["muted"], font=("Segoe UI", 9, "bold")).pack(anchor="w")
            value = tk.Label(card, text=initial, bg=self.palette["panel"], fg=self.palette["text"], font=("Segoe UI", 21, "bold"))
            value.pack(anchor="w", pady=(12, 0))
            self.kpi_values[key] = value

        progress_card = self._card(self.overview_tab)
        progress_card.pack(fill=tk.X, pady=(0, 14))
        progress_heading = tk.Frame(progress_card, bg=self.palette["panel"])
        progress_heading.pack(fill=tk.X)
        tk.Label(progress_heading, text="SIMULATION EN COURS", bg=self.palette["panel"], fg=self.palette["text"], font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        self.progress_label = tk.Label(progress_heading, text="Waiting for simulation", bg=self.palette["panel"], fg=self.palette["muted"], font=("Segoe UI", 10))
        self.progress_label.pack(side=tk.RIGHT)
        self.progress_bar = ttk.Progressbar(progress_card, style="Training.Horizontal.TProgressbar", maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(14, 10))
        self.speed_label = tk.Label(progress_card, text="0 bars/s  ·  ETA globale —", bg=self.palette["panel"], fg=self.palette["muted"], font=("Segoe UI", 10))
        self.speed_label.pack(anchor="w")
        self.metrics_label = tk.Label(progress_card, text="Waiting for generation metrics", bg=self.palette["panel"], fg=self.palette["muted"], font=("Segoe UI", 9))
        self.metrics_label.pack(anchor="w", pady=(8, 0))

        overview_note = self._card(self.overview_tab)
        overview_note.pack(fill=tk.BOTH, expand=True)
        tk.Label(overview_note, text="Latest activity", bg=self.palette["panel"], fg=self.palette["text"], font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.activity_label = tk.Label(overview_note, text="The evaluator will start shortly.", bg=self.palette["panel"], fg=self.palette["muted"], font=("Segoe UI", 10), justify=tk.LEFT, anchor="nw")
        self.activity_label.pack(fill=tk.BOTH, expand=True, anchor="nw", pady=(12, 0))

        analytics_header = tk.Frame(self.analytics_tab, bg=self.palette["background"])
        analytics_header.pack(fill=tk.X, pady=(0, 10))
        tk.Label(analytics_header, text="Generation analytics", bg=self.palette["background"], fg=self.palette["text"], font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT)
        ttk.Button(analytics_header, text="+ Add chart", command=self.add_chart).pack(side=tk.RIGHT)
        self.chart_canvas = tk.Canvas(self.analytics_tab, bg=self.palette["background"], highlightthickness=0)
        self.chart_canvas.pack(fill=tk.BOTH, expand=True)
        self.chart_content = tk.Frame(self.chart_canvas, bg=self.palette["background"])
        self.chart_window = self.chart_canvas.create_window((0, 0), window=self.chart_content, anchor="nw")
        self.chart_canvas.bind("<Configure>", self.resize_chart_area)
        self.chart_content.bind("<Configure>", self.update_chart_scrollregion)
        self.charts: list[dict] = []
        for series in ("Fitness", "Return", "End capital", "Max drawdown"):
            self.add_chart(series)

        history_columns = ("generation", "fitness", "return", "equity", "drawdown", "duration")
        self.history_table = ttk.Treeview(self.history_tab, columns=history_columns, show="headings", style="History.Treeview")
        for column, title, width in (
            ("generation", "GENERATION", 120),
            ("fitness", "FITNESS", 150),
            ("return", "RETURN", 150),
            ("equity", "END CAPITAL", 170),
            ("drawdown", "MAX DRAWDOWN", 150),
            ("duration", "DURATION", 130),
        ):
            self.history_table.heading(column, text=title)
            self.history_table.column(column, width=width, anchor=tk.E)
        self.history_table.pack(fill=tk.BOTH, expand=True)

        system_row = tk.Frame(self.system_tab, bg=self.palette["background"])
        system_row.pack(fill=tk.X)
        self.gpu_labels: dict[str, tk.Label] = {}
        for key, title in (("device", "ACCELERATOR"), ("usage", "GPU UTILIZATION"), ("memory", "GPU MEMORY"), ("temperature", "TEMPERATURE")):
            card = self._card(system_row)
            card.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
            tk.Label(card, text=title, bg=self.palette["panel"], fg=self.palette["muted"], font=("Segoe UI", 9, "bold")).pack(anchor="w")
            value = tk.Label(card, text="Indisponible", bg=self.palette["panel"], fg=self.palette["text"], font=("Segoe UI", 17, "bold"))
            value.pack(anchor="w", pady=(12, 0))
            self.gpu_labels[key] = value
        tk.Label(self.system_tab, text="Training log", bg=self.palette["background"], fg=self.palette["text"], font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(24, 8))
        self.log_text = tk.Text(self.system_tab, height=18, bg=self.palette["panel"], fg=self.palette["muted"], insertbackground=self.palette["text"], relief=tk.FLAT, font=("Consolas", 9), state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.root.after(200, self.process_events)
        self.root.after(2500, self.refresh_gpu_telemetry)

    def _card(self, parent: tk.Widget) -> tk.Frame:
        return tk.Frame(
            parent,
            bg=self.palette["panel"],
            padx=18,
            pady=16,
            highlightbackground=self.palette["border"],
            highlightthickness=1,
        )

    def add_chart(self, series: str = "Fitness") -> None:
        card = self._card(self.chart_content)
        figure, axis = plt.subplots(figsize=(6.0, 3.2), dpi=100)
        figure.patch.set_facecolor(self.palette["panel"])
        axis.set_facecolor(self.palette["panel"])
        axis.tick_params(colors=self.palette["muted"], labelsize=8)
        axis.xaxis.label.set_color(self.palette["muted"])
        axis.yaxis.label.set_color(self.palette["muted"])
        for spine in axis.spines.values():
            spine.set_color(self.palette["border"])
        axis.grid(True, color=self.palette["border"], alpha=0.55, linewidth=0.7)
        canvas = FigureCanvasTkAgg(figure, master=card)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        controls = tk.Frame(card, bg=self.palette["panel"])
        controls.pack(fill=tk.X)
        selector = ttk.Combobox(
            controls,
            values=("Fitness", "Return", "End capital", "Max drawdown", "Turnover", "Generation duration"),
            state="readonly",
            width=24,
        )
        selector.set(series)
        selector.pack(side=tk.LEFT)
        ttk.Button(controls, text="×", width=3, command=lambda: self.remove_chart(chart)).pack(side=tk.RIGHT)
        chart = {
            "card": card,
            "figure": figure,
            "axis": axis,
            "canvas": canvas,
            "selector": selector,
        }
        self.charts.append(chart)
        selector.bind("<<ComboboxSelected>>", lambda _event, item=chart: self.update_chart(item))
        for widget in (card, controls, selector):
            widget.bind("<ButtonPress-1>", lambda event, item=chart: self.begin_chart_drag(event, item))
            widget.bind("<ButtonRelease-1>", lambda event, item=chart: self.finish_chart_drag(event, item))
        self.layout_charts()
        self.update_chart(chart)

    def remove_chart(self, chart: dict) -> None:
        if chart not in self.charts:
            return
        self.charts.remove(chart)
        chart["figure"].clear()
        chart["card"].destroy()
        self.layout_charts()

    def layout_charts(self) -> None:
        for index, chart in enumerate(self.charts):
            chart["card"].grid(
                row=index // 2,
                column=index % 2,
                sticky="nsew",
                padx=5,
                pady=5,
            )
        self.chart_content.grid_columnconfigure(0, weight=1, minsize=350)
        self.chart_content.grid_columnconfigure(1, weight=1, minsize=350)
        for row in range((len(self.charts) + 1) // 2):
            self.chart_content.grid_rowconfigure(row, weight=1)
        self.update_chart_scrollregion()

    def begin_chart_drag(self, event: tk.Event, chart: dict) -> None:
        self.drag_chart = chart

    def finish_chart_drag(self, event: tk.Event, chart: dict) -> None:
        if getattr(self, "drag_chart", None) is not chart:
            return
        target = None
        for candidate in self.charts:
            if candidate is chart:
                continue
            left = candidate["card"].winfo_rootx()
            top = candidate["card"].winfo_rooty()
            right = left + candidate["card"].winfo_width()
            bottom = top + candidate["card"].winfo_height()
            if left <= event.x_root <= right and top <= event.y_root <= bottom:
                target = candidate
                break
        if target is not None:
            source_index = self.charts.index(chart)
            target_index = self.charts.index(target)
            self.charts[source_index], self.charts[target_index] = target, chart
            self.layout_charts()
        self.drag_chart = None

    def resize_chart_area(self, event: tk.Event) -> None:
        self.chart_canvas.itemconfigure(self.chart_window, width=event.width)

    def update_chart_scrollregion(self, _event: tk.Event | None = None) -> None:
        self.chart_canvas.configure(scrollregion=self.chart_canvas.bbox("all"))

    def update_chart(self, chart: dict) -> None:
        series = chart["selector"].get()
        sources = {
            "Fitness": (self.fitnesses, "Fitness", self.palette["blue"]),
            "Return": (self.returns, "Return (%)", self.palette["green"]),
            "End capital": (self.equities, "Capital ($)", self.palette["cyan"]),
            "Max drawdown": (self.drawdowns, "Drawdown (%)", self.palette["red"]),
            "Turnover": (self.turnovers, "Turnover", "#d5a6ff"),
            "Generation duration": (self.generation_durations, "Duration (s)", "#f4bd67"),
        }
        values, ylabel, color = sources[series]
        axis = chart["axis"]
        axis.clear()
        axis.set_facecolor(self.palette["panel"])
        axis.tick_params(colors=self.palette["muted"], labelsize=8)
        axis.grid(True, color=self.palette["border"], alpha=0.55, linewidth=0.7)
        for spine in axis.spines.values():
            spine.set_color(self.palette["border"])
        axis.set_title(series, color=self.palette["text"], loc="left", fontsize=10, fontweight="bold", pad=10)
        axis.set_xlabel("Generation", color=self.palette["muted"])
        axis.set_ylabel(ylabel, color=self.palette["muted"])
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        axis.plot(self.generations[: len(values)], values, color=color, linewidth=2, marker="o", markersize=3)
        chart["canvas"].draw_idle()

    def update_all_charts(self) -> None:
        for chart in self.charts:
            self.update_chart(chart)

    def refresh_gpu_telemetry(self) -> None:
        if self.closed:
            return
        executable = shutil.which("nvidia-smi")
        telemetry_found = False
        if executable:
            try:
                result = subprocess.run(
                    [
                        executable,
                        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=True,
                )
                fields = [field.strip() for field in result.stdout.splitlines()[0].split(",")]
                if len(fields) == 4:
                    self.gpu_labels["usage"].config(text=f"{fields[0]} %")
                    self.gpu_labels["memory"].config(text=f"{fields[1]} / {fields[2]} MiB")
                    self.gpu_labels["temperature"].config(text=f"{fields[3]} °C")
                    telemetry_found = True
            except (OSError, subprocess.SubprocessError, IndexError):
                pass
        if not telemetry_found:
            for device in Path("/sys/class/drm").glob("card[0-9]"):
                device_path = device / "device"
                usage_path = device_path / "gpu_busy_percent"
                used_path = device_path / "mem_info_vram_used"
                total_path = device_path / "mem_info_vram_total"
                try:
                    if usage_path.exists():
                        usage = int(usage_path.read_text(encoding="ascii").strip())
                        self.gpu_labels["usage"].config(text=f"{usage} %")
                    if used_path.exists() and total_path.exists():
                        used = int(used_path.read_text(encoding="ascii").strip())
                        total = int(total_path.read_text(encoding="ascii").strip())
                        self.gpu_labels["memory"].config(
                            text=f"{used / 1024**2:,.0f} / {total / 1024**2:,.0f} MiB"
                        )
                    temperature_paths = sorted(
                        (device_path / "hwmon").glob("hwmon*/temp*_input")
                    )
                    if temperature_paths:
                        millidegrees = int(
                            temperature_paths[0].read_text(encoding="ascii").strip()
                        )
                        self.gpu_labels["temperature"].config(
                            text=f"{millidegrees / 1000:.0f} °C"
                        )
                    if usage_path.exists() or used_path.exists():
                        break
                except (OSError, ValueError):
                    continue
        self.root.after(2500, self.refresh_gpu_telemetry)

    def append_log(self, line: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, line + "\n")
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > 500:
            self.log_text.delete("1.0", f"{line_count - 500}.0")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

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
            elif event_type == "progress":
                self.handle_progress(payload)
            elif event_type == "system":
                self.handle_system(payload)
            elif event_type == "log":
                self.append_log(payload["line"])
            elif event_type == "done":
                self.handle_done(payload)
            elif event_type == "error":
                self.handle_error(payload)

        if not self.training_finished:
            self.root.after(200, self.process_events)
        elif processed:
            self.root.after(500, self.process_events)

    def handle_progress(self, payload: dict) -> None:
        percentage = float(payload["percentage"])
        generation = payload.get("generation")
        display_generation = generation + 1 if generation is not None else 0
        self.progress_bar["value"] = percentage
        generation_text = (
            f"Generation {display_generation}"
            if display_generation
            else "Preparing"
        )
        if self.total_generations:
            generation_text += f" / {self.total_generations}"
        self.progress_label.config(text=f"{generation_text} · {percentage:.1f} %")

        generation_eta_match = GENERATION_TIME_RE.search(
            f"Generation time: {payload['eta']}"
        )
        generation_eta = 0.0
        if generation_eta_match:
            generation_eta = (
                3600 * float(generation_eta_match.group("hours") or 0)
                + 60 * float(generation_eta_match.group("minutes") or 0)
                + float(generation_eta_match.group("seconds"))
            )
        average_duration = (
            sum(self.generation_durations) / len(self.generation_durations)
            if self.generation_durations
            else 0.0
        )
        remaining_generations = max(
            0,
            self.total_generations - display_generation,
        )
        total_eta = generation_eta + average_duration * remaining_generations
        eta_text = self.format_duration(total_eta) if total_eta else payload["eta"]
        self.speed_label.config(
            text=(
                f"{payload['rate']:,.2f} bars/s  ·  "
                f"Generation ETA {payload['eta']}  ·  Total ETA {eta_text}"
            )
        )
        self.activity_label.config(
            text=(
                f"{generation_text}\n"
                f"{payload['completed']:,} / {payload['total']:,} bars processed\n"
                f"Current speed: {payload['rate']:,.2f} bars/s\n"
                f"Estimated time remaining: {eta_text}"
            )
        )

    def handle_system(self, payload: dict) -> None:
        if "total_generations" in payload:
            self.total_generations = payload["total_generations"]
        line = payload.get("line", "")
        if line.startswith("GPU:"):
            self.gpu_labels["device"].config(text=line.partition(":")[2].strip())
        elif line.startswith("GPU memory:"):
            self.gpu_labels["memory"].config(text=line.partition(":")[2].strip())
        elif line.startswith("Accelerated population evaluation:"):
            self.gpu_labels["device"].config(text=line.partition(":")[2].strip())
        elif line == "Using CPU reference evaluation.":
            self.gpu_labels["device"].config(text="CPU")

    @staticmethod
    def format_duration(seconds: float) -> str:
        remaining = max(0, int(seconds))
        hours, remainder = divmod(remaining, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes:02d}m"
        if minutes:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"

    def handle_generation(self, payload: dict) -> None:
        metrics = payload["metrics"]
        generation = int(metrics["generation"])
        return_percent = float(metrics["return_percent"])
        fitness = metrics["best_fitness"]
        final_equity = metrics["final_equity"]
        self.generations.append(generation)
        self.returns.append(return_percent)
        duration = metrics.get("generation_time_seconds")
        self.fitnesses.append(float(fitness) if fitness is not None else float("nan"))
        self.equities.append(float(final_equity) if final_equity is not None else float("nan"))
        drawdown = metrics.get("max_drawdown_percent")
        turnover = metrics.get("turnover")
        self.drawdowns.append(float(drawdown) if drawdown is not None else float("nan"))
        self.turnovers.append(float(turnover) if turnover is not None else float("nan"))
        self.generation_durations.append(float(duration) if duration is not None else float("nan"))

        fitness_text = "—" if fitness is None else f"{float(fitness):.6f}"
        equity_text = "—" if final_equity is None else f"${float(final_equity):,.2f}"
        drawdown_text = "—" if drawdown is None else f"{float(drawdown):.2f}%"
        self.kpi_values["fitness"].config(text=fitness_text)
        self.kpi_values["return"].config(text=f"{return_percent:+.2f}%")
        self.kpi_values["equity"].config(text=equity_text)
        self.kpi_values["drawdown"].config(text=drawdown_text)
        self.status_label.config(text=f"Training · generation {generation}")
        self.latest_label.config(
            text=f"Return {return_percent:+.2f}%   ·   Fitness {fitness_text}   ·   Capital {equity_text}"
        )
        self.metrics_label.config(
            text=f"Generation {generation} completed · max drawdown {drawdown_text} · turnover {turnover if turnover is not None else '—'}"
        )
        self.progress_bar["value"] = 100
        self.progress_label.config(text=f"Generation {generation} · 100%")
        self.history_table.insert(
            "",
            tk.END,
            values=(
                generation,
                fitness_text,
                f"{return_percent:+.2f}%",
                equity_text,
                drawdown_text,
                self.format_duration(float(duration)) if duration is not None else "—",
            ),
        )
        self.update_all_charts()
        self.activity_label.config(
            text=(
                f"Latest completed generation: {generation}\n"
                f"Fitness {fitness_text} · return {return_percent:+.2f}%\n"
                f"End capital {equity_text} · max drawdown {drawdown_text}"
            )
        )
        for index, chart in enumerate(self.charts, start=1):
            chart["figure"].savefig(
                self.run_dir / f"chart_{index:02d}.png",
                dpi=120,
                bbox_inches="tight",
                facecolor=chart["figure"].get_facecolor(),
            )

    def handle_done(self, payload: dict) -> None:
        self.training_finished = True
        return_code = payload["return_code"]

        if return_code == 0:
            self.status_label.config(
                text=f"Training complete · {len(self.generations)} generations"
            )
            self.progress_bar["value"] = 100
            self.progress_label.config(text="Training complete")
            self.speed_label.config(text="All generations have completed")
        else:
            message = f"Training failed · exit code {return_code}"
            self.status_label.config(text=message)
            self.metrics_label.config(text=message)

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

    run_champion = run_dir / "champions" / "best_agent.json"
    print(f"Run champion:\n  {run_champion}")

    if args.resume_from is not None:
        resume_path = resolve_path(args.resume_from)
        if not resume_path.is_file():
            print(f"ERROR: resume champion not found: {resume_path}", file=sys.stderr)
            return 1
        if args.reset_champion:
            print("ERROR: --reset-champion cannot be combined with --resume-from", file=sys.stderr)
            return 1
        run_champion.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resume_path, run_champion)

    print("=" * 72)
    print()

    # ------------------------------------------------------------------------
    # Build command
    # ------------------------------------------------------------------------

    command = build_evolve_command(args, run_dir)

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
