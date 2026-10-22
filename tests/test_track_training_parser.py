from __future__ import annotations

import csv
from queue import Queue

import pytest

import track_training


@pytest.mark.parametrize(
    ("duration", "expected_seconds"),
    [("3m 03s", 183.0), ("12.5s", 12.5), ("1h 02m 03s", 3723.0)],
)
def test_training_runner_records_generation_metrics(
    tmp_path, monkeypatch, duration, expected_seconds
):
    history_path = tmp_path / "generation_history.csv"
    track_training.initialize_history_csv(history_path)
    monkeypatch.setattr(track_training, "save_champion_snapshot", lambda *_: None)
    runner = track_training.TrainingRunner(
        command=[],
        run_dir=tmp_path,
        history_csv=history_path,
        events=Queue(),
    )

    for line in (
        "Generation 0000",
        "Best fitness: -4.786151",
        "Final equity: $97,626.49",
        "Return: -2.37%",
        "Max drawdown: 2.52%",
        "Turnover: 21.8239",
        f"Generation time: {duration}",
    ):
        runner.process_line(line)

    with history_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert row["generation"] == "1"
    assert float(row["best_fitness"]) == pytest.approx(-4.786151)
    assert float(row["return_percent"]) == pytest.approx(-2.37)
    assert float(row["generation_time_seconds"]) == pytest.approx(expected_seconds)


def test_training_runner_emits_simulation_progress(tmp_path):
    events = Queue()
    runner = track_training.TrainingRunner(
        command=[],
        run_dir=tmp_path,
        history_csv=tmp_path / "history.csv",
        events=events,
    )

    runner.process_line("Generations: 500")
    runner.process_line("Generation 0012")
    runner.process_line(
        "\rTraining progress: 12.50% | 1,000/8,000 bars | "
        "12.34 bars/s | ETA 9m 27s"
    )

    received = []
    while not events.empty():
        received.append(events.get_nowait())

    assert ("system", {"total_generations": 500}) in received
    progress = next(payload for kind, payload in received if kind == "progress")
    assert progress == {
        "percentage": 12.5,
        "completed": 1000,
        "total": 8000,
        "rate": 12.34,
        "eta": "9m 27s",
        "generation": 12,
    }


def test_champion_snapshot_uses_run_scoped_champion(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    champion_dir = run_dir / "champions"
    champion_dir.mkdir(parents=True)
    champion = champion_dir / track_training.CHAMPION_PATH.name
    champion.write_text("run champion", encoding="utf-8")
    monkeypatch.setattr(
        track_training,
        "CHAMPION_PATH",
        tmp_path / "unrelated" / "best_agent.json",
    )

    snapshot = track_training.save_champion_snapshot(run_dir, 3)

    assert snapshot == run_dir / "generation_0003.json"
    assert snapshot.read_text(encoding="utf-8") == "run champion"
