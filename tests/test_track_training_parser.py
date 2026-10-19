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
