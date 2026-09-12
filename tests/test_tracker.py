from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path

import pytest

from scripts.update_tracker import (
    TrackerFormatError,
    find_row,
    parse_tracker,
    refresh_tracker,
    select_next_task,
    validate_repository,
)


def tracker_text(task_rows: list[str]) -> str:
    rows = [
        "| Area | Trackable item | Status | Evidence or exit criterion | Next checkpoint |",
        "|---|---|---|---|---|",
        "| Workflow | Next engineering task | Next | Old task | Old checkpoint |",
        "| Data safety | Raw dataset exclusion | Pass | Old evidence | Keep data ignored |",
        "| Quality | Ruff | Pass | Old evidence | Run CI |",
        "| Quality | Pytest | Pass | Old evidence | Add tests |",
        "| Quality | Patch hygiene | Pass | Old evidence | Check patches |",
        "| Version control | Branch synchronization | Pass | Old evidence | Push changes |",
        *task_rows,
        "| Release gate | Repository validation | Pass | Old evidence | Keep checks passing |",
    ]
    return "\n".join(rows) + "\n"


def test_refresh_tracker_selects_first_blocked_engineering_task() -> None:
    text = tracker_text(
        [
            "| Pipeline | Parse datasets | Pending | Missing parsers | Add parsers |",
            "| Model | Train model | Blocked | Missing features | Fix model inputs |",
        ]
    )

    refreshed = refresh_tracker(text, date(2026, 9, 12), "ui")
    rows = parse_tracker(refreshed)

    assert find_row(rows, "Workflow", "Next engineering task")[2:] == [
        "Next",
        "Model: Train model",
        "Fix model inputs",
    ]
    assert find_row(rows, "Quality", "Pytest")[3] == "`make ci` passed pytest on 2026-09-12"


def test_refresh_tracker_selects_pending_when_nothing_is_blocked() -> None:
    text = tracker_text(
        [
            "| Pipeline | Parse datasets | Complete | Parsers exist | Maintain parsers |",
            "| Model | Train model | Pending | Awaiting training | Train model |",
        ]
    )

    rows = parse_tracker(refresh_tracker(text, date(2026, 9, 12), "main"))

    assert select_next_task(rows) == ("Model", "Train model", "Train model")
    assert find_row(rows, "Workflow", "Next engineering task")[3] == "Model: Train model"


def test_parse_tracker_rejects_non_table_content() -> None:
    with pytest.raises(TrackerFormatError, match="only Markdown table rows"):
        parse_tracker("# Tracker\n\n| Area | Trackable item | Status | Evidence | Next |\n")


def test_repository_tracker_is_table_only_and_current_next_task() -> None:
    tracker_path = Path(__file__).resolve().parents[1] / "tracker.md"
    rows = parse_tracker(tracker_path.read_text(encoding="utf-8"))

    assert find_row(rows, "Workflow", "Next engineering task")[3] == (
        "Pipeline: RAVDESS clip discovery"
    )
    assert select_next_task(rows) == (
        "Pipeline",
        "RAVDESS clip discovery",
        "Fix discovery using validated filename fields and add a regression test",
    )


def test_validate_repository_rejects_tracked_raw_data(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    raw_file = tmp_path / "data" / "raw" / "sample.wav"
    raw_file.parent.mkdir(parents=True)
    raw_file.write_bytes(b"not audio")
    subprocess.run(["git", "add", "data/raw/sample.wav"], cwd=tmp_path, check=True)

    with pytest.raises(RuntimeError, match="data/raw contains tracked files"):
        validate_repository(tmp_path)
