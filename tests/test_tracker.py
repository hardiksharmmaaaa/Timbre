from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path

import pytest

from scripts.update_tracker import (
    TrackerFormatError,
    calculate_progress,
    find_row,
    parse_tracker,
    refresh_tracker,
    render_progress,
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
    return "\n".join(rows) + "\n\n" + render_progress(0, len(task_rows), 0, 0)


def test_refresh_tracker_selects_first_blocked_engineering_task() -> None:
    text = tracker_text(
        [
            "| Pipeline | Dataset-specific parsers | Pending | Missing parsers | Add parsers |",
            (
                "| Model | Six-class PCA and HMM training | Blocked | Missing features | "
                "Fix model inputs |"
            ),
        ]
    )

    refreshed = refresh_tracker(text, date(2026, 9, 12), "ui")
    rows = parse_tracker(refreshed)

    assert find_row(rows, "Workflow", "Next engineering task")[2:] == [
        "Next",
        "Model: Six-class PCA and HMM training",
        "Fix model inputs",
    ]
    assert find_row(rows, "Quality", "Pytest")[3] == "`make ci` passed pytest on 2026-09-12"
    assert "\n\n## Project Progress\n\n> **0% complete**" in refreshed
    assert "Estimated effort remaining: 5 developer-days" in refreshed
    assert "| Progress |" not in refreshed


def test_refresh_tracker_selects_pending_when_nothing_is_blocked() -> None:
    text = tracker_text(
        [
            "| Pipeline | RAVDESS clip discovery | Complete | Discovery works | Maintain it |",
            (
                "| Model | Six-class PCA and HMM training | Pending | Awaiting training | "
                "Train model |"
            ),
        ]
    )

    rows = parse_tracker(refresh_tracker(text, date(2026, 9, 12), "main"))

    assert select_next_task(rows) == (
        "Model",
        "Six-class PCA and HMM training",
        "Train model",
    )
    assert calculate_progress(rows) == (1, 2, 50, 3)


def test_parse_tracker_rejects_non_table_content() -> None:
    with pytest.raises(TrackerFormatError, match="only Markdown table rows"):
        parse_tracker("# Tracker\n\n| Area | Trackable item | Status | Evidence | Next |\n")


def test_repository_tracker_has_table_and_highlighted_progress_footer() -> None:
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
    assert calculate_progress(rows) == (7, 18, 39, 24)
    tracker_text = tracker_path.read_text(encoding="utf-8")
    assert "\n\n## Project Progress\n\n> **39% complete**" in tracker_text
    assert "Estimated effort remaining: 24 developer-days" in tracker_text
    assert "| Progress |" not in tracker_text


def test_validate_repository_rejects_tracked_raw_data(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    raw_file = tmp_path / "data" / "raw" / "sample.wav"
    raw_file.parent.mkdir(parents=True)
    raw_file.write_bytes(b"not audio")
    subprocess.run(["git", "add", "data/raw/sample.wav"], cwd=tmp_path, check=True)

    with pytest.raises(RuntimeError, match="data/raw contains tracked files"):
        validate_repository(tmp_path)
