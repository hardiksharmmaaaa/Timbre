"""Refresh the table-only project tracker after repository checks pass."""

from __future__ import annotations

import argparse
import subprocess
from datetime import date
from pathlib import Path

TRACKER_HEADER = [
    "Area",
    "Trackable item",
    "Status",
    "Evidence or exit criterion",
    "Next checkpoint",
]
ENGINEERING_AREAS = ("Pipeline", "Features", "Baselines", "Model", "Uncertainty", "Evaluation")


class TrackerFormatError(ValueError):
    """Raised when tracker.md is not a single valid Markdown table."""


def parse_tracker(text: str) -> list[list[str]]:
    """Parse and validate the table-only tracker."""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise TrackerFormatError("tracker must contain a header and separator")
    if any(not line.startswith("|") or not line.endswith("|") for line in lines):
        raise TrackerFormatError("tracker must contain only Markdown table rows")

    rows = [[cell.strip() for cell in line[1:-1].split("|")] for line in lines]
    if rows[0] != TRACKER_HEADER:
        raise TrackerFormatError("tracker header does not match the expected columns")
    if any(len(row) != len(TRACKER_HEADER) for row in rows):
        raise TrackerFormatError("every tracker row must have five columns")
    if any(cell.strip("-: ") for cell in rows[1]):
        raise TrackerFormatError("tracker separator row is invalid")
    return rows


def render_tracker(rows: list[list[str]]) -> str:
    """Render tracker rows as a Markdown table."""
    return "\n".join(f"| {' | '.join(row)} |" for row in rows) + "\n"


def find_row(rows: list[list[str]], area: str, item: str) -> list[str]:
    """Find one tracker row by its stable area and item fields."""
    matches = [row for row in rows[2:] if row[0] == area and row[1] == item]
    if len(matches) != 1:
        raise TrackerFormatError(f"expected one tracker row for {area}: {item}")
    return matches[0]


def select_next_task(rows: list[list[str]]) -> tuple[str, str, str] | None:
    """Select the first blocked task, then the first pending engineering task."""
    for target_status in ("Blocked", "Pending"):
        for row in rows[2:]:
            if row[0] in ENGINEERING_AREAS and row[2] == target_status:
                return row[0], row[1], row[4]
    return None


def refresh_tracker(text: str, verified_on: date, branch: str) -> str:
    """Refresh automated rows and derive the next engineering task."""
    rows = parse_tracker(text)
    verified_date = verified_on.isoformat()

    automated_updates = {
        ("Data safety", "Raw dataset exclusion"): (
            "Pass",
            f"No raw dataset files were tracked when checked on {verified_date}",
        ),
        ("Quality", "Ruff"): ("Pass", f"`make ci` passed Ruff on {verified_date}"),
        ("Quality", "Pytest"): ("Pass", f"`make ci` passed pytest on {verified_date}"),
        ("Quality", "Patch hygiene"): (
            "Pass",
            f"`git diff --check` passed on {verified_date}",
        ),
        ("Version control", "Branch synchronization"): (
            "Pass",
            f"Tracker refreshed for `{branch}` before commit and push on {verified_date}",
        ),
        ("Release gate", "Repository validation"): (
            "Pass",
            f"`make ci` and tracker validation passed on {verified_date}",
        ),
    }
    for key, (status, evidence) in automated_updates.items():
        row = find_row(rows, *key)
        row[2] = status
        row[3] = evidence

    workflow_row = find_row(rows, "Workflow", "Next engineering task")
    next_task = select_next_task(rows)
    if next_task is None:
        workflow_row[2] = "Complete"
        workflow_row[3] = "No blocked or pending engineering tasks remain"
        workflow_row[4] = "Define the next milestone before shipping"
    else:
        area, item, checkpoint = next_task
        workflow_row[2] = "Next"
        workflow_row[3] = f"{area}: {item}"
        workflow_row[4] = checkpoint

    return render_tracker(rows)


def run_git(repo_root: Path, *args: str) -> str:
    """Run a read-only Git command and return stripped stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_repository(repo_root: Path) -> str:
    """Validate patch hygiene and prevent raw datasets from being tracked."""
    run_git(repo_root, "diff", "--check")
    run_git(repo_root, "diff", "--cached", "--check")
    tracked_raw = run_git(repo_root, "ls-files", "--", "data/raw")
    if tracked_raw:
        raise RuntimeError("refusing to refresh tracker while data/raw contains tracked files")
    return run_git(repo_root, "branch", "--show-current")


def main() -> int:
    """Refresh tracker.md after the caller confirms repository checks passed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checks-passed",
        action="store_true",
        help="confirm make ci passed immediately before this command",
    )
    parser.add_argument("--check", action="store_true", help="fail instead of writing if stale")
    args = parser.parse_args()
    if not args.checks_passed:
        parser.error("--checks-passed is required; run make ci first")

    repo_root = Path(__file__).resolve().parents[1]
    tracker_path = repo_root / "tracker.md"
    branch = validate_repository(repo_root)
    current = tracker_path.read_text(encoding="utf-8")
    refreshed = refresh_tracker(current, date.today(), branch)

    if args.check:
        if current != refreshed:
            print("tracker.md is stale; run make tracker-update")
            return 1
        print("tracker.md is current")
        return 0

    tracker_path.write_text(refreshed, encoding="utf-8")
    next_task_row = find_row(parse_tracker(refreshed), "Workflow", "Next engineering task")
    print(f"Updated tracker.md; next task: {next_task_row[3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
