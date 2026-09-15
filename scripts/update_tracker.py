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
PROGRESS_HEADING = "## Project Progress"
ENGINEERING_AREAS = ("Pipeline", "Features", "Baselines", "Model", "Uncertainty", "Evaluation")
DELIVERY_AREAS = (
    "UI",
    "Planning",
    "Data",
    "Pipeline",
    "Features",
    "Baselines",
    "Model",
    "Uncertainty",
    "Evaluation",
)
COMPLETED_STATUSES = frozenset({"Complete", "Pass"})
EFFORT_DAYS = {
    ("UI", "Frontend experience"): 1,
    ("Planning", "Model and data plan"): 1,
    ("Data", "RAVDESS download"): 1,
    ("Data", "CREMA-D download"): 1,
    ("Data", "TESS download"): 1,
    ("Data", "RIRS_NOISES download"): 1,
    ("Pipeline", "RAVDESS clip discovery"): 1,
    ("Pipeline", "Dataset-specific parsers"): 2,
    ("Pipeline", "Canonical manifest"): 2,
    ("Pipeline", "Audio integrity screening"): 2,
    ("Pipeline", "Speaker-disjoint splits"): 2,
    ("Features", "Cached extraction"): 2,
    ("Baselines", "Logistic regression and random forest"): 2,
    ("Model", "Six-class PCA and HMM training"): 3,
    ("Model", "Hand-written Viterbi validation"): 1,
    ("Uncertainty", "Probability calibration and abstention"): 3,
    ("Evaluation", "In-domain and TESS evaluation"): 3,
    ("Evaluation", "Training-only augmentation ablation"): 2,
}


class TrackerFormatError(ValueError):
    """Raised when tracker.md is not a single valid Markdown table."""


def parse_tracker(text: str) -> list[list[str]]:
    """Parse and validate the task table before the progress footer."""
    marker = f"\n\n{PROGRESS_HEADING}\n"
    if text.count(marker) > 1:
        raise TrackerFormatError("tracker contains multiple progress sections")
    table_text = text.split(marker, maxsplit=1)[0]
    lines = [line for line in table_text.splitlines() if line.strip()]
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


def render_progress(completed: int, total: int, percentage: int, remaining_days: int) -> str:
    """Render the prominent progress summary below the tracker table."""
    working_weeks = (remaining_days + 4) // 5
    return (
        f"{PROGRESS_HEADING}\n\n"
        f"> **{percentage}% complete** — {completed} of {total} tracked delivery milestones.\n"
        ">\n"
        f"> **Estimated effort remaining: {remaining_days} developer-days "
        f"(about {working_weeks} working weeks).** This excludes model-training runtime and "
        "external approvals.\n"
    )


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


def calculate_progress(rows: list[list[str]]) -> tuple[int, int, int, int]:
    """Calculate completed milestones, percentage, and remaining developer-days."""
    delivery_rows = [row for row in rows[2:] if row[0] in DELIVERY_AREAS]
    if not delivery_rows:
        raise TrackerFormatError("tracker does not contain delivery milestones")

    completed = 0
    remaining_days = 0
    for row in delivery_rows:
        key = (row[0], row[1])
        if key not in EFFORT_DAYS:
            raise TrackerFormatError(f"missing effort estimate for {row[0]}: {row[1]}")
        if row[2] in COMPLETED_STATUSES:
            completed += 1
        else:
            remaining_days += EFFORT_DAYS[key]

    total = len(delivery_rows)
    percentage = round(completed * 100 / total)
    return completed, total, percentage, remaining_days


def refresh_tracker(text: str, verified_on: date, branch: str) -> str:
    """Refresh automated rows and derive the next engineering task."""
    rows = parse_tracker(text)
    verified_date = verified_on.isoformat()

    automated_updates = {
        ("Data safety", "Raw dataset exclusion"): (
            "Pass",
            f"No data, model, experiment, or recording files were tracked on {verified_date}",
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

    completed, total, percentage, remaining_days = calculate_progress(rows)
    return render_tracker(rows) + "\n" + render_progress(
        completed,
        total,
        percentage,
        remaining_days,
    )


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
    """Validate patch hygiene and reject tracked private data and generated artifacts."""
    run_git(repo_root, "diff", "--check")
    run_git(repo_root, "diff", "--cached", "--check")
    for protected_dir in ("data", "models", "experiments", "recordings"):
        if run_git(repo_root, "ls-files", "--", protected_dir):
            raise RuntimeError(
                f"refusing to refresh tracker while {protected_dir} contains tracked files"
            )
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
