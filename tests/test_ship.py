"""Exercise shipping preflight in temporary repositories, stopping before mutation."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("filename, expected_status", [(".env.example", 17), (".env.local", 3)])
def test_ship_allows_public_template_but_blocks_environment_files(
    tmp_path, filename, expected_status
):
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (
        ["init", "-q"],
        ["config", "user.name", "Test Author"],
        ["config", "user.email", "author@example.test"],
        ["remote", "add", "origin", "https://example.test/repo.git"],
    ):
        subprocess.run(["git", *args], cwd=repo, check=True)
    (repo / filename).write_text("PLACEHOLDER=\n")

    # A failing check proves preflight passed and prevents staging, commit, or push.
    command_dir = tmp_path / "bin"
    command_dir.mkdir()
    check_command = command_dir / "make"
    check_command.write_text("#!/bin/sh\nexit 17\n")
    check_command.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / "scripts" / "ship.sh"
    result = subprocess.run(
        ["bash", str(script), "Update configuration template"],
        cwd=repo,
        env={**os.environ, "PATH": f"{command_dir}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == expected_status, result.stderr
    assert not subprocess.run(
        ["git", "ls-files"], cwd=repo, capture_output=True, check=True
    ).stdout
