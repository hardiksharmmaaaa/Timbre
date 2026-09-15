"""Run deterministic regressions against the actual browser scripts, without a browser."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_frontend_runtime():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for frontend runtime regression tests")
    script = Path(__file__).with_name("frontend_runtime.cjs")
    result = subprocess.run(
        [node, "--test", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
