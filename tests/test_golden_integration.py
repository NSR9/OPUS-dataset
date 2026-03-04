"""Integration tests for the golden dataset builder allocation."""
import os
import subprocess
import sys

import pytest


SCRIPT = "scripts/build_golden_dataset.py"


@pytest.mark.parametrize("n", [64, 128, 256, 512])
def test_dry_run_succeeds(n):
    """Dry run at various N values should succeed and show correct total."""
    result = subprocess.run(
        [sys.executable, SCRIPT, "--total-samples", str(n), "--dry-run"],
        capture_output=True, text=True, timeout=10,
        env={"PYTHONPATH": ".", **os.environ},
    )
    assert result.returncode == 0, f"Failed at N={n}: {result.stderr}"
    output = result.stdout + result.stderr
    assert f"Total samples: {n}" in output


def test_dry_run_too_few_samples():
    """N < num_datasets should fail."""
    result = subprocess.run(
        [sys.executable, SCRIPT, "--total-samples", "30", "--dry-run"],
        capture_output=True, text=True, timeout=10,
        env={"PYTHONPATH": ".", **os.environ},
    )
    assert result.returncode != 0


def test_dry_run_warns_non_power_of_2():
    """N=100 should warn about not being power of 2."""
    result = subprocess.run(
        [sys.executable, SCRIPT, "--total-samples", "100", "--dry-run"],
        capture_output=True, text=True, timeout=10,
        env={"PYTHONPATH": ".", **os.environ},
    )
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "not a power of 2" in output.lower() or "NOT power of 2" in output
