"""Regression test: the CLI must work when invoked as ``python -m flood_pipeline.cli``.

The dashboard's Run page (and flexth_step's pattern) invoke the module form,
not the ``flood-pipeline`` console script. Without a ``__main__`` guard the
module imports silently and exits 0 — the pipeline appears to "run" but does
nothing.
"""

import subprocess
import sys


def test_module_invocation_shows_usage() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "flood_pipeline.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage" in result.stdout, (
        "python -m flood_pipeline.cli produced no output — missing __main__ guard?"
    )
