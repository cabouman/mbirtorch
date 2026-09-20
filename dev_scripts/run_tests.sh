#!/bin/bash
# Run the test suite from the repository root, wherever this script is started from.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "Running pytest with multiple workers on all tests."

# Worker count: 4 on a GPU machine, since too many workers initializing CUDA at once can abort
# the run, and 8 on CPU.  PYTEST_NPROC overrides both.
if [ -n "${PYTEST_NPROC:-}" ]; then
  NPROC="$PYTEST_NPROC"
elif command -v nvidia-smi >/dev/null 2>&1; then
  NPROC=4
else
  NPROC=8
fi

# Use `python -m pytest`, not bare `pytest`: the latter resolves to whatever pytest console
# script is first on PATH, which on some clusters is a stale ~/.local/bin/pytest whose shebang
# Python can't import pytest.  `python -m pytest` runs pytest via the active environment's
# interpreter (the one `which python` points to), so it uses the env's pytest regardless of
# PATH ordering.
# -ra: print the short test-summary block (incl. `FAILED <nodeid>` lines) so the regression
# harness can capture WHICH tests failed for the dashboard, not just the count.
python -m pytest -ra -n "$NPROC" tests


