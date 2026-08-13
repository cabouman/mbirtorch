#!/bin/bash
# Check a published mbirtorch wheel: install it into a clean environment and
# run the test suite against the INSTALLED package, the way a user gets it.
#
# Usage:
#   dev_scripts/check_published_wheel.sh                    # newest from PyPI
#   dev_scripts/check_published_wheel.sh --testpypi         # newest from TestPyPI
#   dev_scripts/check_published_wheel.sh --version 0.1.0    # a specific version
#   dev_scripts/check_published_wheel.sh --wheel dist/mbirtorch-0.1.0-py3-none-any.whl
#
# The tests run from a COPY of tests/ in a temporary directory, so pytest
# cannot import the repo checkout by accident; the goldens stay opt-in as in
# CI.  Set PYTHON to use a specific interpreter (default: python3).
set -euo pipefail

cd "$(dirname "$0")/.."
INDEX="pypi"
VERSION=""
WHEEL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --testpypi) INDEX="testpypi"; shift ;;
    --version)  VERSION="==$2"; shift 2 ;;
    --wheel)    WHEEL="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
echo "Clean environment: $WORK"

"${PYTHON:-python3}" -m venv "$WORK/venv"
PY="$WORK/venv/bin/python"
"$PY" -m pip install --quiet --upgrade pip

if [[ -n "$WHEEL" ]]; then
  echo "Installing $WHEEL"
  "$PY" -m pip install "$WHEEL"
elif [[ "$INDEX" == "testpypi" ]]; then
  # TestPyPI does not host the dependencies; those come from PyPI.
  echo "Installing mbirtorch$VERSION from TestPyPI"
  "$PY" -m pip install --index-url https://test.pypi.org/simple/ \
        --extra-index-url https://pypi.org/simple "mbirtorch$VERSION"
else
  echo "Installing mbirtorch$VERSION from PyPI"
  "$PY" -m pip install "mbirtorch$VERSION"
fi

"$PY" -m pip install --quiet pytest pytest-xdist
"$PY" -c "import mbirtorch, os; print('Installed mbirtorch', mbirtorch.__version__, 'at', os.path.dirname(mbirtorch.__file__))"

cp -R tests "$WORK/tests"
rm -rf "$WORK/tests/goldens" "$WORK/tests/__pycache__"
# Register the goldens mark (normally done by pyproject.toml, absent here).
printf '[pytest]\nmarkers =\n    goldens: opt-in cross-framework parity tests\n' > "$WORK/pytest.ini"

cd "$WORK"
"$PY" -m pytest -n 4 -m "not goldens" tests
echo "PASS: the installed wheel ran the suite clean."
