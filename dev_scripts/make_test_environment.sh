#!/bin/bash
# Make a clean conda environment named test and install mbirtorch into it.
# Run with source so the environment stays active in your shell:
#
#   source dev_scripts/make_test_environment.sh             # latest release, from PyPI
#   source dev_scripts/make_test_environment.sh 0.2.0rc1    # release candidate, from TestPyPI
NEW_NAME="test"

if [ "$CONDA_DEFAULT_ENV" = "$NEW_NAME" ]; then
    conda deactivate
fi

conda env remove --name "$NEW_NAME" -y
conda create --name "$NEW_NAME" python=3.13 -y
conda activate "$NEW_NAME"

# A release candidate is on TestPyPI; its dependencies are still on PyPI.
case "$1" in
  "")   pip install "mbirtorch[test]" ;;
  *rc*) pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "mbirtorch[test]==$1" ;;
  *)    pip install "mbirtorch[test]==$1" ;;
esac

echo
echo "The $NEW_NAME environment is active.  Now run: dev_scripts/run_tests.sh"
echo
