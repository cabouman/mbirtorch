#!/bin/bash
# Make a clean conda environment named test and install mbirtorch into it.
#
#   dev_scripts/make_test_environment.sh             # latest release, from PyPI
#   dev_scripts/make_test_environment.sh 0.2.0rc1    # release candidate, from TestPyPI
NEW_NAME="test"

conda env remove --name "$NEW_NAME" -y
conda create --name "$NEW_NAME" python=3.13 -y

# A release candidate is on TestPyPI; its dependencies are still on PyPI.
case "$1" in
  "")   conda run -n "$NEW_NAME" pip install "mbirtorch[test]" ;;
  *rc*) conda run -n "$NEW_NAME" pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "mbirtorch[test]==$1" ;;
  *)    conda run -n "$NEW_NAME" pip install "mbirtorch[test]==$1" ;;
esac

echo
echo "Now run:"
echo "  conda activate $NEW_NAME"
echo "  dev_scripts/run_tests.sh"
echo
