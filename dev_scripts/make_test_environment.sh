#!/bin/bash
# Make a clean conda environment named test with one published version of mbirtorch.
#
#   dev_scripts/make_test_environment.sh 0.2.0rc1    # from TestPyPI
#   dev_scripts/make_test_environment.sh 0.2.0       # from PyPI
NEW_NAME="test"
VERSION="${1:?usage: make_test_environment.sh X.Y.Z[rcN]}"

conda env remove --name "$NEW_NAME" -y
conda create --name "$NEW_NAME" python=3.13 -y

# A release candidate is on TestPyPI; its dependencies are still on PyPI.
case "$VERSION" in
  *rc*) conda run -n "$NEW_NAME" pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "mbirtorch[test]==$VERSION" ;;
  *)    conda run -n "$NEW_NAME" pip install "mbirtorch[test]==$VERSION" ;;
esac

echo
echo "Now run: conda activate $NEW_NAME"
echo
