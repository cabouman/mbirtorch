#!/bin/bash
# Editable install into the ACTIVE environment (mirror of mbirjax's dev_scripts
# convention).  Run from the repo root or this directory.
cd "$(dirname "$0")/.." || exit 1
pip uninstall -y mbirtorch
pip install -e ".[test]"
