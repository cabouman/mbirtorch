#!/bin/bash
# Run the test suite from the repo root.
cd "$(dirname "$0")/.." || exit 1
pytest tests -q
