"""Smoke test: the package imports and reports a version."""

import mbirtorch


def test_import():
    assert mbirtorch.__version__
