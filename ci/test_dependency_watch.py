"""Unit tests for the dependency watch's pure checker, on saved copies of
its inputs (the plan's increment 1): the index fixture includes a
free-threaded cp314t tag, a non-wheel file, a pre-release, a yanked file,
and non-Linux wheels, all of which must be excluded."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dependency_watch import (parse_torch_index, parse_version_file,
                              parse_pyproject, divergence, branch_name)

INDEX_HTML = """
<html><body>
<a href="#">torch-2.12.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl</a>
<a href="#">torch-2.13.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl</a>
<a href="#">torch-2.13.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl</a>
<a href="#">torch-2.13.0%2Bcpu-cp313-cp313-manylinux_2_28_x86_64.whl</a>
<a href="#">torch-2.13.0%2Bcpu-cp314-cp314-manylinux_2_28_x86_64.whl</a>
<a href="#">torch-2.13.0%2Bcpu-cp314t-cp314t-manylinux_2_28_x86_64.whl</a>
<a href="#">torch-2.13.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl</a>
<a href="#">torch-2.13.0%2Bcpu-cp312-cp312-win_amd64.whl</a>
<a href="#">torch-2.13.0.tar.gz</a>
<a href="#">torch-2.14.0rc1%2Bcpu-cp315-cp315-manylinux_2_28_x86_64.whl</a>
<a data-yanked="broken" href="#">torch-2.13.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl</a>
</body></html>
""".replace("%2B", "+")

VERSION_FILE_JSON = '{"test": ["3.11", "3.12"], "docs": "3.12"}'

PYPROJECT_TOML = '''
[project]
requires-python = ">=3.11"
dependencies = [
    "torch>=2.13",
    "numpy",
]
'''


def test_parse_torch_index_filters_and_picks_newest_stable():
    release, pys = parse_torch_index(INDEX_HTML)
    assert release == "2.13.0"          # rc excluded, yanked 2.13.1 excluded
    assert pys == ["3.11", "3.12", "3.13", "3.14"]   # cp314t and non-linux excluded


def test_parse_version_file():
    matrix, docs = parse_version_file(VERSION_FILE_JSON)
    assert matrix == ["3.11", "3.12"]
    assert docs == "3.12"


def test_parse_pyproject_floors():
    python_floor, torch_floor = parse_pyproject(PYPROJECT_TOML)
    assert python_floor == "3.11"
    assert torch_floor == "2.13"


def test_multi_version_addition_with_below_floor_exclusion():
    d = divergence("2.13.0", ["3.10", "3.11", "3.12", "3.13", "3.14"],
                   ["3.11", "3.12"], "3.11", "2.13")
    assert d["below_floor"] == ["3.10"]           # informational only
    assert d["additions"] == ["3.13", "3.14"]
    assert d["removals"] == []
    assert d["torch_advance"] is None
    assert branch_name(d) == "nightly/python-matrix-add-3.13-3.14"


def test_removal_due_when_torch_drops_a_version():
    d = divergence("2.13.0", ["3.12", "3.13"], ["3.11", "3.12"], "3.11", "2.13")
    assert d["removals"] == ["3.11"]
    assert d["additions"] == ["3.13"]
    assert branch_name(d) == "nightly/python-matrix-add-3.13-drop-3.11"


def test_torch_floor_advance():
    d = divergence("2.14.1", ["3.11", "3.12"], ["3.11", "3.12"], "3.11", "2.13")
    assert d["additions"] == [] and d["removals"] == []
    assert d["torch_advance"] == "2.14"
    assert d["any"]
    assert branch_name(d) == "nightly/python-matrix-torch-2.14"


def test_no_divergence():
    d = divergence("2.13.0", ["3.11", "3.12"], ["3.11", "3.12"], "3.11", "2.13")
    assert not d["any"]
    assert branch_name(d) is None
