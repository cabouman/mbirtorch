"""Unit tests for the dependency watch's pure checker, on saved copies of
its inputs (the plan's increment 1): the index fixture includes a
free-threaded cp314t tag, a non-wheel file, a pre-release, a yanked file,
and non-Linux wheels, all of which must be excluded."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dependency_watch import (parse_torch_index, parse_version_file,
                              parse_pyproject, parse_runner_manifest,
                              parse_torch_exclusions, parse_ledger,
                              divergence, branch_name)

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

PYPROJECT_TOML_212 = PYPROJECT_TOML.replace('"torch>=2.13"', '"torch>=2.12,!=2.12.1,!=2.14.0"')

# Validated minors 2.12 .. 2.17 (six), one rejected patch, one unavailable release.
LEDGER_TXT = '''# torch ledger
2.12.0 validated 2026-06-01 bootstrap
2.13.0 validated 2026-09-03 bootstrap:env-at-canary-enablement python=3.12
2.14.0 candidate 2026-09-04 pypi python=3.12
2.14.0 validated 2026-09-04 regression_gpu_20260904T030512Z_26bd0ea9_g0001.yaml python=3.12
2.15.0 validated 2026-12-01 run.yaml
2.16.0 validated 2027-03-01 run.yaml
2.16.1 rejected 2027-03-20 run_g0007.yaml: hard gate FAIL python=3.12
2.17.0 validated 2027-06-01 run.yaml
2.18.0 unavailable 2027-09-02 index=cu130 index-newest=2.17.0 python=3.12
'''
LEDGER = parse_ledger(LEDGER_TXT)


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


RUNNER_MANIFEST_JSON = """[
  {"version": "3.15.0-rc.2", "stable": false},
  {"version": "3.14.2", "stable": true},
  {"version": "3.14.0", "stable": true},
  {"version": "3.13.9", "stable": true},
  {"version": "3.12.12", "stable": true},
  {"version": "3.11.14", "stable": true}
]"""


def test_parse_runner_manifest_stable_minors_only():
    minors = parse_runner_manifest(RUNNER_MANIFEST_JSON)
    assert minors == {"3.11", "3.12", "3.13", "3.14"}   # 3.15 rc excluded


def test_version_on_torch_index_but_not_on_runners_is_not_proposed():
    d = divergence("2.13.0", ["3.11", "3.12", "3.13", "3.14", "3.15"],
                   ["3.11", "3.12"], "3.11", "2.13",
                   runner_minors={"3.11", "3.12", "3.13", "3.14"})
    assert d["not_on_runners"] == ["3.15"]        # informational only
    assert d["additions"] == ["3.13", "3.14"]
    assert branch_name(d) == "nightly/python-matrix-add-3.13-3.14"


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


def test_parse_torch_exclusions():
    assert parse_torch_exclusions(PYPROJECT_TOML) == []
    assert parse_torch_exclusions(PYPROJECT_TOML_212) == ["2.12.1", "2.14.0"]
    assert parse_pyproject(PYPROJECT_TOML_212) == ("3.11", "2.12")   # floor still parsed


def test_parse_ledger_latest_line_wins():
    assert LEDGER["2.14.0"]["state"] == "validated"        # candidate line superseded
    assert LEDGER["2.16.1"]["state"] == "rejected"
    assert LEDGER["2.16.1"]["evidence"].startswith("run_g0007.yaml: hard gate FAIL")
    assert LEDGER["2.18.0"]["state"] == "unavailable"
    assert len(LEDGER) == 8
    assert parse_ledger("# only a comment\n\n") == {}
    import pytest
    with pytest.raises(ValueError):            # fail closed on anything unreadable
        parse_ledger("2.1.0 bogus 2026-01-01 x\n")
    with pytest.raises(ValueError):
        parse_ledger("garbage line\n")


def test_torch_floor_advance_by_window_rule():
    # Six validated minors (2.12..2.17), FLOOR_WINDOW=4, FLOOR_SLACK=1: with the floor at 2.12
    # five validated minors sit above it, so the floor advances to the oldest of the newest
    # four (2.14); with the floor at 2.13 only four sit above it and nothing moves.
    d = divergence("2.18.0", ["3.11", "3.12"], ["3.11", "3.12"], "3.11", "2.12", ledger=LEDGER)
    assert d["additions"] == [] and d["removals"] == []
    assert d["torch_advance"] == "2.14"
    assert d["newest_validated"] == "2.17.0"
    assert d["any"]
    assert branch_name(d) == "nightly/python-matrix-torch-2.14-exclude-2.16.1"
    d2 = divergence("2.18.0", ["3.11", "3.12"], ["3.11", "3.12"], "3.11", "2.13",
                    ledger=LEDGER, exclusions=("2.16.1",))
    assert d2["torch_advance"] is None and d2["torch_exclusions"] == []
    assert not d2["any"]


def test_torch_floor_rule_below_window_never_advances():
    small = parse_ledger("2.13.0 validated 2026-09-03 x\n2.14.0 validated 2026-10-01 x\n"
                         "2.15.0 validated 2026-12-01 x\n")
    d = divergence("2.15.0", ["3.11", "3.12"], ["3.11", "3.12"], "3.11", "2.13", ledger=small)
    assert d["torch_advance"] is None and not d["any"]


def test_unknown_ledger_proposes_nothing_torch():
    d = divergence("2.18.0", ["3.11", "3.12"], ["3.11", "3.12"], "3.11", "2.12", ledger=None)
    assert d["ledger_known"] is False
    assert d["torch_advance"] is None and d["torch_exclusions"] == []
    assert not d["any"]


def test_exclusion_proposed_only_for_rejected_at_or_above_floor():
    d = divergence("2.18.0", ["3.11", "3.12"], ["3.11", "3.12"], "3.11", "2.13", ledger=LEDGER)
    assert d["torch_exclusions"] == ["2.16.1"]              # rejected, above the floor
    assert d["torch_rejected"]["2.16.1"]["date"] == "2027-03-20"
    assert branch_name(d) == "nightly/python-matrix-exclude-2.16.1"
    low = parse_ledger("2.12.5 rejected 2026-07-01 x\n")
    d2 = divergence("2.13.0", ["3.11", "3.12"], ["3.11", "3.12"], "3.11", "2.13", ledger=low)
    assert d2["torch_exclusions"] == []                     # below the floor: irrelevant


def test_no_divergence():
    d = divergence("2.13.0", ["3.11", "3.12"], ["3.11", "3.12"], "3.11", "2.13")
    assert not d["any"]
    assert branch_name(d) is None


def test_compose_addition_edits_only_the_version_file():
    import json
    from dependency_watch import compose
    d = divergence("2.13.0", ["3.11", "3.12", "3.13", "3.14"],
                   ["3.11", "3.12"], "3.11", "2.13")
    pr = compose(d, VERSION_FILE_JSON, PYPROJECT_TOML, base_sha="abc1234")
    assert pr["branch"] == "nightly/python-matrix-add-3.13-3.14"
    assert pr["title"] == "Add Python 3.13 and 3.14 to the CI test matrix"
    assert "torch 2.13.0" in pr["body"] and "abc1234" in pr["body"]
    assert list(pr["edits"]) == [".github/python-versions.json"]
    data = json.loads(pr["edits"][".github/python-versions.json"])
    assert data["test"] == ["3.11", "3.12", "3.13", "3.14"]
    assert data["docs"] == "3.12"                       # untouched by additions


def test_compose_removal_bumps_floor_and_moves_docs_pin():
    from dependency_watch import compose
    d = divergence("2.13.0", ["3.11", "3.13"], ["3.11", "3.12"], "3.11", "2.13")
    # 3.12 removed (docs pin!), 3.13 added; floor stays 3.11 since it remains.
    pr = compose(d, VERSION_FILE_JSON, PYPROJECT_TOML)
    import json
    data = json.loads(pr["edits"][".github/python-versions.json"])
    assert data["test"] == ["3.11", "3.13"]
    assert data["docs"] == "3.13"                       # moved off the removed pin
    assert "REMOVAL" in pr["body"]
    # A removal of the floor version itself bumps requires-python.
    d2 = divergence("2.13.0", ["3.12"], ["3.11", "3.12"], "3.11", "2.13")
    pr2 = compose(d2, VERSION_FILE_JSON, PYPROJECT_TOML)
    assert 'requires-python = ">=3.12"' in pr2["edits"]["pyproject.toml"]


def test_compose_torch_advance_edits_pyproject_only():
    from dependency_watch import compose
    # Floor 2.12 -> 2.14 by the window rule; the 2.12.1 exclusion falls below the new floor and
    # goes, the 2.14.0 one stays, and the ledger's rejected 2.16.1 joins.
    d = divergence("2.18.0", ["3.11", "3.12"], ["3.11", "3.12"], "3.11", "2.12",
                   ledger=LEDGER, exclusions=("2.12.1", "2.14.0"))
    pr = compose(d, VERSION_FILE_JSON, PYPROJECT_TOML_212)
    assert list(pr["edits"]) == ["pyproject.toml"]
    assert '"torch>=2.14,!=2.14.0,!=2.16.1"' in pr["edits"]["pyproject.toml"]
    assert "CPU suite" in pr["body"] and "2027-03-20" in pr["body"]
    assert pr["title"] == "Advance the torch floor to 2.14; exclude torch 2.16.1"


def test_compose_exclusion_alone_edits_pyproject():
    from dependency_watch import compose
    d = divergence("2.18.0", ["3.11", "3.12"], ["3.11", "3.12"], "3.11", "2.13", ledger=LEDGER)
    pr = compose(d, VERSION_FILE_JSON, PYPROJECT_TOML)
    assert list(pr["edits"]) == ["pyproject.toml"]
    assert '"torch>=2.13,!=2.16.1"' in pr["edits"]["pyproject.toml"]
    assert pr["branch"] == "nightly/python-matrix-exclude-2.16.1"
    assert pr["title"] == "Exclude torch 2.16.1"
    assert "rejected by the cluster nightly" in pr["body"]


def test_two_night_confirmation(tmp_path):
    from dependency_watch import consecutive_night_status
    state = str(tmp_path / "state" / "last")
    assert consecutive_night_status(state, "nightly/x") == "first sighting"
    assert consecutive_night_status(state, "nightly/x") == "confirmed"
    assert consecutive_night_status(state, "nightly/y") == "first sighting"
    assert consecutive_night_status(state, None) == "quiet"
    assert consecutive_night_status(state, "nightly/y") == "first sighting"
