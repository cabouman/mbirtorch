"""The dependency watch's pure checker (increment 1 of the plan at
mbirjax_plans: plans/torch_port/active/python_matrix_nightly_check.md).

Detection only: read torch's CPU package index, the version file, and
pyproject.toml; print the divergence or "none".  Writes nothing anywhere.
The gautschi nightly's watchdog line runs this same checker read-only.

Usage:
  python ci/dependency_watch.py                 # local checkout (repo root inferred)
  python ci/dependency_watch.py --remote        # read the version file and
                                                # pyproject from the public
                                                # prerelease branch on GitHub
  python ci/dependency_watch.py --json          # machine-readable verdict
"""

import argparse
import json
import os
import re
import sys
import urllib.request

CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu/torch/"
REMOTE_RAW = "https://raw.githubusercontent.com/cabouman/mbirtorch/prerelease/"
VERSION_FILE = ".github/python-versions.json"
PYPROJECT = "pyproject.toml"

# A parsed Python version must match this before it is used anywhere.
_VERSION_RE = re.compile(r"^3\.\d+$")
# A stable release version: numeric dot-separated segments only (a "+cpu"
# local-version suffix is stripped before this test).
_STABLE_RE = re.compile(r"^\d+(\.\d+)*$")
# Wheel filename: name-version-pythontag-abitag-platform.whl
_WHEEL_RE = re.compile(r"^torch-([^-]+)-([^-]+)-([^-]+)-(.+)\.whl$")


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def parse_torch_index(html):
    """The torch list from the CPU index's PEP 503 file listing.

    Returns (newest_stable_version, sorted list of Python versions like
    "3.12").  Pre-releases, yanked files, non-wheel files, free-threaded
    variant tags (cp314t), and non-Linux-x86_64 wheels are all excluded.
    """
    files = {}   # stable version -> set of python versions
    for m in re.finditer(r"<a\s+([^>]*)>([^<]+)</a>", html):
        attrs, name = m.group(1), m.group(2).strip()
        if "data-yanked" in attrs:
            continue
        wm = _WHEEL_RE.match(name)
        if wm is None:
            continue                      # not a wheel (e.g. .tar.gz)
        version, pytag, _abi, platform = wm.groups()
        version = version.split("+")[0]   # strip the +cpu local suffix
        if not _STABLE_RE.match(version):
            continue                      # pre-release (rc/a/b/dev)
        if not ("manylinux" in platform or "linux_x86_64" in platform):
            continue
        if "x86_64" not in platform:
            continue
        tm = re.match(r"^cp3(\d+)$", pytag)
        if tm is None:
            continue                      # cp314t and friends are not versions
        py = f"3.{tm.group(1)}"
        if not _VERSION_RE.match(py):
            continue
        files.setdefault(version, set()).add(py)

    if not files:
        raise ValueError("no stable torch Linux x86_64 wheels found in the index")
    newest = max(files, key=lambda v: tuple(int(x) for x in v.split(".")))
    return newest, sorted(files[newest], key=lambda v: int(v.split(".")[1]))


def parse_version_file(text):
    """The matrix from the version file.  Returns (test_list, docs_version)."""
    data = json.loads(text)
    test = list(data["test"])
    for v in test + [data["docs"]]:
        if not _VERSION_RE.match(v):
            raise ValueError(f"bad Python version in version file: {v!r}")
    return test, data["docs"]


def parse_pyproject(text):
    """The floors from pyproject.toml.  Returns (python_floor, torch_floor),
    e.g. ("3.11", "2.13")."""
    pm = re.search(r'requires-python\s*=\s*"\s*>=\s*([\d.]+)\s*"', text)
    tm = re.search(r'"torch\s*>=\s*([\d.]+)"', text)
    if pm is None or tm is None:
        raise ValueError("could not find requires-python or the torch floor "
                         "in pyproject.toml")
    return pm.group(1), tm.group(1)


def _minor(v):
    return tuple(int(x) for x in v.split(".")[:2])


def divergence(torch_release, torch_list, matrix, python_floor, torch_floor):
    """The divergence, as a dict.  Versions below the Python floor are
    reported informationally and never proposed."""
    below_floor = [v for v in torch_list if _minor(v) < _minor(python_floor)]
    eligible = [v for v in torch_list if _minor(v) >= _minor(python_floor)]
    additions = [v for v in eligible if v not in matrix]
    removals = [v for v in matrix if v not in torch_list]
    torch_newest_minor = ".".join(str(x) for x in _minor(torch_release))
    torch_advance = (torch_newest_minor
                     if _minor(torch_release) > _minor(torch_floor) else None)
    return {
        "torch_release": torch_release,
        "torch_list": torch_list,
        "matrix": matrix,
        "python_floor": python_floor,
        "torch_floor": torch_floor,
        "below_floor": below_floor,
        "additions": additions,
        "removals": removals,
        "torch_advance": torch_advance,
        "any": bool(additions or removals or torch_advance),
    }


def branch_name(d):
    """The branch name is determined entirely by the divergence."""
    parts = []
    if d["additions"]:
        parts.append("add-" + "-".join(d["additions"]))
    if d["removals"]:
        parts.append("drop-" + "-".join(d["removals"]))
    if d["torch_advance"]:
        parts.append("torch-" + d["torch_advance"])
    return "nightly/python-matrix-" + "-".join(parts) if parts else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--remote", action="store_true",
                    help="read the version file and pyproject.toml from the "
                         "public prerelease branch instead of a local checkout")
    ap.add_argument("--json", action="store_true", help="print the verdict as JSON")
    args = ap.parse_args(argv)

    # The report states three facts separately: the files read, the matrix
    # found, and the verdict -- so a missing file never reads as "no
    # divergence".
    if args.remote:
        vf_source = REMOTE_RAW + VERSION_FILE
        pp_source = REMOTE_RAW + PYPROJECT
        read = fetch
    else:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vf_source = os.path.join(root, VERSION_FILE)
        pp_source = os.path.join(root, PYPROJECT)
        read = lambda p: open(p).read()

    print(f"dependency-watch: torch index {CPU_INDEX_URL}")
    torch_release, torch_list = parse_torch_index(fetch(CPU_INDEX_URL))
    print(f"dependency-watch: torch {torch_release} supports {torch_list}")

    try:
        matrix, docs_version = parse_version_file(read(vf_source))
    except (OSError, urllib.error.URLError) as e:
        print(f"dependency-watch: VERSION FILE NOT READ ({vf_source}): {e}")
        print("dependency-watch: verdict UNKNOWN (no matrix; this is not "
              "'no divergence')")
        return 1
    print(f"dependency-watch: matrix {matrix}, docs {docs_version} ({vf_source})")

    python_floor, torch_floor = parse_pyproject(read(pp_source))
    d = divergence(torch_release, torch_list, matrix, python_floor, torch_floor)

    if args.json:
        print(json.dumps(d, indent=2))
    if d["below_floor"]:
        print(f"dependency-watch: below the {python_floor} floor, not proposed: "
              f"{d['below_floor']}")
    if d["any"]:
        print(f"dependency-watch: DIVERGENCE -> branch {branch_name(d)}")
        if d["additions"]:
            print(f"dependency-watch:   additions due: {d['additions']}")
        if d["removals"]:
            print(f"dependency-watch:   removals due: {d['removals']}")
        if d["torch_advance"]:
            print(f"dependency-watch:   torch floor advance due: "
                  f"{torch_floor} -> {d['torch_advance']}")
    else:
        print("dependency-watch: verdict none (matrix and floors match torch)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
