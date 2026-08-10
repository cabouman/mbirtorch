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


def compose(d, version_file_text, pyproject_text, base_sha=None):
    """The pull request the watch would open for divergence ``d``: branch,
    title, body, and the complete new content of every edited file.  Pure
    composition; nothing is written anywhere."""
    if not d["any"]:
        return None
    edits = {}

    new_test = sorted((set(d["matrix"]) | set(d["additions"])) - set(d["removals"]),
                      key=_minor)
    data = json.loads(version_file_text)
    if d["additions"] or d["removals"]:
        data["test"] = new_test
        if data["docs"] in d["removals"]:
            data["docs"] = new_test[-1]
        edits[VERSION_FILE] = json.dumps(data, indent=2) + "\n"

    new_pyproject = pyproject_text
    if d["removals"]:
        new_floor = new_test[0]
        new_pyproject = re.sub(r'(requires-python\s*=\s*")>=[\d.]+(")',
                               rf"\g<1>>={new_floor}\g<2>", new_pyproject)
    if d["torch_advance"]:
        new_pyproject = re.sub(r'("torch)\s*>=\s*[\d.]+(")',
                               rf"\g<1>>={d['torch_advance']}\g<2>", new_pyproject)
    if new_pyproject != pyproject_text:
        edits[PYPROJECT] = new_pyproject

    def _join(vs):
        return vs[0] if len(vs) == 1 else ", ".join(vs[:-1]) + " and " + vs[-1]

    title_parts = []
    if d["additions"]:
        title_parts.append(f"add Python {_join(d['additions'])} to the CI test matrix")
    if d["removals"]:
        title_parts.append(f"drop Python {_join(d['removals'])}")
    if d["torch_advance"]:
        title_parts.append(f"advance the torch floor to {d['torch_advance']}")
    title = "; ".join(title_parts)
    title = title[0].upper() + title[1:]

    body = ["Opened by the dependency watch (plans repo: "
            "plans/torch_port/active/python_matrix_nightly_check.md).",
            "",
            f"Trigger: torch {d['torch_release']} publishes CPU-index Linux "
            f"x86_64 wheels for Python {_join(d['torch_list'])}.",
            "",
            "Policy: mbirtorch tests the Python versions torch supports, and "
            "advances its torch floor deliberately."]
    if base_sha:
        body += ["", f"Branch cut from `{base_sha}`."]
    if d["removals"]:
        body += ["", "This includes a REMOVAL.  CI cannot prove a removal "
                     "correct; the reviewer must judge it."]
    if d["torch_advance"]:
        body += ["", "The torch floor advance's green checks cover the CPU "
                     "suite; the Triton and CUDA paths are proven by the "
                     "cluster nightly after the merge."]
    return {"branch": branch_name(d), "title": title,
            "body": "\n".join(body), "edits": edits}


def consecutive_night_status(state_path, branch):
    """The two-night confirmation state.  Returns 'quiet', 'first sighting',
    or 'confirmed', and records tonight's observation in ``state_path``
    (the only file this checker ever writes)."""
    previous = None
    if os.path.exists(state_path):
        with open(state_path) as f:
            previous = f.read().strip() or None
    os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
    with open(state_path, "w") as f:
        f.write(branch or "")
    if branch is None:
        return "quiet"
    return "confirmed" if previous == branch else "first sighting"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--remote", action="store_true",
                    help="read the version file and pyproject.toml from the "
                         "public prerelease branch instead of a local checkout")
    ap.add_argument("--dry-run", action="store_true",
                    help="compose and print the pull request the watch would "
                         "open (branch, title, body, every edited file); "
                         "write nothing but the --state file")
    ap.add_argument("--act", action="store_true",
                    help="on a divergence confirmed on two consecutive "
                         "nights: apply the edits to the working tree, write "
                         "the pull-request body beside the state file, and "
                         "emit branch/title/act outputs for the workflow "
                         "steps that push and open the pull request")
    ap.add_argument("--state", default=None,
                    help="path of the two-night confirmation state file")
    ap.add_argument("--base-sha", default=None,
                    help="commit identifier the read is pinned to (recorded "
                         "in the pull-request body)")
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

    status = None
    if args.state:
        status = consecutive_night_status(args.state, branch_name(d))
        print(f"dependency-watch: two-night status: {status}")
        if status == "first sighting":
            print("dependency-watch: a live watch would wait for tomorrow's "
                  "confirmation before acting")

    if args.dry_run and d["any"]:
        pr = compose(d, read(vf_source), read(pp_source), base_sha=args.base_sha)
        print()
        print("dependency-watch: DRY RUN -- the pull request the watch would open:")
        print(f"  branch: {pr['branch']}")
        print(f"  title:  {pr['title']}")
        print("  body:")
        for line in pr["body"].splitlines():
            print(f"    {line}")
        for path, content in pr["edits"].items():
            print(f"  edited file: {path}")
            for line in content.splitlines():
                print(f"    | {line}")

    if args.act:
        confirmed = args.state and status == "confirmed"
        act = bool(d["any"] and confirmed)
        if d["any"] and not confirmed:
            print("dependency-watch: not acting (waiting for the two-night "
                  "confirmation)")
        if act:
            pr = compose(d, read(vf_source), read(pp_source),
                         base_sha=args.base_sha)
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for path, content in pr["edits"].items():
                with open(os.path.join(root, path), "w") as f:
                    f.write(content)
                print(f"dependency-watch: edited {path}")
            body_path = os.path.join(os.path.dirname(args.state), "pr-body.md")
            with open(body_path, "w") as f:
                f.write(pr["body"])
            _emit_output("branch", pr["branch"])
            _emit_output("title", pr["title"])
            _emit_output("body_path", body_path)
        _emit_output("act", "true" if act else "false")
    return 0


def _emit_output(name, value):
    """One output for the workflow's later steps (GitHub's output file)."""
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            f.write(f"{name}={value}\n")


if __name__ == "__main__":
    sys.exit(main())
