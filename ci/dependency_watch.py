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
# The Python versions GitHub's hosted runners can install (setup-python's
# source of truth).  A version torch supports but the runners lack cannot
# be tested and must not be proposed.
RUNNER_MANIFEST_URL = ("https://raw.githubusercontent.com/actions/"
                       "python-versions/main/versions-manifest.json")
REMOTE_RAW = "https://raw.githubusercontent.com/cabouman/mbirtorch/prerelease/"
VERSION_FILE = ".github/python-versions.json"
PYPROJECT = "pyproject.toml"
# The torch version ledger the cluster nightly's dependency canary writes (plans repo:
# plans/mbirtorch_metrics/torch_version_policy_plan.md).  Lines "<version> <state> <date>
# <evidence>", states candidate | validated | rejected | unavailable, the latest line per
# version wins.  The watch proposes a torch floor only from validated minors, and an
# exclusion only for a rejected version.
LEDGER_URL = ("https://raw.githubusercontent.com/cabouman/mbirtorch_metrics/main/"
              "state/gpu/torch_ledger.txt")
# The floor keeps this many validated minors installable and advances only when a newer
# validated minor would otherwise widen the window.  Four minors is about a year of releases.
FLOOR_WINDOW = 4
# Hysteresis: the floor advances only when this many validated minors MORE than the window sit
# above it, so a proposal comes about every two minors and moves the floor by two, rather than a
# pull request per release.
FLOOR_SLACK = 1
_LEDGER_STATES = {"candidate", "validated", "rejected", "unavailable"}

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


def parse_runner_manifest(text):
    """The Python versions GitHub's runners install, as minors like "3.12".
    Only entries marked stable count; release candidates do not make a
    version testable."""
    minors = set()
    for entry in json.loads(text):
        if not entry.get("stable"):
            continue
        parts = str(entry.get("version", "")).split(".")
        if len(parts) >= 2 and parts[0] == "3" and parts[1].isdigit():
            minors.add(f"3.{parts[1]}")
    if not minors:
        raise ValueError("no stable Python versions found in the runner manifest")
    return minors


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
    tm = re.search(r'"torch\s*>=\s*([\d.]+)[^"]*"', text)
    if pm is None or tm is None:
        raise ValueError("could not find requires-python or the torch floor "
                         "in pyproject.toml")
    return pm.group(1), tm.group(1)


def _minor(v):
    return tuple(int(x) for x in v.split(".")[:2])


def _vtuple(v):
    return tuple(int(x) for x in v.split("."))


def parse_torch_exclusions(text):
    """The versions pyproject already excludes on the torch line, e.g. ["2.14.0"]."""
    tm = re.search(r'"torch\s*>=\s*[\d.]+([^"]*)"', text)
    if tm is None:
        return []
    return re.findall(r"!=\s*([\d.]+)", tm.group(1))


def parse_ledger(text):
    """The torch ledger as {version: {"state", "date", "evidence"}}; the
    latest line per version wins; comments, blanks and malformed lines are
    skipped."""
    ledger = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 3)
        if (len(parts) < 3 or parts[1] not in _LEDGER_STATES
                or not _STABLE_RE.match(parts[0])):
            # Fail closed: a line the watch cannot read means the ledger cannot be trusted,
            # so the torch part is UNKNOWN rather than "no rejections".
            raise ValueError(f"malformed ledger line: {line!r}")
        ledger[parts[0]] = {"state": parts[1], "date": parts[2],
                            "evidence": parts[3] if len(parts) > 3 else ""}
    return ledger


def _floor_from_ledger(ledger, torch_floor, floor_window, floor_slack=FLOOR_SLACK):
    """The floor the window rule asks for, as "X.Y", or None: the oldest of
    the newest ``floor_window`` validated minors, proposed only once
    ``floor_window + floor_slack`` validated minors sit above the current
    floor.  A minor with no validated version is never named."""
    validated = sorted({_minor(v) for v, e in ledger.items()
                        if e["state"] == "validated"})
    above = [m for m in validated if m > _minor(torch_floor)]
    if len(above) < floor_window + floor_slack:
        return None
    candidate = validated[-floor_window]
    if candidate <= _minor(torch_floor):
        return None
    return ".".join(str(x) for x in candidate)


def _torch_line(floor, exclusions):
    """The quoted torch requirement for pyproject: the floor, then the
    exclusions in version order."""
    excl = sorted(set(exclusions), key=_vtuple)
    return '"torch>=' + floor + "".join(f",!={v}" for v in excl) + '"'


def divergence(torch_release, torch_list, matrix, python_floor, torch_floor,
               runner_minors=None, ledger=None, exclusions=(),
               floor_window=FLOOR_WINDOW):
    """The divergence, as a dict.  Versions below the Python floor, and
    versions absent from ``runner_minors`` (the versions GitHub's runners
    install), are reported informationally and never proposed.  The torch
    part comes from ``ledger`` (parse_ledger): a floor advance by the window
    rule, and an exclusion for each rejected version at or above the floor
    that ``exclusions`` (pyproject's own) does not already carry.  With no
    ledger the torch part is unknown and nothing torch-related is proposed."""
    below_floor = [v for v in torch_list if _minor(v) < _minor(python_floor)]
    eligible = [v for v in torch_list if _minor(v) >= _minor(python_floor)]
    additions = [v for v in eligible if v not in matrix]
    not_on_runners = []
    if runner_minors is not None:
        not_on_runners = [v for v in additions if v not in runner_minors]
        additions = [v for v in additions if v in runner_minors]
    removals = [v for v in matrix if v not in torch_list]
    ledger_known = ledger is not None
    torch_advance, torch_exclusions, torch_rejected, newest_validated = None, [], {}, None
    if ledger_known:
        torch_advance = _floor_from_ledger(ledger, torch_floor, floor_window)
        validated = [v for v, e in ledger.items() if e["state"] == "validated"]
        if validated:
            newest_validated = max(validated, key=_vtuple)
        torch_exclusions = sorted(
            (v for v, e in ledger.items()
             if e["state"] == "rejected" and _minor(v) >= _minor(torch_floor)
             and v not in exclusions),
            key=_vtuple)
        torch_rejected = {v: ledger[v] for v in torch_exclusions}
    return {
        "torch_release": torch_release,
        "torch_list": torch_list,
        "matrix": matrix,
        "python_floor": python_floor,
        "torch_floor": torch_floor,
        "below_floor": below_floor,
        "not_on_runners": not_on_runners,
        "additions": additions,
        "removals": removals,
        "torch_advance": torch_advance,
        "torch_exclusions": torch_exclusions,
        "torch_rejected": torch_rejected,
        "ledger_known": ledger_known,
        "newest_validated": newest_validated,
        "any": bool(additions or removals or torch_advance or torch_exclusions),
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
    if d.get("torch_exclusions"):
        parts.append("exclude-" + "-".join(d["torch_exclusions"]))
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
    if d["torch_advance"] or d.get("torch_exclusions"):
        floor = d["torch_advance"] or d["torch_floor"]
        # Exclusions below the (new) floor are dead text and go; the rejected ones join.
        keep = [v for v in parse_torch_exclusions(pyproject_text)
                if _minor(v) >= _minor(floor)]
        new_pyproject = re.sub(r'"torch\s*>=\s*[\d.]+[^"]*"',
                               _torch_line(floor, keep + list(d.get("torch_exclusions", []))),
                               new_pyproject)
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
    if d.get("torch_exclusions"):
        title_parts.append(f"exclude torch {_join(d['torch_exclusions'])}")
    title = "; ".join(title_parts)
    title = title[0].upper() + title[1:]

    body = ["Opened by the dependency watch (plans repo: "
            "plans/torch_port/closed/python_matrix_nightly_check.md; the torch "
            "policy: plans/mbirtorch_metrics/torch_version_policy_plan.md).",
            "",
            f"Trigger: torch {d['torch_release']} publishes CPU-index Linux "
            f"x86_64 wheels for Python {_join(d['torch_list'])}.",
            "",
            "Policy: mbirtorch tests the Python versions torch supports, "
            "validates every torch release through the cluster nightly's "
            "canary, advances its torch floor by the "
            f"{FLOOR_WINDOW}-validated-minor window rule, and excludes only "
            "versions the canary rejected."]
    if base_sha:
        body += ["", f"Branch cut from `{base_sha}`."]
    if d["removals"]:
        body += ["", "This includes a REMOVAL.  CI cannot prove a removal "
                     "correct; the reviewer must judge it."]
    if d["torch_advance"]:
        body += ["", f"The floor advances because the ledger holds more than "
                     f"{FLOOR_WINDOW} validated minors (newest validated: "
                     f"{d.get('newest_validated') or '?'}); exclusions below "
                     f"{d['torch_advance']} are dropped.  The green checks cover "
                     "the CPU suite, including the floor job; the Triton and "
                     "CUDA paths were validated by the cluster nightly's canary "
                     "(the ledger's evidence lines)."]
    for v in d.get("torch_exclusions", []):
        e = d.get("torch_rejected", {}).get(v, {})
        body += ["", f"torch {v} was rejected by the cluster nightly's canary on "
                     f"{e.get('date', '?')}: {e.get('evidence', '')}.  The "
                     "exclusion keeps a fresh install off it; it is lifted only "
                     "by the floor passing it."]
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
    ap.add_argument("--ledger-file", default=None,
                    help="read the torch ledger from this file instead of "
                         "the metrics repo on GitHub (tests, dry runs)")
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
        runner_minors = parse_runner_manifest(fetch(RUNNER_MANIFEST_URL))
    except (OSError, urllib.error.URLError, ValueError) as e:
        print(f"dependency-watch: RUNNER MANIFEST NOT READ "
              f"({RUNNER_MANIFEST_URL}): {e}")
        print("dependency-watch: verdict UNKNOWN (cannot tell which versions "
              "the runners install; this is not 'no divergence')")
        return 1

    try:
        matrix, docs_version = parse_version_file(read(vf_source))
    except (OSError, urllib.error.URLError) as e:
        print(f"dependency-watch: VERSION FILE NOT READ ({vf_source}): {e}")
        print("dependency-watch: verdict UNKNOWN (no matrix; this is not "
              "'no divergence')")
        return 1
    print(f"dependency-watch: matrix {matrix}, docs {docs_version} ({vf_source})")

    python_floor, torch_floor = parse_pyproject(read(pp_source))
    exclusions = parse_torch_exclusions(read(pp_source))

    # The ledger is read best-effort: without it the Python part still runs, the torch part is
    # reported UNKNOWN, and nothing torch-related is proposed.
    ledger = None
    ledger_source = args.ledger_file or LEDGER_URL
    try:
        ledger = parse_ledger(open(args.ledger_file).read() if args.ledger_file
                              else fetch(LEDGER_URL))
        validated = [v for v, e in ledger.items() if e["state"] == "validated"]
        print(f"dependency-watch: torch ledger {ledger_source}: {len(ledger)} versions, "
              f"newest validated {max(validated, key=_vtuple) if validated else 'none'}")
    except (OSError, urllib.error.URLError, ValueError) as e:
        print(f"dependency-watch: LEDGER NOT READ ({ledger_source}): {e}")
        print("dependency-watch: the torch floor and exclusions are UNCHECKED "
              "tonight (this is not 'no divergence')")
    d = divergence(torch_release, torch_list, matrix, python_floor, torch_floor,
                   runner_minors=runner_minors, ledger=ledger, exclusions=exclusions)

    if args.json:
        print(json.dumps(d, indent=2))
    if d["below_floor"]:
        print(f"dependency-watch: below the {python_floor} floor, not proposed: "
              f"{d['below_floor']}")
    if d["not_on_runners"]:
        print(f"dependency-watch: torch supports but GitHub runners do not "
              f"install yet, not proposed: {d['not_on_runners']}")
    if d["any"]:
        print(f"dependency-watch: DIVERGENCE -> branch {branch_name(d)}")
        if d["additions"]:
            print(f"dependency-watch:   additions due: {d['additions']}")
        if d["removals"]:
            print(f"dependency-watch:   removals due: {d['removals']}")
        if d["torch_advance"]:
            print(f"dependency-watch:   torch floor advance due: "
                  f"{torch_floor} -> {d['torch_advance']}")
        if d["torch_exclusions"]:
            print(f"dependency-watch:   torch exclusions due (rejected by the "
                  f"canary): {d['torch_exclusions']}")
    elif not d["ledger_known"]:
        print("dependency-watch: verdict UNKNOWN (the torch ledger was not read; "
              "the Python matrix matches, the torch floor and exclusions were "
              "not checked)")
    else:
        print("dependency-watch: verdict none (matrix and floors match torch)")

    status = None
    new_release = False
    if args.state:
        status = consecutive_night_status(args.state, branch_name(d))
        print(f"dependency-watch: two-night status: {status}")
        # A newest stable torch that differs from the last run's is a new release: the workflow
        # then dispatches CI on prerelease so the CPU suite runs on it tonight.  An empty previous
        # value (first run, or a cache miss) is not a new release.
        rel_path = os.path.join(os.path.dirname(args.state) or ".", "last-release")
        previous_release = ""
        if os.path.exists(rel_path):
            with open(rel_path) as f:
                previous_release = f.read().strip()
        with open(rel_path, "w") as f:
            f.write(torch_release)
        new_release = bool(previous_release) and previous_release != torch_release
        print(f"dependency-watch: torch release {torch_release}: "
              + ("NEW since " + previous_release if new_release
                 else "seen before" if previous_release
                 else "first sighting (no CI dispatch)"))
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
        _emit_output("new_release", "true" if new_release else "false")
    return 0


def _emit_output(name, value):
    """One output for the workflow's later steps (GitHub's output file)."""
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            f.write(f"{name}={value}\n")


if __name__ == "__main__":
    sys.exit(main())
