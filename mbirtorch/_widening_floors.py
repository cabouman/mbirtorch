"""Per-geometry, per-count SPEED floors for the automatic device count.

The automatic device-count policy
(:meth:`TomographyModel._apply_device_policy`) picks the widest device count
whose modeled peak memory fits.  Fitting is not the same as being faster:
widening a small problem was measured 13x slower on parallel beam and 5.1x
slower on a sparse-view shape.  A floor is the problem size, in SINOGRAM
ELEMENTS (``prod(sinogram_shape)``), at or above which a device count is
worth using.  Below its floor a count is not admitted by the automatic path
-- not removed from the search, only pushed behind every admitted count, so
capacity still wins when nothing admitted fits.  A count of 1 is always
admitted.

Sinogram elements is the size metric because the decision site knows the
sinogram shape before any array is placed, and because measurement chose it:
on a sparse-view problem whose sinogram elements and recon voxels point at
different sizes, widening to two devices was a 1.87x regression, which is
what the sinogram count predicts.

Each count's floor is the size at which it overtakes the best SMALLER
ADMITTED count, not where it overtakes one device: the parallel n=4 floor is
where n=4 overtakes n=2.  Where the measurement brackets the crossing point
between two tested sizes, the floor takes the larger, because the risk is
lopsided -- widening too early has cost multiples, while holding a
just-large-enough problem at a smaller count costs a few percent.

``elements=None`` -- a "sentinel" row, as the refusal message and the tests
call it -- records a count with no admission size at or below the largest
size tested.  It is not a permanent refusal: it excludes that count
everywhere until a refresh finds an admission size, and it carries
``largest_tested`` so the refresh knows where to start.  Both denoiser rows
are in that state today: sharded denoising lost at every size probed, so the
automatic path holds a denoiser at one device and only capacity widens it.
A count with no row inherits the row of the next MEASURED count
above it (n=3 is governed by the n=4 floor, as is any count above 4), and a
model declaring no ``_floor_family`` gets the parallel floors, the more
permissive measured set, with the reason string and the verbose-2 log both
saying so.

The nightly runs validate none of this: every one of them, n=1 included, pins
the count through ``MBIRTORCH_NUM_DEVICES``, and a pin bypasses the guard by
construction.  The chosen-count tests in ``tests/test_device_policy.py`` are
the standing coverage -- the only place the ordering rule runs end to end.

MAINTENANCE.  These are measurements, only as good as the projection code
they were taken against, and a table that no longer matches that code still
governs and still runs.  The first guard consultation in a process hashes the
projection-cost inputs against :data:`BLESSED_COST_HASHES`, and
:func:`stale_note` names whatever moved on every automatic device selection
until it stops moving; ``tests/test_widening_floors.py`` prints the same
report, warns, and PASSES, because out-of-date numbers are a reason to
re-measure rather than a reason to stop a reconstruction or a test run.  Two
things do fail hard: :data:`TABLE_CHECKSUM` binds the floors, the recorded
hashes and :data:`STALE_SINCE` into one unit, so hand-editing a hash to
silence the note is caught, and the provenance checks (dates, brackets,
floors rising with the count) are assertions.
``dev_scripts/refresh_widening_floors.py`` is the SOLE writer of those three
constants, and pasting its output is the one thing that clears the note.  If
code that determines projection cost MOVES (a new driver function in
``tomography_model``, batching logic moving to a new file), add it to
:data:`COST_INPUT_FILES` or :data:`COST_INPUT_METHODS` in the same change and
re-record the hashes: the check covers only what it names.

Set the environment variable named in :data:`GUARD_ENV_VAR` to ``0`` to turn
the guard off and restore the pure capacity order.  An explicit
``configure_devices`` call and ``MBIRTORCH_NUM_DEVICES`` bypass it anyway: a
count the caller named is not the library's to second-guess.
"""

import hashlib
import inspect
import os
from collections import namedtuple

#: Set to '0' (or 'false'/'no'/'off') to disable the speed guard entirely.
GUARD_ENV_VAR = 'MBIRTORCH_WIDENING_GUARD'
_GUARD_OFF_VALUES = ('0', 'false', 'no', 'off')

#: The family a model with no ``_floor_family`` is governed by.
DEFAULT_FAMILY = 'parallel'

#: The hardware and the run protocol every row below was measured under.
MEASURED_GPU = 'NVIDIA H100 80GB HBM3 (4 per node)'
MEASURED_CONFIG = ('warm median of 3 seeded 3-iteration VCD recons, cold '
                   'pass discarded, package-default subset schedule, Triton '
                   'kernels on, torch.compile auto')

#: These floors are validated at MEASURED_CONFIG only.  A different iteration
#: count or a different subset schedule moves the per-subset host-sync cost
#: that sets these crossovers, so it falls outside what was measured: the
#: floors are still applied there, but they were not validated there.  A
#: workload that runs at a different configuration should re-measure rather
#: than assume.
MEASUREMENT_CAVEAT = ('validated at MEASURED_CONFIG only; a different '
                      'iteration count or subset schedule is outside the '
                      'measured envelope')

#: Where a crossover was pinned down: the largest problem size at which the
#: count under test LOST, the smallest at which it WON, and the speedups
#: measured there against the comparison count.  A side is None when no such
#: size was measured; ``winning_cell is None`` is what makes ``elements``
#: None.  A "cell" here is one measured (views, rows, channels) sinogram
#: shape.
Bracket = namedtuple('Bracket', ('losing_cell', 'losing_speedup',
                                 'winning_cell', 'winning_speedup'))

#: One row of the table.  ``elements`` is the floor in sinogram elements, or
#: None when no admission size was measured.  ``cell`` is the (views, rows,
#: channels) sinogram shape the floor was read off; ``against`` is the count
#: the crossover was taken against (the best smaller ADMITTED count);
#: ``spread`` is the widest spread among the warm repeats the row was read
#: from, which is the run-to-run noise the crossover had to clear.
Floor = namedtuple('Floor', ('family', 'count', 'elements', 'cell', 'against',
                             'bracket', 'spread', 'gpu', 'config', 'measured',
                             'commit', 'largest_tested', 'note'))

# ── the measured table ───────────────────────────────────────────────────────
# The measured shapes, each named for its view count.  For the projection
# families these are sinogram shapes; for the denoiser the same tuples are
# IMAGE shapes, because its sinogram shape is its image shape.
#     384-class  (384, 336, 288)     =    37,158,912 elements
#     512-class  (512, 448, 384)     =    88,080,384 elements
#     768-class  (768, 672, 576)     =   297,271,296 elements
#    1024-class  (1024, 1008, 992)   = 1,023,934,464 elements
FLOORS = {
    ('parallel', 2): Floor(
        family='parallel', count=2, elements=88_080_384, cell=(512, 448, 384),
        against=1,
        bracket=Bracket(losing_cell=(384, 336, 288), losing_speedup=0.74,
                        winning_cell=(512, 448, 384), winning_speedup=1.26),
        spread=0.01794, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-16', commit='dba9652',
        largest_tested=297_271_296,
        note='refreshed on the merged, unpadded-split tree (mg16, job '
             '15304592) and unchanged: 0.74x losing and 1.26x winning, '
             'the same bracket as 2026-08-13 to within the run spread'),
    ('parallel', 4): Floor(
        family='parallel', count=4, elements=297_271_296,
        cell=(768, 672, 576), against=2,
        bracket=Bracket(losing_cell=(512, 448, 384), losing_speedup=0.68,
                        winning_cell=(768, 672, 576), winning_speedup=1.02),
        spread=0.0111, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-16', commit='dba9652',
        largest_tested=1_023_934_464,
        note='the tightest admission in the table again: 1.02x at the '
             '768-class against a 1.1 percent spread, with the 512-class '
             'losing at 0.68x.  Still the row to watch on the next '
             'refresh'),
    ('cone', 2): Floor(
        family='cone', count=2, elements=88_080_384, cell=(512, 448, 384),
        against=1,
        bracket=Bracket(losing_cell=(384, 336, 288), losing_speedup=0.80,
                        winning_cell=(512, 448, 384), winning_speedup=1.29),
        spread=0.05138, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-16', commit='dba9652',
        largest_tested=297_271_296,
        note='unchanged by the mg16 refresh: the same floor with nearly '
             'the same bracket as 2026-08-13 (0.80x losing, 1.29x '
             'winning), so neither the prerelease merge nor the pad '
             'removal moved this crossover'),
    ('cone', 4): Floor(
        family='cone', count=4, elements=1_023_934_464,
        cell=(1024, 1008, 992), against=2,
        bracket=Bracket(losing_cell=(768, 672, 576), losing_speedup=0.96,
                        winning_cell=(1024, 1008, 992), winning_speedup=1.71),
        spread=0.04207, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-16', commit='dba9652',
        largest_tested=1_023_934_464,
        note='unchanged at the top of the ladder, margin 1.71x, with the '
             '768-class still just under admission at 0.96x.  A refresh '
             'can confirm this floor but cannot lower it without a '
             'larger cell'),
    ('denoiser', 2): Floor(
        family='denoiser', count=2, elements=None, cell=None,
        against=1,
        bracket=Bracket(losing_cell=(1024, 1008, 992), losing_speedup=0.65,
                        winning_cell=None, winning_speedup=None),
        spread=0.02309, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-16', commit='dba9652',
        largest_tested=1_023_934_464,
        note='first measurement of the family, and a sentinel: sharded '
             'denoising LOST at every probe cell, 0.58x at the 512-class '
             'rising to 0.65x at the 1024-class.  A denoise sweep is '
             'seconds long and its per-subset host syncs dominate, so a '
             'split has nothing to amortize at these sizes.  Read in '
             'IMAGE VOXELS: the denoiser sinogram shape is its image '
             'shape.  The timed call is denoise with sigma_noise '
             'supplied, not recon, at the shared warm-median protocol.  '
             'The ratio rises with size, so a larger ladder may yet find '
             'admission; capacity still widens a denoise that cannot fit '
             'one device'),
    ('denoiser', 4): Floor(
        family='denoiser', count=4, elements=None, cell=None,
        against=1,
        bracket=Bracket(losing_cell=(1024, 1008, 992), losing_speedup=0.59,
                        winning_cell=None, winning_speedup=None),
        spread=0.02309, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-16', commit='dba9652',
        largest_tested=1_023_934_464,
        note='a sentinel like n=2, losing 0.44x to 0.59x across the '
             'probes, measured against n=1 because no smaller denoiser '
             'count is admitted.  Read in image voxels; the n=2 note '
             'records the mechanism'),
}

# ── the projection-cost inputs the floors were measured against ──────────────
#: Files whose contents price a projection.  Hashed WHOLE, deliberately: the
#: module-level chunk constants and the budget class attributes these files
#: carry are exactly the kind of tuning that moves a crossover without
#: touching any function this table names, so a function-level hash would
#: miss them.  ``_sharding.py`` is here because it holds the cross-device
#: transfer primitives the multi-device drivers are built from, and how much
#: those move is most of what a wider device count costs.
COST_INPUT_FILES = ('triton_parallel.py', 'triton_cone.py', 'projectors.py',
                    '_sharding.py')

#: Methods of TomographyModel that drive the multi-device projections.  The
#: rest of that module moves for reasons unrelated to projection cost, so the
#: hash is taken over these sources rather than the whole file.  The column
#: gather is a third driver rather than a branch of the first, so it is named
#: here in its own right; leaving it out would let the pixel batch it walks
#: change without anything noticing.
COST_INPUT_METHODS = ('_sparse_forward_project_sharded',
                      '_sparse_forward_project_columns',
                      '_sparse_back_project_sharded')

#: sha256 of each cost input as of the measurement above -- the recorded
#: ("blessed") values the live check compares against.  Re-record them with
#: ``python dev_scripts/refresh_widening_floors.py --bless``.
BLESSED_COST_HASHES = {
    'TomographyModel._sparse_back_project_sharded':
        'f73fa28f32d2393fac3f06139d8ddeccc55260fceb28bbca7f102334839fb5a4',
    'TomographyModel._sparse_forward_project_columns':
        '70a2e2156a6b64cd44cb7b7a061ae514d3b43ef2e0580d55e503d1b7fb94f298',
    'TomographyModel._sparse_forward_project_sharded':
        '9b75c174555688a3fedba931a22de7dcf6236827fde3b5dc8ca27820728cd8fc',
    '_sharding.py':
        '3afe5c538dade99efd39fbeeaf32fd688a9fb9672a095014ec7789f9e5af9077',
    'projectors.py':
        '2c2c44fe308df095aa962febc42101673394c997f0f488da95adea1c20300c6b',
    'triton_cone.py':
        '36e72e6f0f6f3ca768c52d7b92715f75e7ad005372f2cdad50fc0bf60e003411',
    'triton_parallel.py':
        '78fc531463124e634950c97bb3ebb40fd69bc9cce1e42073a4b7345ba50ca9f1',
}

#: A hand-written staleness date, or None -- currently None, from the
#: 2026-08-10 refresh run.  A refresh ALWAYS writes None here, because
#: :func:`stale_note` detects a changed cost input by itself and a
#: hand-written date adds nothing.  The field stays because
#: :data:`TABLE_CHECKSUM` binds it, and because someone may still want to
#: record a reason to re-measure that no hash can see -- new hardware, a
#: changed subset schedule.
STALE_SINCE = None

#: sha256 binding FLOORS, BLESSED_COST_HASHES and STALE_SINCE together.
#: These three move as one unit or not at all -- editing a hash by hand to
#: green the test leaves this behind, and the test says so.  Recomputed and
#: printed by ``refresh_widening_floors.py --bless``.
TABLE_CHECKSUM = \
    '091bd6255de989c35ec429e81bb4a881e20d90c216103d9592ca4863c38bea0d'


# ── the env knob ─────────────────────────────────────────────────────────────
def guard_enabled():
    """Whether the speed guard is consulted.  Read per call, like the other
    environment knobs, so a test or a script can turn it off around one
    block without reimporting anything."""
    return os.environ.get(GUARD_ENV_VAR, '').strip().lower() \
        not in _GUARD_OFF_VALUES


# ── staleness, detected rather than declared ─────────────────────────────────
#: The live check for changed cost inputs, computed once per process on the
#: first guard consultation and cached here: ``(changed_names, failure)``.
#: None means it has not run yet.  A test that changes what the check would
#: see resets this to None first.
_DRIFT_CHECK = None

#: The one command that clears a staleness note, named in every note that
#: reports one.
_REFRESH_COMMAND = 'dev_scripts/refresh_widening_floors.py'


def _drift_check():
    """The cost inputs that no longer hash to their recorded values, as
    ``(names, failure)``, computed once per process.

    Hashing three files and two method sources is milliseconds, but the guard
    is consulted on every automatic device selection, so it is done once and
    cached.  A check that CANNOT run -- a file moved mid-refactor, a method
    renamed out from under :data:`COST_INPUT_METHODS` -- returns its failure
    for the log instead of raising: saying the measurements need refreshing
    must never be able to break a reconstruction, which is the whole reason
    the note replaced a hard failure.
    """
    global _DRIFT_CHECK
    if _DRIFT_CHECK is None:
        try:
            _DRIFT_CHECK = (tuple(name for name, _want, _got
                                  in stale_cost_inputs()), None)
        except Exception as exc:                                  # noqa: BLE001
            _DRIFT_CHECK = ((), '{}: {}'.format(type(exc).__name__, exc))
    return _DRIFT_CHECK


def stale_note():
    """The one-line notice for the device-selection log that the floors need
    re-measuring, or None.

    Either of two things puts a note here, and they are reported together
    when both hold:

      * A projection-cost input no longer matches its recorded hash, so the
        floors were measured against code that has since changed.  This is
        the ordinary path and it needs no human in it: the note names the
        inputs that moved.
      * :data:`STALE_SINCE` -- a hand-written date, kept for reasons to
        re-measure that no hash can see.

    A check that could not run says so rather than implying freshness.  The
    note is advisory in every case: the floors still govern, and a real
    refresh is what clears it.
    """
    drifted, failure = _drift_check()
    parts = []
    if failure is not None:
        parts.append('the widening speed floors could not be checked against '
                     'the projection-cost code ({}), so whether they are '
                     'stale is unknown'.format(failure))
    elif drifted:
        parts.append('the widening speed floors were measured against '
                     'projection-cost code that has since changed ({}), so '
                     'they may no longer describe it'.format(
                         ', '.join(drifted)))
    if STALE_SINCE is not None:
        parts.append('the widening speed floors are marked stale since {}: '
                     'the floors were not re-measured when they were '
                     'recorded as owing a measurement'.format(STALE_SINCE))
    if not parts:
        return None
    return '{}; re-measuring with {} clears this'.format('; '.join(parts),
                                                         _REFRESH_COMMAND)


# ── the metric ───────────────────────────────────────────────────────────────
def sinogram_elements(sinogram_shape):
    """``prod(sinogram_shape)`` -- the floors' metric, as an int."""
    total = 1
    for extent in sinogram_shape:
        total *= int(extent)
    return int(total)


def _millions(elements):
    return '{:.1f}M'.format(elements / 1e6)


# ── the lookup ───────────────────────────────────────────────────────────────
def families():
    """Every family the table has rows for."""
    return sorted({family for family, _count in FLOORS})


def _measured_counts(family):
    return sorted(count for fam, count in FLOORS if fam == family)


def governing_floor(family, count):
    """The table row that governs ``count`` in ``family``, or None when the
    family has no measured rows at all.

    A count with no row of its own inherits the next MEASURED count ABOVE it
    -- the conservative direction, since floors rise with the count.  A count
    above every measured one inherits the largest measured row for the same
    reason.
    """
    counts = _measured_counts(family)
    if not counts:
        return None
    above = [c for c in counts if c >= count]
    return FLOORS[(family, above[0] if above else counts[-1])]


def admitted(family, count, sino_elements):
    """Whether the AUTOMATIC path may widen to ``count`` devices at this
    problem size, and the sentence explaining the verdict.

    Args:
        family (str or None): ``'parallel'``, ``'cone'``, or None for a model
            that declares no ``_floor_family``; None takes the parallel
            floors, which are the more permissive measured set.
        count (int): the device count under test.
        sino_elements (int): ``prod(sinogram_shape)``.

    Returns:
        (bool, str): admitted, and the reason -- phrased for the run log's
        device line when it is a refusal.
    """
    resolved = DEFAULT_FAMILY if family is None else family
    substituted = ('' if family is not None else
                   ' (this model names no _floor_family, so the {} floors '
                   'apply)'.format(DEFAULT_FAMILY))
    size = _millions(sino_elements)

    if count <= 1:
        return True, 'a single device is always admitted'

    floor = governing_floor(resolved, count)
    if floor is None:
        return True, ('no speed floors are measured for the {} family, so '
                      'every count is admitted'.format(resolved))

    inherited = ('' if floor.count == count else
                 ', which n={} inherits'.format(count))
    named = 'the {} n={} floor{}'.format(resolved, floor.count, inherited)

    if floor.elements is None:
        return False, (
            'held by the speed floor: {} is a sentinel -- no admission point '
            'has been measured at or below {} sinogram elements, the largest '
            'size tested; configure_devices(num_devices={}) '
            'overrides'.format(named, _millions(floor.largest_tested), count)
            + substituted)
    if sino_elements >= floor.elements:
        return True, ('{} sinogram elements >= {} ({})'.format(
            size, _millions(floor.elements), named) + substituted)
    return False, (
        'held by the speed floor: {} sinogram elements < {} ({}); '
        'configure_devices(num_devices={}) overrides'.format(
            size, _millions(floor.elements), named, count) + substituted)


def fallback_reason(family, count, sino_elements):
    """The note for a count the capacity search settles on DESPITE its floor.

    An excluded count is only ever reached after every admitted count has
    been refused, so choosing one is capacity overriding speed.  The run log
    says which happened rather than leaving the count looking like a plain
    rejection.
    """
    resolved = DEFAULT_FAMILY if family is None else family
    floor = governing_floor(resolved, count)
    if floor is None or floor.elements is None:
        detail = ('no admission point is measured for {} n={} at or below {} '
                  'sinogram elements'.format(
                      resolved, count,
                      _millions(floor.largest_tested) if floor else 0))
    else:
        detail = '{} sinogram elements < {}'.format(
            _millions(sino_elements), _millions(floor.elements))
    return ('chosen past its speed floor because no admitted count fits: '
            '{} (the {} n={} floor)'.format(detail, resolved, floor.count
                                            if floor else count))


# ── the table's own invariants ───────────────────────────────────────────────
def monotone_violations():
    """Families whose FINITE floors fall as the count rises.

    A higher count doing MORE work per device than a lower one at the same
    size is not something the measurements have ever shown, and it would make
    the two-group ordering incoherent, so the table asserts against it.
    """
    bad = []
    for family in families():
        finite = [(count, FLOORS[(family, count)].elements)
                  for count in _measured_counts(family)
                  if FLOORS[(family, count)].elements is not None]
        for (low, low_floor), (high, high_floor) in zip(finite, finite[1:]):
            if high_floor < low_floor:
                bad.append('{}: n={} floor {} < n={} floor {}'.format(
                    family, high, high_floor, low, low_floor))
    return bad


assert not monotone_violations(), \
    'the widening floors are not monotone in count: ' \
    + '; '.join(monotone_violations())


# ── the cost-input hashes and the three-way binding ──────────────────────────
def cost_input_hashes():
    """sha256 of every projection-cost input, keyed as in
    :data:`BLESSED_COST_HASHES`.

    The model class is imported lazily: this module is imported BY
    ``tomography_model``, so a module-level import would be circular.
    """
    from .tomography_model import TomographyModel

    here = os.path.dirname(os.path.abspath(__file__))
    digests = {}
    for name in COST_INPUT_FILES:
        with open(os.path.join(here, name), 'rb') as handle:
            digests[name] = hashlib.sha256(handle.read()).hexdigest()
    for name in COST_INPUT_METHODS:
        source = inspect.getsource(getattr(TomographyModel, name))
        digests['TomographyModel.' + name] = hashlib.sha256(
            source.encode('utf-8')).hexdigest()
    return digests


def stale_cost_inputs():
    """The cost inputs whose hash no longer matches the recorded one, as
    ``[(name, recorded, actual), ...]``.  Empty means the floors still
    describe the code they were measured against."""
    actual = cost_input_hashes()
    stale = []
    for name in sorted(set(actual) | set(BLESSED_COST_HASHES)):
        want = BLESSED_COST_HASHES.get(name)
        got = actual.get(name)
        if want != got:
            stale.append((name, want, got))
    return stale


def table_checksum():
    """sha256 over the three things a refresh writes together: the floors,
    the recorded cost-input hashes, and the staleness date.

    Hand-editing one of them -- the cheap way to green the hash test -- moves
    this and fails a different assertion, which is the point.
    """
    payload = '\n'.join([repr(sorted(FLOORS.items())),
                         repr(sorted(BLESSED_COST_HASHES.items())),
                         repr(STALE_SINCE)])
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def bless_lines(stale_since=None):
    """The paste-ready replacement for the three bound constants.

    ``stale_since`` is None for every refresh: a changed cost input is
    detected live, so a re-record that follows a real measurement leaves
    :data:`STALE_SINCE` unset and the note clears itself.  The parameter
    remains for a human recording a reason to re-measure that no hash can
    see.
    """
    hashes = cost_input_hashes()
    lines = ['BLESSED_COST_HASHES = {']
    for name, digest in sorted(hashes.items()):
        lines.append("    '{}':".format(name))
        lines.append("        '{}',".format(digest))
    lines.append('}')
    lines.append('')
    lines.append('STALE_SINCE = {!r}'.format(stale_since))
    lines.append('')
    payload = '\n'.join([repr(sorted(FLOORS.items())),
                         repr(sorted(hashes.items())),
                         repr(stale_since)])
    lines.append("TABLE_CHECKSUM = '{}'".format(
        hashlib.sha256(payload.encode('utf-8')).hexdigest()))
    return '\n'.join(lines)
