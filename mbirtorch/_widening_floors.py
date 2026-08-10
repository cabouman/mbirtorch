"""Per-geometry, per-count SPEED floors for the automatic device count.

WHAT A FLOOR IS.  The automatic device-count policy
(:meth:`TomographyModel._apply_device_policy`) is a CAPACITY rule: it walks
the visible counts largest-first and takes the first one whose modeled peak
fits.  Capacity alone widens small problems onto counts that run SLOWER --
measured harm of 13x at a 128-class cell and 5.1x at a sparse-view shape.  A
floor is the problem size, in SINOGRAM ELEMENTS (``prod(sinogram_shape)``),
at or above which a device count is worth using at all.  Below its floor a
count is not admitted by the AUTOMATIC path; it is never removed from the
search, only pushed behind every admitted count, so capacity still wins when
nothing admitted fits.

The metric is sinogram elements because that is what the decision site
already knows before any array is placed, and because the measurement chose
it: a sparse-view probe whose sinogram elements and recon voxels disagree by
a full ladder step landed on the side the sinogram metric predicted, where
widening to two devices was a 1.87x regression.

THE CROSSOVER RULE THAT SETS A FLOOR.  Each count's floor is its crossover
against the best SMALLER ADMITTED count -- not against n=1 unconditionally.
So the parallel n=4 floor is where n=4 overtakes n=2 (n=2 being admitted
below it), while the cone n=4 floor is where n=4 overtakes n=1, because cone
n=2 is never admitted and so is never the count n=4 has to beat.  Where the
measurement brackets a crossover between two ladder sizes, the floor is set
at the CONSERVATIVE end -- the larger size -- because the measured asymmetry
is lopsided: widening a below-knee problem cost multiples, while holding a
just-above-knee problem at a smaller count costs a few percent.

SENTINEL ENTRIES.  ``elements=None`` means no admission point was found at
or below the largest size measured.  It is NOT a hard "never": it excludes
the count at every size until a refresh measures an admission point, and it
carries ``largest_tested`` so the refresh knows where to start probing.

COUNTS WITH NO ENTRY inherit the entry of the next MEASURED count above
them, which is the conservative direction: n=3 is governed by the n=4 floor,
and so is any count above 4.  A count of 1 is always admitted.

FAMILY.  A model names its family with the class attribute ``_floor_family``
(``'parallel'``, ``'cone'``).  ``None`` -- the base-class default -- means
the parallel floors apply, since they are the more permissive measured set;
the accessor's reason string says so, and the selection path logs the
substitution at verbose 2, so a geometry that was never measured never has
the fact hidden from it.

WHAT VALIDATES THESE NUMBERS.  Not the nightly, and not the measurement
campaigns: every nightly row, n=1 included, is env-pinned through
``MBIRTORCH_NUM_DEVICES``, and a pin bypasses this guard by construction.
The chosen-count unit tests in ``tests/test_device_policy.py`` are therefore
the guard's standing regression coverage -- they are the only place the
ordering rule is exercised end to end.

MAINTENANCE.  These numbers are measurements, not constants, and they are
only as good as the projection code they were measured against.

  * ``dev_scripts/refresh_widening_floors.py`` re-measures them on a 4-GPU
    node and prints a paste-ready replacement for :data:`FLOORS`, provenance
    included.  It is the SOLE writer of the three things that must move
    together: the floors, their provenance, and the blessed hashes below.
    :data:`TABLE_CHECKSUM` binds all three, so greening the test by hand
    editing a hash fails a different assertion instead.
  * ``tests/test_widening_floors.py`` FAILS as soon as the projection-cost
    code changes -- the Triton kernels, the projector drivers, or the two
    sharded projection methods -- and keeps failing until the floors are
    re-measured or the hashes are re-blessed.
  * ``refresh_widening_floors.py --bless --accept-stale`` re-blesses the
    hashes WITHOUT re-measuring, and stamps :data:`STALE_SINCE`.  The test
    passes in that state; the device-selection log carries the debt instead.

The escape hatch is the environment variable named in :data:`GUARD_ENV_VAR`:
set it to ``0`` and the guard is consulted nowhere, restoring the pure
capacity order.  Both pin mechanisms -- an explicit ``configure_devices``
call and ``MBIRTORCH_NUM_DEVICES`` -- bypass the guard by construction,
because a count the caller named is not the library's to second-guess.
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

#: THE ENVELOPE CAVEAT.  These floors are validated at MEASURED_CONFIG only.
#: A different iteration count or a different subset schedule moves the
#: per-subset host-sync cost that sets these knees, and is outside the
#: measured envelope: the floors are still applied there, but they were not
#: validated there.  A workload that lives at a different configuration
#: should re-measure rather than assume.
MEASUREMENT_CAVEAT = ('validated at MEASURED_CONFIG only; a different '
                      'iteration count or subset schedule is outside the '
                      'measured envelope')

#: Where a crossover was pinned down: the largest cell where the count under
#: test LOST, the smallest where it WON, and the speedups measured there
#: against the comparison count.  A side is None when no such cell was
#: measured -- ``winning_cell is None`` is what makes a row a sentinel.
Bracket = namedtuple('Bracket', ('losing_cell', 'losing_speedup',
                                 'winning_cell', 'winning_speedup'))

#: One row of the table.  ``elements`` is the floor in sinogram elements, or
#: None for a sentinel (no admission point measured).  ``cell`` is the
#: (views, rows, channels) shape the floor was read off; ``against`` is the
#: count the crossover was taken against (the best smaller ADMITTED count);
#: ``spread`` is the largest warm-repeat spread among the arms the row was
#: read from, which is the noise the crossover had to clear.
Floor = namedtuple('Floor', ('family', 'count', 'elements', 'cell', 'against',
                             'bracket', 'spread', 'gpu', 'config', 'measured',
                             'commit', 'largest_tested', 'note'))

# ── the measured table ───────────────────────────────────────────────────────
# Sizes named by their ladder class:
#     384-class  (384, 336, 288)     =    37,158,912 sinogram elements
#     512-class  (512, 448, 384)     =    88,080,384 sinogram elements
#     768-class  (768, 672, 576)     =   297,271,296 sinogram elements
#    1024-class  (1024, 1008, 992)   = 1,023,934,464 sinogram elements
FLOORS = {
    ('parallel', 2): Floor(
        family='parallel', count=2, elements=88_080_384, cell=(512, 448, 384),
        against=1,
        bracket=Bracket(losing_cell=(384, 336, 288), losing_speedup=0.63,
                        winning_cell=(512, 448, 384), winning_speedup=1.22),
        spread=0.054, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-09', commit='f7e08da',
        largest_tested=1_023_934_464,
        note='n=2 overtakes n=1 at the 512-class cell and wins at every '
             'larger size measured.  The spread is the 384-class n=1 arm, '
             'the noisiest in the bracket'),
    ('parallel', 4): Floor(
        family='parallel', count=4, elements=1_023_934_464,
        cell=(1024, 1008, 992), against=2,
        bracket=Bracket(losing_cell=(768, 672, 576), losing_speedup=0.74,
                        winning_cell=(1024, 1008, 992), winning_speedup=1.68),
        spread=0.0092, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-09', commit='f7e08da',
        largest_tested=1_023_934_464,
        note='n=4 overtakes n=2 somewhere in the 768-to-1024 bracket; the '
             'floor takes the conservative end, since at the 768-class cell '
             'n=2 still led n=1 by 1.60x against n=4\'s 1.18x'),
    ('cone', 2): Floor(
        family='cone', count=2, elements=None, cell=None, against=1,
        bracket=Bracket(losing_cell=(1024, 1008, 992), losing_speedup=0.92,
                        winning_cell=None, winning_speedup=None),
        spread=0.011, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-09', commit='f7e08da',
        largest_tested=1_023_934_464,
        note='SENTINEL: no admission point at or below the 1024-class cell, '
             'the largest size measured -- it reads 0.55x, 0.66x, 1.02x and '
             '0.92x across the 256-, 384-, 512- and 1024-class cells, so the '
             'one nominal win sits inside its own spread.  Not a permanent '
             'never: a refresh that probes above largest_tested can replace '
             'this row'),
    ('cone', 4): Floor(
        family='cone', count=4, elements=1_023_934_464,
        cell=(1024, 1008, 992), against=1,
        bracket=Bracket(losing_cell=(512, 448, 384), losing_speedup=0.68,
                        winning_cell=(1024, 1008, 992), winning_speedup=1.16),
        spread=0.0064, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-09', commit='f7e08da',
        largest_tested=1_023_934_464,
        note='the comparison count is n=1, not n=2, because cone n=2 is '
             'never admitted.  COARSE BRACKET: the cone ladder has no cell '
             'between the 512- and 1024-class sizes, a 12x jump in sinogram '
             'elements, and the winning cell was measured in a separate job '
             'on a different node, so the floor takes the conservative end'),
}

# ── the projection-cost inputs the floors were measured against ──────────────
#: Files whose contents price a projection.  Hashed WHOLE, deliberately: the
#: module-level chunk constants and the budget class attributes these files
#: carry are exactly the kind of tuning that moves a crossover without
#: touching any function this table names, so a function-level hash would
#: miss them.
COST_INPUT_FILES = ('triton_parallel.py', 'triton_cone.py', 'projectors.py')

#: Methods of TomographyModel that drive the multi-device projections.  The
#: rest of that module moves for reasons unrelated to projection cost, so the
#: hash is taken over these two sources rather than the whole file.
COST_INPUT_METHODS = ('_sparse_forward_project_sharded',
                      '_sparse_back_project_sharded')

#: sha256 of each cost input as of the measurement above.  Re-bless with
#: ``python dev_scripts/refresh_widening_floors.py --bless``.
BLESSED_COST_HASHES = {
    'TomographyModel._sparse_back_project_sharded':
        '8a39fb4d97a9573933520ce780eae5dd2097e5a068caa3ee2178114ba8989772',
    'TomographyModel._sparse_forward_project_sharded':
        'e5cb2e89df6452d998f2558487c3924a404620378bfedda7cb9598be9ed9d7c3',
    'projectors.py':
        '2f0520ee7519550daa639373d5b7bc442a2fa1ccd21166bfe40aaeba551613c8',
    'triton_cone.py':
        '8d3820c2101f8d3fbb7823f2d9b6e6e6253164bd14a2c276d167d9ba0a135154',
    'triton_parallel.py':
        '78fc531463124e634950c97bb3ebb40fd69bc9cce1e42073a4b7345ba50ca9f1',
}

#: The date the hashes were re-blessed WITHOUT a new measurement, or None
#: when the floors and the hashes were written together by a real refresh.
#: A non-None value is acknowledged debt: the tests pass, and every
#: automatic device selection logs that the floors are stale.
#: Stamped 2026-08-10 by the back-loop residency pair: the two stale-partial
#: releases in the banded drivers and the block release in the back view
#: loop.  All three are residency-only -- no arithmetic, no summation order,
#: no chunk constant, no batch size moves -- but the refresh script names "a
#: banded driver" as a cost input whose change moves a crossover, so the debt
#: is recorded rather than blessed away.  A refresh run clears it.
STALE_SINCE = '2026-08-10'

#: sha256 binding FLOORS, BLESSED_COST_HASHES and STALE_SINCE together.
#: These three move as one unit or not at all -- editing a hash by hand to
#: green the test leaves this behind, and the test says so.  Recomputed and
#: printed by ``refresh_widening_floors.py --bless``.
TABLE_CHECKSUM = \
    '9367439054bde4e1d5070f187ca1b842d80891044a61ef89b8f9e2b083f1fdb3'


# ── the env knob ─────────────────────────────────────────────────────────────
def guard_enabled():
    """Whether the speed guard is consulted.  Read per call, like the other
    environment knobs, so a test or a script can turn it off around one
    block without reimporting anything."""
    return os.environ.get(GUARD_ENV_VAR, '').strip().lower() \
        not in _GUARD_OFF_VALUES


def stale_note():
    """The one-line debt notice for the device-selection log, or None."""
    if STALE_SINCE is None:
        return None
    return ('the widening speed floors are marked stale since {}: the '
            'projection-cost hashes were re-blessed without re-measuring '
            '(dev_scripts/refresh_widening_floors.py re-measures '
            'them)'.format(STALE_SINCE))


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
    """The cost inputs whose hash no longer matches the blessed one, as
    ``[(name, blessed, actual), ...]``.  Empty means the floors still
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
    the blessed hashes, and the staleness stamp.

    Hand-editing one of them -- the cheap way to green the hash test -- moves
    this and fails a different assertion, which is the point.
    """
    payload = '\n'.join([repr(sorted(FLOORS.items())),
                         repr(sorted(BLESSED_COST_HASHES.items())),
                         repr(STALE_SINCE)])
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def bless_lines(stale_since=None):
    """The paste-ready replacement for the three bound constants.

    ``stale_since`` is the date stamped by ``--bless --accept-stale``; None
    is a bless that followed a real measurement.
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
