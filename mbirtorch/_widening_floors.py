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

The table is COARSE by rule (ruled 2026-08-19).  A count is admitted at the
smallest measured class where it wins by at least :data:`ADMISSION_MARGIN`,
and a thinner win rounds the floor up one class.  The margin is what
survives a different GPU, a different shape in the same class, and
run-to-run drift; the earlier rule admitted any win wider than the cell's
warm spread, which shipped admissions as thin as 1.02x.  Where parallel and
cone measure the same floor at a count, the table carries one shared row
for the pair rather than two values that agree by coincidence, and the
rows' notes name the sharing (the n=4 rows share today; the n=2 rows split
when the sorted forward kernel moved parallel's crossover).

``elements=None`` -- a "sentinel" row, as the refusal message and the tests
call it -- records a count with no admission SIZE at or below the largest
size tested.  It is not a permanent refusal: it excludes that count
everywhere until a refresh finds an admission size, and it carries
``largest_tested`` so the refresh knows where to start.  Six rows are in
that state today.  In five of them -- both denoiser rows, both translation
rows, and multiaxis n=4 -- splitting lost at every size probed.  The sixth,
multiaxis n=2, WON at one probed size and lost above it: two devices win
only in a window of problem sizes, a floor cannot express a window, and a
floor placed at the window's start would widen the larger sizes into a
measured slowdown.  That row therefore holds the family to one device until
the mechanism is understood (multigpu_findings.md sections 1.22 and 1.25);
its note carries the readings.  In every sentinel case the automatic path
holds those models to fewer devices and only capacity widens them.
A count with no row inherits the row of the next MEASURED count
above it (n=3 is governed by the n=4 floor, as is any count above 4), and a
model declaring no ``_floor_family`` gets the parallel floors, with the
reason string and the verbose-2 log both saying so.

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
constants, and pasting its output is the one thing that clears the note.  The
cost inputs are named PER FAMILY (:data:`FAMILY_COST_INPUTS`), so the note
says which families' floors a change touched, and the refresh tool's
``--families`` mode re-measures exactly those.  If code that determines
projection cost MOVES (a new driver function in ``tomography_model``,
batching logic moving to a new file), add it to :data:`FAMILY_COST_INPUTS`
under every family it prices in the same change and re-record the hashes:
the check covers only what it names.

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

#: The coarse admission rule's margin (ruled 2026-08-19).  A device count's
#: floor is the smallest measured class where it beats the best smaller
#: admitted count by at least this factor; a thinner win rounds the floor up
#: one class.  A win must also clear 1.0 by more than the cell's warm spread,
#: which keeps one noisy cell from moving a floor.  The refresh tool applies
#: both conditions when it reads a floor off a run's speedups.
ADMISSION_MARGIN = 1.15

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
        family='parallel', count=2, elements=297_271_296,
        cell=(768, 672, 576), against=1,
        bracket=Bracket(losing_cell=(512, 448, 384), losing_speedup=0.97,
                        winning_cell=(768, 672, 576), winning_speedup=1.38),
        spread=0.01483, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-19', commit='c761b24',
        largest_tested=297_271_296,
        note='the first scoped refresh (mg40, job 15369689), on the '
             'channel-sorted forward kernel: THE FLOOR MOVED UP one '
             'class.  The sorted kernel made the one-device forward fast '
             'enough that the 512-class win disappeared (1.20x on the '
             'per-tap kernel, 0.97x sorted), and the 768-class wins at '
             '1.38x.  Cone n=2 keeps its 512-class floor, so the n=2 '
             'rows no longer share; the pair still shares the n=4 row'),
    ('parallel', 4): Floor(
        family='parallel', count=4, elements=1_023_934_464,
        cell=(1024, 1008, 992), against=2,
        bracket=Bracket(losing_cell=(768, 672, 576), losing_speedup=0.80,
                        winning_cell=(1024, 1008, 992), winning_speedup=1.32),
        spread=0.01307, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-19', commit='c761b24',
        largest_tested=1_023_934_464,
        note='the first scoped refresh (mg40, job 15369689), on the '
             'channel-sorted forward kernel: the coarse floor holds at '
             'the 1024-class, now by measurement rather than by '
             'rounding.  mg26 read a thin 1.06x win at the 768-class and '
             'the margin rule rounded it up; on the sorted kernel that '
             'cell LOSES outright at 0.80x, so the rounding anticipated '
             'exactly this kernel change.  The 1024-class wins at 1.32x.  '
             'Shared row with cone n=4'),
    ('cone', 2): Floor(
        family='cone', count=2, elements=88_080_384, cell=(512, 448, 384),
        against=1,
        bracket=Bracket(losing_cell=(384, 336, 288), losing_speedup=0.87,
                        winning_cell=(512, 448, 384), winning_speedup=1.34),
        spread=0.03913, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-18', commit='64dedb8',
        largest_tested=297_271_296,
        note='mg26 (job 15342578), read under the coarse rule: 0.87x at '
             'the 384-class, 1.34x at the 512-class, so the 512-class win '
             'clears the 1.15x margin and the floor sits there.  The '
             'parallel n=2 floor moved above this one on the sorted '
             'forward kernel (mg40), so this row is cone\'s own; the '
             'pair still shares the n=4 row'),
    ('cone', 4): Floor(
        family='cone', count=4, elements=1_023_934_464,
        cell=(1024, 1008, 992), against=2,
        bracket=Bracket(losing_cell=(768, 672, 576), losing_speedup=1.14,
                        winning_cell=(1024, 1008, 992), winning_speedup=1.60),
        spread=0.007508, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-18', commit='64dedb8',
        largest_tested=1_023_934_464,
        note='mg26 (job 15342578), read under the coarse rule: 1.145x at '
             'the 768-class, 1.60x at the 1024-class.  The 768-class win '
             'misses the 1.15x margin by 0.005, and the ruling rounds it '
             'up rather than bending the margin to fit -- a margin chosen '
             'after seeing the numbers is not a rule.  Shared row with '
             'parallel n=4'),
    ('multiaxis', 2): Floor(
        family='multiaxis', count=2, elements=None, cell=None,
        against=1,
        bracket=Bracket(losing_cell=(1024, 1008, 992), losing_speedup=0.80,
                        winning_cell=None, winning_speedup=None),
        spread=0.001937, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-18', commit='64dedb8',
        largest_tested=1_023_934_464,
        note='a sentinel BY RULING (2026-08-19), not by losing everywhere: '
             'mg26 (job 15342578) read 0.35x at the 512-class, a 1.46x WIN '
             'at the 768-class, and a 0.80x LOSS at the 1024-class.  Two '
             'devices win only in a window of sizes, a floor cannot '
             'express a window, and the old 768-class floor widened '
             'automatic 1024-class reconstructions into the measured 20 '
             'percent slowdown.  Held to one device until the mechanism '
             'is understood (multigpu_findings.md sections 1.22 and '
             '1.25); the cost of holding is the forgone 1.46x at the '
             '768-class'),
    ('multiaxis', 4): Floor(
        family='multiaxis', count=4, elements=None, cell=None,
        against=2,
        bracket=Bracket(losing_cell=(1024, 1008, 992), losing_speedup=0.40,
                        winning_cell=None, winning_speedup=None),
        spread=0.02673, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-18', commit='64dedb8',
        largest_tested=1_023_934_464,
        note='a sentinel again at the full protocol (mg26): four devices '
             'lose at every probe against two, 0.87x at the 512-class, '
             '0.23x at the 768-class, and 0.40x at the 1024-class.  The '
             '0.23x reading shares the shape of the n=2 anomaly recorded '
             'in the n=2 row'),
    ('denoiser', 2): Floor(
        family='denoiser', count=2, elements=None, cell=None,
        against=1,
        bracket=Bracket(losing_cell=(1024, 1008, 992), losing_speedup=0.67,
                        winning_cell=None, winning_speedup=None),
        spread=0.01694, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-18', commit='64dedb8',
        largest_tested=1_023_934_464,
        note='still a sentinel on the padded tree (mg26): sharded '
             'denoising lost at every probe cell, 0.56x at the 512-class '
             'to 0.67x at the 1024-class.  Read in IMAGE VOXELS: the '
             'denoiser sinogram shape is its image shape.  The timed '
             'call is denoise with sigma_noise supplied, at the shared '
             'warm-median protocol.  The ratio rises with size, so a '
             'larger ladder may yet find admission; capacity still '
             'widens a denoise that cannot fit one device'),
    ('denoiser', 4): Floor(
        family='denoiser', count=4, elements=None, cell=None,
        against=1,
        bracket=Bracket(losing_cell=(1024, 1008, 992), losing_speedup=0.61,
                        winning_cell=None, winning_speedup=None),
        spread=0.06266, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-18', commit='64dedb8',
        largest_tested=1_023_934_464,
        note='a sentinel like n=2, losing 0.45x to 0.61x across the '
             'probes, measured against n=1 because no smaller denoiser '
             'count is admitted.  Read in image voxels; the n=2 note '
             'records the protocol'),
    ('translation', 2): Floor(
        family='translation', count=2, elements=None, cell=None,
        against=1,
        bracket=Bracket(losing_cell=(256, 1900, 3000), losing_speedup=0.89,
                        winning_cell=None, winning_speedup=None),
        spread=0.06554, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-18', commit='64dedb8',
        largest_tested=1_459_200_000,
        note='a sentinel again (mg26): two devices lose at every probe, '
             '0.64x to 0.89x, rising with size.  The probe cells are '
             'production-anchored translation scans, a fixed 16x16 '
             'translation grid with the detector and the spacing scaled '
             'together, ending at the production cell (256, 1900, 3000); '
             'they are NOT the shared ladder, and the floor reads in '
             'sinogram elements as usual.  The rising trend says '
             'admission may sit just above the largest size tested'),
    ('translation', 4): Floor(
        family='translation', count=4, elements=None, cell=None,
        against=1,
        bracket=Bracket(losing_cell=(256, 1900, 3000), losing_speedup=0.65,
                        winning_cell=None, winning_speedup=None),
        spread=0.01699, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-18', commit='64dedb8',
        largest_tested=1_459_200_000,
        note='a sentinel like n=2, losing 0.16x to 0.65x across the same '
             'production-anchored cells, measured against n=1 because no '
             'smaller translation count is admitted.  The production-cell '
             'reading (0.65x against a 0.2 percent spread) is what holds '
             'the row'),
}

# ── the cost inputs the floors were measured against, named per family ───────
#: The inputs every projection family's cost runs through: the projector
#: assembly, the cross-device transfer primitives the multi-device drivers
#: are built from, the utility module that holds ``padded_kernel_width``, and
#: the three TomographyModel methods that drive the multi-device projections.
#: A method entry is hashed over its source; a file entry is hashed WHOLE,
#: deliberately -- the module-level chunk constants and budget class
#: attributes these files carry are exactly the kind of tuning that moves a
#: crossover without touching any named function.
_SHARED_COST_INPUTS = (
    'projectors.py', '_sharding.py', '_utils.py',
    'TomographyModel._sparse_forward_project_sharded',
    'TomographyModel._sparse_forward_project_cylinders',
    'TomographyModel._sparse_back_project_sharded',
)

#: What prices each family's floors, so staleness is detected PER FAMILY and
#: a refresh can be scoped to the families whose costs actually moved.  On
#: top of the shared set, each projection family adds its geometry body file,
#: and the two kernel families add their kernel files -- ``triton_cone.py``
#: appears in both because it hosts the shared kernel helpers that
#: ``triton_parallel.py`` imports.  The denoiser runs no projector at all, so
#: its set is its own two modules and none of the shared set.
FAMILY_COST_INPUTS = {
    'parallel': _SHARED_COST_INPUTS + ('triton_parallel.py', 'triton_cone.py',
                                       'parallel_beam.py'),
    'cone': _SHARED_COST_INPUTS + ('triton_cone.py', 'cone_beam.py'),
    'multiaxis': _SHARED_COST_INPUTS + ('multiaxis_parallel.py',),
    'translation': _SHARED_COST_INPUTS + ('translation_model.py',),
    'denoiser': ('denoising.py', 'qggmrf.py'),
}

#: Every distinct cost input across the families, split by how it is hashed.
#: Derived from the partition above so the two views cannot drift apart.
_ALL_COST_INPUTS = sorted({name for names in FAMILY_COST_INPUTS.values()
                           for name in names})
COST_INPUT_FILES = tuple(name for name in _ALL_COST_INPUTS
                         if not name.startswith('TomographyModel.'))
COST_INPUT_METHODS = tuple(name.split('.', 1)[1] for name in _ALL_COST_INPUTS
                           if name.startswith('TomographyModel.'))

#: sha256 of each cost input as of the measurements above -- the recorded
#: ("blessed") values the live check compares against.  Re-record them with
#: ``python dev_scripts/refresh_widening_floors.py --bless``.  Recorded
#: 2026-08-19 with the mg40 scoped refresh: ``triton_parallel.py`` is the
#: channel-sorted kernel the fresh parallel rows price, and every other
#: input is unchanged since the mg26 refresh, which is what let mg40 carry
#: the other families' rows.  The names added when the inputs became
#: per-family on 2026-08-19 (the geometry body files, ``_utils.py``, and
#: the denoiser modules) are recorded at their 64dedb8 content, which none
#: of them has moved from.
BLESSED_COST_HASHES = {
    'TomographyModel._sparse_back_project_sharded':
        'f73fa28f32d2393fac3f06139d8ddeccc55260fceb28bbca7f102334839fb5a4',
    'TomographyModel._sparse_forward_project_cylinders':
        '3107ed365cd4d06340996901a9bb0b8440b53cae55195234d89a454535bef48f',
    'TomographyModel._sparse_forward_project_sharded':
        'f4389b004622d77eb74be81a9c6a61628007e1558f657d0a09b7bf790e69266d',
    '_sharding.py':
        '6e05b082984506f1f765491fd33498772758fdad4e3e10c72a8832e02ac6e610',
    '_utils.py':
        '7a4e6a2857be395423b04968f0a9cae2837224f6f6869c0a8f95223e13e9a5b9',
    'cone_beam.py':
        '5b52ff9cf972fec61e8a7ac5dd4ba568bc276e65f80c9caa0df30dc71710ccba',
    'denoising.py':
        '09f902af84895b88bceed9ca4a3f833c279cdd93c3a442cf95c0ce431beb443a',
    'multiaxis_parallel.py':
        '3208b93998462dedf8341351f7dfe30a42d683417b7c1a6da60d9d2b3395d852',
    'parallel_beam.py':
        '065572c91201ce0b354aa9fd15d7aaef5f25bbea24d65b1b46ba8305c5943da1',
    'projectors.py':
        '1cd3b1e39fd0d7f2c5d835545154c044c5e90d399604dd44c944f995551c85ce',
    'qggmrf.py':
        '64e35c114ed049764a3f2f3005fb6341235747270aa2ebe181fff60947bc8602',
    'translation_model.py':
        '0e39120c721edf0ed7d082f74593ce3e0d45a8142919c87224414af9c3cad5b8',
    'triton_cone.py':
        'a4d8350b350cd34a358bb54c33ae3d408f5b1d0131044d0d88b462c5cbaf2dbc',
    'triton_parallel.py':
        '79c4aaaa071c03567f505978a65ed9a183908d6c11342b59ac5668fb10ad0d55',
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
    'a9abaf9cc2935b7ca7e6d7de0c722a5be36994f41efeacab096d31f2254434c6'


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
        families_hit = _families_priced_by(drifted)
        parts.append('the widening speed floors were measured against '
                     'projection-cost code that has since changed ({}, '
                     'pricing the {} floors), so they may no longer '
                     'describe it'.format(', '.join(drifted),
                                          ', '.join(families_hit)))
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
            floors.
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


def _families_priced_by(input_names):
    """The floor families whose cost-input sets contain any of these names."""
    moved = set(input_names)
    return sorted(family for family, names in FAMILY_COST_INPUTS.items()
                  if moved & set(names))


def stale_families():
    """The floor families whose recorded cost inputs no longer hash to their
    recorded values.  Empty means every family's rows still describe the code
    they were measured against.  The refresh tool's family-scoped mode
    measures exactly this set by default, and refuses to carry a family that
    appears here."""
    return _families_priced_by(name for name, _want, _got
                               in stale_cost_inputs())


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
