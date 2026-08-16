"""Re-measure the multi-GPU widening speed floors and print a paste-ready table.

WHY THIS EXISTS.  ``mbirtorch/_widening_floors.py`` holds a MEASUREMENT, not a
constant: the sinogram size at which each device count starts paying for
itself.  Change what a projection costs -- a kernel, a chunk constant, a
transient budget, a banded driver -- and the crossovers move.

NOTHING FORCES YOU TO RUN THIS.  A changed cost input is found automatically:
the library hashes the cost inputs on the first guard consultation in a
process and ``_widening_floors.stale_note()`` names whatever moved, in every
automatic device selection, until it stops moving.
``tests/test_widening_floors.py`` reports the same thing and PASSES.
Re-measuring is work to schedule deliberately -- a planned nightly automation
will own it -- and running this script, then pasting its output, is what
clears the note.

THIS SCRIPT IS THE SOLE WRITER of the three things that must move together:
the floors, their provenance, and the recorded cost-input hashes.  It prints
all three as one block, bound by a checksum, so silencing the staleness note
by hand editing a hash fails the checksum test instead.  That test still
fails HARD: tamper protection was never the optional part.

THE WORDS THIS SCRIPT USES.  A CELL is one problem to measure: a geometry
family and a sinogram shape.  An ARM is one timed reconstruction of a cell at
one device count.  The LADDER is the fixed list of sinogram shapes a refresh
measures at.  A SENTINEL row is one whose ``elements`` is None -- a device
count for which no admission size has been found yet.

THE METHOD, per cell and per device count: one cold recon, DISCARDED, then the
warm median of 3 seeded 3-iteration reconstructions.  Every arm runs in a
FRESH SUBPROCESS with ``MBIRTORCH_NUM_DEVICES`` pinned -- that pin fixes the
count while keeping the model on the automatic branch, and a count changed
inside one process would inherit the previous arm's compiled kernels and
allocator state.  One sinogram per cell is generated once, at ONE device, and
every arm of that cell loads it; otherwise the arms would reconstruct
different arrays and the comparison would not be controlled.

THE DENOISER FAMILY differs in three ways, and every one of them follows from
the denoiser having no projectors.  Its staged input is a noisy phantom rather
than a forward projection, its timed call is ``denoise`` rather than ``recon``,
and its device count is set with ``configure_devices`` rather than with the
environment pin -- ``denoise`` makes no automatic device-count decision, and
the pin acts only through that decision.  Its cell is an IMAGE shape, because
QGGMRFDenoiser sets its sinogram shape equal to its image shape.

THE CROSSOVER RULE.  A count's floor is its crossover against the best smaller
ADMITTED count, not against n=1 unconditionally: parallel n=4 must beat n=2.
The admitted set is read off the CURRENT table, so the comparison count can
change between refreshes -- cone n=4 was measured against n=1 while cone n=2
had no admission size, and the 2026-08-10 refresh that gave cone n=2 a floor
makes the next cone n=4 refresh compare against n=2.  A win counts only when
it clears 1.0x by more than that cell's warm spread, so one noisy cell cannot
move a floor.  Where the crossover falls between two ladder cells, the floor
takes the CONSERVATIVE end -- the larger cell.

WHICH CELLS RUN.  For a finite floor: the ladder cells bracketing it -- one
below, the floor's own, one above.  For a sentinel: the 512-, 768- and
1024-class cells, looking for the admission size that has never been found.
A family the table has no rows for at all takes those same probe cells,
looking for its first admission size; see UNTABLED_FAMILIES.  A sentinel
becomes a finite floor when its count clears 1.0x by more than the measured
spread, and the script prints the proposed floor.  Sizes above a row's
``largest_tested`` are not measured here and are reported as such.

Run:
    python dev_scripts/refresh_widening_floors.py            # a 4-GPU node
    python dev_scripts/refresh_widening_floors.py --plan     # arms, then exit
    python dev_scripts/refresh_widening_floors.py --smoke    # tiny, CPU, fast
    python dev_scripts/refresh_widening_floors.py --bless    # hashes only

``--plan``, ``--smoke`` and ``--bless`` need no GPU.  The real run needs CUDA
and takes roughly 30-60 minutes.  ``--accept-stale`` is accepted and does
nothing: acknowledging by hand that a re-measurement is owed is what the
automatic detection replaced.

Environment:
    REFRESH_PYTHON      interpreter for the arm subprocesses (default: this one)
    REFRESH_RESULTS     scratch directory for the staged sinograms and rows
    REFRESH_ITERATIONS  VCD iterations per recon (default 3)
    REFRESH_REPEATS     warm repeats after the discarded cold pass (default 3)
"""

import argparse
import datetime
import json
import os
import statistics
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbirtorch import _widening_floors as wf                      # noqa: E402

# ── configuration ────────────────────────────────────────────────────────────
TORCH_PYTHON = os.environ.get('REFRESH_PYTHON', sys.executable)
RESULTS_DIR = os.environ.get(
    'REFRESH_RESULTS',
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '_widening_floor_runs'))
ITERATIONS = int(os.environ.get('REFRESH_ITERATIONS', '3'))
WARM_REPEATS = int(os.environ.get('REFRESH_REPEATS', '3'))
SEED = 13

#: The sizes every refresh measures at, roughly geometric, largest last.  A
#: floor is always one of these sizes, so a refresh reports a floor by naming
#: a cell.
LADDER = [(128, 112, 96), (192, 168, 144), (256, 224, 192), (384, 336, 288),
          (512, 448, 384), (768, 672, 576), (1024, 1008, 992)]
#: The cells a sentinel row is measured at.
SENTINEL_PROBES = [(512, 448, 384), (768, 672, 576), (1024, 1008, 992)]
#: Tiny stand-ins so --smoke exercises every path in seconds on a CPU.
SMOKE_LADDER = [(8, 12, 16), (12, 12, 16), (16, 12, 16)]

#: Floor families the table has no rows for yet, and the device counts to
#: measure for each.  The FLOORS table gains a family only through a paste of
#: this script's output, so a family that has never been measured cannot be
#: planned from the table: it is named here instead, and the first refresh
#: that clears one prints its first rows.  Remove a family from here in the
#: same change that pastes its rows in, so it is planned from the table
#: afterwards like every other.  The denoiser left this set on 2026-08-16,
#: when the mg16 refresh gave it its first (sentinel) rows.
UNTABLED_FAMILIES = {}

#: What a family's floor counts, for a family whose metric is not the plain
#: sinogram element count.  Printed with the plan and again under the pasted
#: rows, because the row's own note is where this has to end up and a note is
#: the one field a machine cannot fill in.
FAMILY_METRIC_NOTES = {
    'denoiser': ('QGGMRFDenoiser sets its sinogram shape equal to its image '
                 'shape, so a denoiser floor is read in IMAGE VOXELS where '
                 'every other family reads sinogram elements.  Say so in the '
                 "row's note."),
}


def planned_rows():
    """Every ``(family, count)`` a refresh measures: the table's rows, plus
    the counts named for the families the table has no rows for."""
    rows = set(wf.FLOORS)
    for family, counts in UNTABLED_FAMILIES.items():
        rows.update((family, int(count)) for count in counts)
    return sorted(rows)


def elements(cell):
    return int(cell[0]) * int(cell[1]) * int(cell[2])


def cell_named(target_elements, ladder):
    for cell in ladder:
        if elements(cell) == target_elements:
            return cell
    return None


# ── the plan ─────────────────────────────────────────────────────────────────
def comparison_count(family, count):
    """The best smaller ADMITTED count -- what ``count`` has to beat.

    Read off the table rather than assumed.  A sentinel row is never
    admitted, so it is never the count a wider one has to overtake; a row
    that gains a finite floor joins the comparison at the next refresh, as
    cone n=2 did on 2026-08-10.
    """
    for n in range(count - 1, 1, -1):
        row = wf.FLOORS.get((family, n))
        # Only a count with its OWN measured, finite row can be the
        # comparison: a count that merely INHERITS a floor was never measured
        # in its own right, and a sentinel row is never admitted at all.
        if row is not None and row.elements is not None:
            return n
    return 1


def build_plan(smoke=False):
    """``[{family, count, cell, counts, role}, ...]`` in run order, plus the
    families that have no measured rows at all."""
    ladder = SMOKE_LADDER if smoke else LADDER
    rows = []
    for (family, count) in planned_rows():
        floor = wf.FLOORS.get((family, count))
        against = comparison_count(family, count)
        arms = sorted({1, against, count})
        if floor is None or floor.elements is None:
            # No admission size is known: a row that has never been measured
            # and a sentinel row are searched the same way, at the probe
            # cells, because neither says where its crossover is.
            cells = (ladder[-len(SENTINEL_PROBES):] if smoke
                     else list(SENTINEL_PROBES))
            role = 'first measurement' if floor is None else 'sentinel probe'
        else:
            at = cell_named(floor.elements, ladder)
            if at is None:                       # smoke, or a moved ladder
                at = ladder[len(ladder) // 2]
            index = ladder.index(at)
            cells = ladder[max(0, index - 1):index + 2]
            role = 'bracket'
        for cell in cells:
            rows.append(dict(family=family, count=count, cell=list(cell),
                             counts=arms, role=role, against=against,
                             cell_elements=elements(cell)))
    return rows


def unmeasured_families():
    """Model classes whose automatic device count is set by floors that were
    never measured for them, keyed by the floor family each class declares.

    A geometry reaches this state two ways, and both are work to do rather
    than a surprise in someone's log:

      * it DECLARES a ``_floor_family`` the table has no rows for, keyed here
        under that name; or
      * it declares no family at all -- the inherited base value -- and so
        falls back to the ``wf.DEFAULT_FAMILY`` floors, keyed here under None.

    The second case is the one a newly ported geometry arrives in, and it is
    the one this function used to skip, which left the tool silent about
    exactly the classes relying on the fallback.
    """
    import mbirtorch
    from mbirtorch.tomography_model import TomographyModel

    seen, known = {}, set(wf.families())
    for name in dir(mbirtorch):
        cls = getattr(mbirtorch, name)
        if not (isinstance(cls, type) and issubclass(cls, TomographyModel)):
            continue
        # The base class is not a geometry, and a subclass that does not
        # reconstruct through the shared VCD loop never reaches the automatic
        # device-count decision the floors govern -- QGGMRFDenoiser subclasses
        # TomographyModel but refuses recon, so no floor ever applies to it.
        if cls is TomographyModel or cls.recon is not TomographyModel.recon:
            continue
        family = getattr(cls, '_floor_family', None)
        if family is None or family not in known:
            # An exported alias and its class are the same object, so record
            # the class's own name once rather than once per exported name.
            seen.setdefault(family, set()).add(cls.__name__)
    return {family: sorted(names) for family, names in seen.items()}


def print_plan(plan, smoke):
    ladder = SMOKE_LADDER if smoke else LADDER
    print('widening-floor refresh plan ({}), interpreter {}'.format(
        'SMOKE, CPU' if smoke else 'CUDA', TORCH_PYTHON))
    print('  ladder: ' + ', '.join(
        '{} ({:,})'.format(c, elements(c)) for c in ladder))
    print()
    header = '{:>9}{:>4}{:>9}{:>20}{:>16}{:>18}'.format(
        'family', 'n', 'against', 'cell', 'sino elements', 'role')
    print(header)
    for row in plan:
        print('{:>9}{:>4}{:>9}{:>20}{:>16,}{:>18}'.format(
            row['family'], row['count'], row['against'], str(tuple(row['cell'])),
            row['cell_elements'], row['role']))
    arms = sum(len(row['counts']) for row in plan)
    cells = {(row['family'], tuple(row['cell'])) for row in plan}
    print('\n{} rows, {} distinct cells, {} timed arms + {} generators'.format(
        len(plan), len(cells), arms, len(cells)))
    top = elements(ladder[-1])
    for (family, count) in sorted({(r['family'], r['count']) for r in plan}):
        floor = wf.FLOORS.get((family, count))
        if floor is None:
            print('  NOTE {} n={}: the FLOORS table has no row for this '
                  'family yet, so this run is looking for its first '
                  'admission size at the {} cells above.  Until a row is '
                  'pasted in, the {} floors govern it.'.format(
                      family, count, len(SENTINEL_PROBES), wf.DEFAULT_FAMILY))
        elif floor.elements is None:
            print('  NOTE {} n={}: SENTINEL.  Measured at the {} cells '
                  'above; sizes above largest_tested ({:,} sinogram '
                  'elements) are NOT measured by this plan.'.format(
                      family, count, len(SENTINEL_PROBES),
                      floor.largest_tested))
        elif floor.elements >= top:
            print('  NOTE {} n={}: its floor sits at the TOP of the ladder, '
                  'so the bracket has no cell above it -- a refresh can '
                  'confirm the floor but cannot lower it without a larger '
                  'cell.'.format(family, count))
    planned = {row['family'] for row in plan}
    for family in sorted(planned & set(FAMILY_METRIC_NOTES)):
        print('  METRIC {}: {}'.format(family, FAMILY_METRIC_NOTES[family]))
    missing = unmeasured_families()
    if missing:
        # The None key sorts first: a class that declares no family is taking
        # the fallback silently, which is the case worth reading first.
        for family, classes in sorted(
                missing.items(),
                key=lambda item: (item[0] is not None, item[0] or '')):
            if family is None:
                print('  NEEDS MEASUREMENT: {} declare no floor family, so the '
                      '{} floors govern their automatic device count.'
                      .format(', '.join(classes), wf.DEFAULT_FAMILY))
            else:
                print('  NEEDS MEASUREMENT: floor family {!r} is declared by {} '
                      'but has no rows; it currently inherits the {} floors.'
                      .format(family, ', '.join(classes), wf.DEFAULT_FAMILY))
    else:
        print('  every model class is governed by floors measured for it.')


# ── the worker: one arm, one subprocess ──────────────────────────────────────
#: The noise standard deviation the denoiser family is measured at, as a
#: fraction of the phantom's unit dynamic range.  It is passed to ``denoise``
#: rather than estimated, so no arm spends time on the host-side estimate,
#: which does no device work and would add a size-dependent constant to every
#: reading.
DENOISE_SIGMA = 0.1


def _build_model(family, cell, device, n_dev=None):
    """The model one arm times.

    ``n_dev`` is the arm's device count.  It matters only for the denoiser
    family, which is placed explicitly; see the denoiser branch below.
    """
    import numpy as np

    import mbirtorch

    num_views, _rows, num_channels = cell
    if family == 'cone':
        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
        model = mbirtorch.ConeBeamModel(
            tuple(cell), angles, source_detector_dist=4.0 * num_channels,
            source_iso_dist=2.0 * num_channels)
    elif family == 'parallel':
        angles = np.linspace(0, np.pi, num_views, endpoint=False)
        model = mbirtorch.ParallelBeamModel(tuple(cell), angles)
    elif family == 'denoiser':
        # The denoiser's input is an image, not a sinogram, and its
        # sinogram_shape is set equal to that image shape, so the cell IS the
        # image shape here.  Its floors are therefore read in image voxels
        # where every other family's are read in sinogram elements, and the
        # rows the refresh prints have to record that.
        model = mbirtorch.QGGMRFDenoiser(tuple(cell))
        if device == 'cuda':
            # The arm's count is configured explicitly.  This is the
            # protocol the family's rows were first measured under
            # (2026-08-16, before denoise called the device policy), and it
            # still pins correctly now that denoise does: an explicit
            # layout is never second-guessed.  The environment pin would
            # work today too -- a pin bypasses the floors -- but keeping
            # the explicit list keeps the measured protocol unchanged
            # across refreshes.  Each arm reports realized_devices, so a
            # count that did not take is visible in the run output.
            model.configure_devices(
                devices=['cuda:{}'.format(i) for i in range(n_dev or 1)])
    else:
        # Falling through to parallel beam here would time parallel beam and
        # record the result under this family's name, which is the one way a
        # floor can be wrong without anything looking wrong.
        raise ValueError(
            'refresh_widening_floors cannot build a model for floor family '
            '{!r}.  Add its geometry to _build_model before measuring it.'
            .format(family))
    if device != 'cuda':
        # CPU/MPS only: the env pin is a CUDA mechanism (the policy
        # short-circuits below two visible devices), so the smoke has to place
        # the model by hand.  On CUDA nothing is configured, which keeps the
        # model on the automatic branch the pin acts through.  The denoiser
        # branch above is the exception, and it is placed there.
        model.configure_devices(devices=[device])
    model.set_params(no_warning=True, verbose=0)
    return model


def _input_path(family, cell):
    """Where one cell's staged input array lives: a sinogram for the
    projection families, a noisy image for the denoiser."""
    return os.path.join(RESULTS_DIR, '_sino_{}_{}x{}x{}.npy'.format(
        family, *cell))


def generate(cfg):
    """Stage one input array per cell, at ONE device, so every arm of that
    cell works on the same array.

    A projection family's input is the forward projection of a phantom.  The
    denoiser has no projectors, so its input is that phantom plus seeded
    gaussian noise -- the array a denoiser is actually given.
    """
    import numpy as np

    import mbirtorch

    model = _build_model(cfg['family'], cfg['cell'],
                         'cpu' if cfg['device'] != 'cuda' else 'cuda',
                         n_dev=1)
    recon_shape = tuple(model.get_params('recon_shape'))
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    if cfg['family'] == 'denoiser':
        noise = np.random.RandomState(SEED).randn(*recon_shape)
        staged = np.asarray(phantom + DENOISE_SIGMA * noise, dtype=np.float32)
    else:
        staged = np.asarray(_to_numpy(model.forward_project(phantom)),
                            dtype=np.float32)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(_input_path(cfg['family'], cfg['cell']),
            np.ascontiguousarray(staged))
    return dict(cfg, role='generator', recon_shape=list(recon_shape),
                sino_shape=list(staged.shape))


def _to_numpy(x):
    """The one host exit.  ``Shards.gather()`` ALREADY returns numpy, so a
    gather is never followed by ``.detach()`` -- re-detaching one is a
    recorded way to lose every multi-device row in a run."""
    import numpy as np

    if isinstance(x, np.ndarray):
        return x
    if callable(getattr(x, 'gather', None)) and hasattr(x, 'placement'):
        return x.gather()
    return (x.detach().cpu().numpy()
            if callable(getattr(x, 'detach', None)) else np.asarray(x))


def measure(cfg):
    """One arm: cold pass discarded, then the warm median of REPEATS runs."""
    import numpy as np
    import torch

    family = cfg['family']
    model = _build_model(family, cfg['cell'], cfg['device'],
                         n_dev=cfg.get('n_dev'))
    staged = np.load(_input_path(family, cfg['cell']))
    weights = (None if family == 'denoiser' else
               np.exp(-staged / (2 * np.max(staged))).astype(np.float32))

    def one():
        np.random.seed(SEED)
        if family == 'denoiser':
            # The denoiser's own entry point; recon raises on it.  The
            # iteration count and the exact-iteration stop match the
            # reconstruction call below, so both families are timed over the
            # same amount of work per run.
            out, _info = model.denoise(staged, sigma_noise=DENOISE_SIGMA,
                                       max_iterations=ITERATIONS,
                                       stop_threshold_change_pct=0.0)
        else:
            out, _info = model.recon(staged, weights=weights,
                                     max_iterations=ITERATIONS,
                                     stop_threshold_change_pct=0.0)
        if cfg['device'] == 'cuda':
            # Both placements name the same device list; the recon one is
            # named because it is the one the denoiser divides its image on.
            for device in model.recon_placement.devices:
                torch.cuda.synchronize(device)
        return _to_numpy(out)

    start = time.perf_counter()
    out = one()
    cold = time.perf_counter() - start
    warm = []
    for _ in range(WARM_REPEATS):
        start = time.perf_counter()
        out = one()
        warm.append(time.perf_counter() - start)
    median = statistics.median(warm)
    return dict(cfg, role='arm', cold_s=cold, warm_all=warm, warm_s=median,
                spread=(max(warm) - min(warm)) / median,
                realized_devices=len(model.recon_placement.devices),
                recon_checksum=float(np.sum(np.abs(out), dtype=np.float64)))


# ── the runner ───────────────────────────────────────────────────────────────
def arm_env(cfg):
    """The environment that DEFINES an arm, set explicitly so nothing is
    inherited from the submitting shell."""
    env = dict(os.environ)
    env.pop('MBIRTORCH_MEMORY_CALIBRATION', None)
    env.pop('MBIRTORCH_NUM_DEVICES', None)
    env.pop('MBIRTORCH_WIDENING_GUARD', None)   # the pin already bypasses it
    env['MBIRTORCH_DISABLE_TRITON'] = '0'       # production: kernels on
    if cfg.get('n_dev') and cfg['device'] == 'cuda':
        env['MBIRTORCH_NUM_DEVICES'] = str(cfg['n_dev'])
    return env


def run_one(cfg, tag):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, '_cfg_{}.json'.format(tag))
    out_path = os.path.join(RESULTS_DIR, '_out_{}.json'.format(tag))
    with open(cfg_path, 'w') as handle:
        json.dump(cfg, handle)
    if os.path.exists(out_path):
        os.remove(out_path)
    proc = subprocess.run([TORCH_PYTHON, '-u', os.path.abspath(__file__),
                           '_worker', cfg_path, out_path], env=arm_env(cfg))
    if proc.returncode != 0 and not os.path.exists(out_path):
        return dict(cfg, error='worker exited {}'.format(proc.returncode))
    with open(out_path) as handle:
        return json.load(handle)


def _worker_main(cfg_path, out_path):
    with open(cfg_path) as handle:
        cfg = json.load(handle)
    try:
        row = generate(cfg) if cfg['role'] == 'generator' else measure(cfg)
    except Exception:                                             # noqa: BLE001
        row = dict(cfg, error=traceback.format_exc()[-3000:])
    with open(out_path, 'w') as handle:
        json.dump(row, handle)


def run_plan(plan, device):
    """Every generator, then every arm.  Returns ``{(family, cell, n): row}``."""
    cells = sorted({(row['family'], tuple(row['cell'])) for row in plan})
    for family, cell in cells:
        cfg = dict(family=family, cell=list(cell), device=device,
                   role='generator')
        tag = 'gen_{}_{}x{}x{}'.format(family, *cell)
        print('  generator {} {} ...'.format(family, cell), flush=True)
        row = run_one(cfg, tag)
        if row.get('error'):
            print('    ERROR: {}'.format(str(row['error'])[:400]))

    measured = {}
    arms = sorted({(row['family'], tuple(row['cell']), n)
                   for row in plan for n in row['counts']})
    for index, (family, cell, n_dev) in enumerate(arms):
        cfg = dict(family=family, cell=list(cell), device=device,
                   role='arm', n_dev=n_dev)
        tag = 'arm_{}_{}x{}x{}_n{}'.format(family, *cell, n_dev)
        print('  [{}/{}] {} {} n={} ...'.format(index + 1, len(arms), family,
                                                cell, n_dev), flush=True)
        row = run_one(cfg, tag)
        measured[(family, cell, n_dev)] = row
        if row.get('error'):
            print('    ERROR: {}'.format(str(row['error'])[:400]))
        else:
            print('    warm {:.3f}s  spread {:.1%}  realized {} device(s)'
                  .format(row['warm_s'], row['spread'],
                          row['realized_devices']))
    return measured


# ── the analysis ─────────────────────────────────────────────────────────────
def speedup(measured, family, cell, count, against):
    """``against``'s warm median over ``count``'s: above 1.0 means the wider
    count is faster."""
    a = measured.get((family, cell, against))
    b = measured.get((family, cell, count))
    if not a or not b or a.get('error') or b.get('error'):
        return None, None
    spread = max(a['spread'], b['spread'])
    return a['warm_s'] / b['warm_s'], spread


def verdict(plan, measured):
    """Per table row: the per-cell speedups, and the floor the crossover
    rule above reads off them.

    A win counts only when it clears 1.0x by MORE than that cell's warm
    spread, which is what keeps one noisy cell from moving a floor.
    """
    out = {}
    # Read off the PLAN rather than the table, so a family the table has no
    # rows for is judged too.
    for (family, count) in sorted({(r['family'], r['count']) for r in plan}):
        cells = [tuple(row['cell']) for row in plan
                 if row['family'] == family and row['count'] == count]
        against = comparison_count(family, count)
        rows, winner = [], None
        for cell in cells:
            ratio, spread = speedup(measured, family, cell, count, against)
            wins = ratio is not None and ratio - 1.0 > spread
            rows.append((cell, ratio, spread, wins))
            if wins and winner is None:
                winner = cell
        out[(family, count)] = dict(against=against, rows=rows, winner=winner)
    return out


def print_verdict(verdicts):
    for (family, count), record in sorted(verdicts.items()):
        floor = wf.FLOORS.get((family, count))
        print('\n===== {} n={}  (crossover against n={}) ====='.format(
            family, count, record['against']))
        for cell, ratio, spread, wins in record['rows']:
            if ratio is None:
                print('  {:>20} {:>16,}   (no measurement)'.format(
                    str(cell), elements(cell)))
                continue
            print('  {:>20} {:>16,}   {:.3f}x  spread {:.1%}   {}'.format(
                str(cell), elements(cell), ratio, spread,
                'WINS' if wins else 'loses'))
        largest = max(elements(c) for c, _r, _s, _w in record['rows'])
        winner = record['winner']
        proposed = elements(winner) if winner else None
        # A family with no table row is in the same state as a sentinel: no
        # admission size is known for it.  The two are reported separately so
        # the reader can tell a first measurement from a re-measurement.
        if winner is None and floor is None:
            print('  no admission size in these cells -> {} n={} still has '
                  'no row; sizes above {:,} elements are unmeasured.'.format(
                      family, count, largest))
        elif winner is None and floor.elements is None:
            print('  no admission size in these cells -> the row stays a '
                  'SENTINEL; sizes above {:,} sinogram elements are still '
                  'unmeasured.'.format(largest))
        elif winner is None:
            print('  NOTHING WON: the floor is above every cell measured '
                  '({:,} sinogram elements).  Re-run with a larger ladder; '
                  'this run cannot place a floor.'.format(largest))
        elif floor is None:
            print('  FIRST FLOOR: n={} beats 1.0x by more than its spread at '
                  '{}.  Proposed floor: {:,} elements.  Paste the row below, '
                  'then drop {!r} from UNTABLED_FAMILIES.'.format(
                      count, winner, proposed, family))
        elif floor.elements is None:
            print('  SENTINEL CLEARED: n={} beats 1.0x by more than its '
                  'spread at {}.  Proposed finite floor: {:,} sinogram '
                  'elements.'.format(count, winner, proposed))
        elif proposed != floor.elements:
            print('  FLOOR MOVES: {:,} -> {:,} sinogram elements.'.format(
                floor.elements, proposed))
        else:
            print('  floor unchanged at {:,} sinogram elements.'.format(
                proposed))


_ROW_TEMPLATE = """    ('{family}', {count}): Floor(
        family='{family}', count={count}, elements={elements}, cell={cell},
        against={against},
        bracket=Bracket(losing_cell={low_cell}, losing_speedup={low_x},
                        winning_cell={cell}, winning_speedup={win_x}),
        spread={spread:.4g}, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='{today}', commit='{commit}',
        largest_tested={largest},
        note='...'),"""


def print_table(verdicts, commit):
    """The paste-ready replacement for FLOORS, provenance included."""
    today = datetime.date.today().isoformat()
    print('\n' + '=' * 78)
    print('PASTE INTO mbirtorch/_widening_floors.py (FLOORS, then the three')
    print('bound constants printed by --bless).  All of it, or none of it.')
    print('=' * 78 + '\nFLOORS = {')
    for (family, count), record in sorted(verdicts.items()):
        winner, rows = record['winner'], record['rows']
        found = elements(winner) if winner else None
        # The bracket's low side is the largest cell BELOW the winner that
        # lost.  A cell above the winner may also read under 1.0x on noise,
        # and calling that the losing side would invert the bracket.
        below = [(c, r) for c, r, _s, wins in rows if r and not wins
                 and (found is None or elements(c) < found)]
        low = below[-1] if below else (None, None)
        win_ratio = next((r for c, r, _s, _w in rows if c == winner), None)
        print(_ROW_TEMPLATE.format(
            family=family, count=count, against=record['against'],
            elements='{:_}'.format(found) if found else 'None',
            cell=winner if winner else 'None',
            low_cell=low[0] if low[0] else 'None',
            low_x='{:.2f}'.format(low[1]) if low[1] else 'None',
            win_x='{:.2f}'.format(win_ratio) if win_ratio else 'None',
            spread=max([s for _c, _r, s, _w in rows if s is not None] or [0.0]),
            today=today, commit=commit,
            # THIS run's largest cell, never the old row's: claiming a size
            # this refresh never measured is the false claim the field exists
            # to prevent.
            largest='{:_}'.format(max(elements(c)
                                      for c, _r, _s, _w in rows))))
    print('}')
    print('\nCheck MEASURED_GPU and MEASURED_CONFIG still describe this run, '
          'and rewrite each note by hand: a note is the one field a machine '
          'cannot fill in.')
    for family in sorted({f for f, _c in verdicts} & set(FAMILY_METRIC_NOTES)):
        print('\n{}: {}'.format(family, FAMILY_METRIC_NOTES[family]))
    new = sorted({f for f, _c in verdicts} & set(UNTABLED_FAMILIES))
    if new:
        print('\nThese families are new to the table: {}.  Drop each from '
              "UNTABLED_FAMILIES in this script once its rows are pasted in, "
              'so it is planned from the table afterwards.'.format(
                  ', '.join(new)))


def head_commit():
    try:
        out = subprocess.run(['git', 'log', '-1', '--format=%h'],
                             capture_output=True, text=True, timeout=10,
                             cwd=os.path.dirname(os.path.dirname(
                                 os.path.abspath(__file__))))
        return out.stdout.strip() or 'unknown'
    except Exception:                                             # noqa: BLE001
        return 'unknown'


def do_bless():
    """``--bless``: print fresh values for the three bound constants."""
    stale = wf.stale_cost_inputs()
    print('projection-cost inputs that moved since the hashes were last '
          'recorded: {}'.format(
              ', '.join(name for name, _w, _g in stale) or 'none'))
    print('\nRecording the current hashes after a measurement run.  If you')
    print('have NOT just re-measured, doing this hides a real change from')
    print('the automatic staleness note; leave the old hashes in place')
    print('instead and let the note stand until the floors are')
    print('re-measured.\n')
    print(wf.bless_lines(stale_since=None))
    return 0


def do_accept_stale():
    """``--accept-stale``, kept parseable and made a no-op.

    It used to stamp STALE_SINCE so the test would go green.  The test no
    longer goes red over drift and the note no longer needs a stamp, so the
    flag has nothing left to do.
    """
    print('--accept-stale: no longer needed -- drift is detected and logged '
          'automatically; run a real refresh to clear the note.')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='re-measure the multi-GPU widening speed floors')
    parser.add_argument('--plan', action='store_true',
                        help='print the arm list and exit (no GPU needed)')
    parser.add_argument('--smoke', action='store_true',
                        help='run tiny CPU cells end to end to prove the '
                             'plumbing (no GPU needed)')
    parser.add_argument('--bless', action='store_true',
                        help='recompute and print the three bound constants, '
                             'recording the current cost-input hashes '
                             '(no GPU needed)')
    parser.add_argument('--accept-stale', action='store_true',
                        help='accepted and ignored: staleness is detected and '
                             'logged automatically, so there is nothing to '
                             'acknowledge by hand')
    args = parser.parse_args(argv)

    # Checked before --bless: the pair used to mean "record the hashes
    # without measuring", and someone typing it from memory should be told it
    # is unnecessary rather than quietly re-record the hashes for real.
    if args.accept_stale:
        return do_accept_stale()
    if args.bless:
        return do_bless()

    plan = build_plan(smoke=args.smoke)
    if args.plan:
        print_plan(plan, args.smoke)
        return 0

    device = 'cpu' if args.smoke else 'cuda'
    if not args.smoke:
        import torch
        if not torch.cuda.is_available():
            print('a real refresh needs CUDA; use --smoke to test the '
                  'plumbing on this machine.')
            return 2
        print('{} CUDA device(s) visible'.format(torch.cuda.device_count()))
    else:
        print('SMOKE: tiny cells on the CPU.  The device-count pin is a CUDA '
              'mechanism, so every arm realizes ONE device here and the '
              'speedups are all 1.0x -- this proves the plumbing (batching, '
              'subprocess env, analysis, table printing), not the floors.')

    print_plan(plan, args.smoke)
    print('\nrunning ...')
    started = time.time()
    measured = run_plan(plan, device)
    verdicts = verdict(plan, measured)
    print_verdict(verdicts)
    print_table(verdicts, head_commit())
    print('\nthen: python dev_scripts/refresh_widening_floors.py --bless')
    print('elapsed {:.1f} min; rows under {}'.format(
        (time.time() - started) / 60, RESULTS_DIR))
    return 0


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == '_worker':
        _worker_main(sys.argv[2], sys.argv[3])
    else:
        sys.exit(main())
