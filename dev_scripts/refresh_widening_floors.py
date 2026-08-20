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
measures at.  Most families share one ladder; a family whose problems that
ladder does not describe has a ladder of its own, named in FAMILY_LADDERS.  A
SENTINEL row is one whose ``elements`` is None -- a device count for which no
admission size has been found yet.

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

THE MULTIAXIS FAMILY takes the shared ladder, and its models are built the
way mg18 built them: two angles per view, azimuths evenly spaced over half a
turn and elevations swept across +/- 0.5 radians.  The elevation range is
part of what a multiaxis cell measures rather than a free choice, because the
automatic geometry sizes the reconstruction from it.

THE TRANSLATION FAMILY has a ladder of its own.  The shared ladder does not
describe a translation scan: production runs a 16x16 grid of translations,
which is 256 views, against a large detector.  Translation's three cells keep
that grid and halve the detector and the spacing together, so all three scan
the same object at three resolutions, and the largest is the production cell
mg18 measured.  A sinogram shape alone does not determine a translation model
the way an angle list does, so TRANSLATION_SPECS records the grid and the
spacing for every translation cell and ``_build_model`` reads them.  The
generator differs too.  The shepp-logan phantom comes back all zeros on a
volume only a few voxels deep, which is what the --smoke translation cell
reconstructs, so the generator falls back to a seeded uniform phantom and
records that it did.

The first translation refresh measures BOTH counts against one device.  The
crossover rule below reads the admitted set off the CURRENT table, and
translation has no measured smaller count there yet.  Once its rows are
pasted in, the next refresh compares n=4 against n=2, provided n=2 gained a
finite floor.  Cone n=4 reached its comparison count the same way.

THE CROSSOVER RULE.  A count's floor is its crossover against the best smaller
ADMITTED count, not against n=1 unconditionally: parallel n=4 must beat n=2.
The admitted set is read off the CURRENT table, so the comparison count can
change between refreshes -- cone n=4 was measured against n=1 while cone n=2
had no admission size, and the 2026-08-10 refresh that gave cone n=2 a floor
makes the next cone n=4 refresh compare against n=2.  A cell WINS only under
the coarse admission rule (ruled 2026-08-19): the speedup must reach
``ADMISSION_MARGIN`` (1.15x) AND clear 1.0x by more than that cell's warm
spread.  The floor is the smallest measured cell that wins; a thinner win
rounds the floor up one class, and a thin win at the largest measured cell
means this run cannot place the floor and says so.  Where the crossover
falls between two ladder cells, the floor takes the CONSERVATIVE end -- the
larger cell.

THE SCOPED REFRESH (``--families``).  A full refresh re-measures every
family, which costs about 12 GPU-hours, and most of that re-measures rows
whose costs did not change.  ``--families cone,parallel`` measures the named
families only; bare ``--families`` measures exactly the families whose
recorded cost inputs moved (``_widening_floors.stale_families()``).  Every
other family's rows are CARRIED: printed verbatim in the paste, provenance
untouched, so the tool stays the sole writer of the whole table and the
checksum binding survives.  Carrying a family asserts its costs did not
change, and with per-family cost inputs that assertion is checkable: a
scoped run REFUSES to start if a family it would carry has drifted inputs.
A change to a shared input therefore forces the full refresh, and a change
confined to one family's files allows the scoped one.

WHICH CELLS RUN.  For a finite floor: the ladder cells bracketing it -- one
below, the floor's own, one above.  For a sentinel: the top three cells of
that family's ladder, looking for the admission size that has never been
found.  On the shared ladder those are the 512-, 768- and 1024-class cells.
A family the table has no rows for at all takes those same probe cells,
looking for its first admission size; see UNTABLED_FAMILIES.  A sentinel
becomes a finite floor when its count clears 1.0x by more than the measured
spread, and the script prints the proposed floor.  Sizes above a row's
``largest_tested`` are not measured here and are reported as such.

Run:
    python dev_scripts/refresh_widening_floors.py            # a 4-GPU node
    python dev_scripts/refresh_widening_floors.py --families # stale families only
    python dev_scripts/refresh_widening_floors.py --families cone,parallel
    python dev_scripts/refresh_widening_floors.py --plan     # arms, then exit
    python dev_scripts/refresh_widening_floors.py --smoke    # tiny, CPU, fast
    python dev_scripts/refresh_widening_floors.py --bless    # hashes only

``--plan``, ``--smoke`` and ``--bless`` need no GPU.  The real run needs CUDA.
How long it takes follows the plan: probing the largest cells at three device
counts costs hours by itself, so run ``--plan`` first and budget from the arm
list it prints.  ``--accept-stale`` is accepted and does nothing:
acknowledging by hand that a re-measurement is owed is what the automatic
detection replaced.

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
#: The cells a sentinel row on the shared ladder is measured at -- its top
#: three.  A family with a ladder of its own takes that ladder's top three
#: instead; see ``probe_cells``.
SENTINEL_PROBES = [(512, 448, 384), (768, 672, 576), (1024, 1008, 992)]
#: Tiny stand-ins so --smoke exercises every path in seconds on a CPU.
SMOKE_LADDER = [(8, 12, 16), (12, 12, 16), (16, 12, 16)]

#: The translation grid and spacing behind each translation cell:
#: ``cell -> (num_x, num_z, x_spacing, z_spacing)``.  A translation scan
#: moves the object across a fixed source and detector on a grid, so its
#: views are grid positions rather than angles, and the same sinogram shape
#: could come from many different grids.  The grid is therefore written down
#: here rather than derived, and ``_build_model`` reads it.  The three
#: measured cells keep production's 16x16 grid and halve the detector and the
#: spacing together; the largest is the production cell mg18 measured, and
#: its grid, spacing and source distances are mg18's.
TRANSLATION_SPECS = {
    (256, 475, 750): (16, 16, 6.0, 4.0),
    (256, 950, 1500): (16, 16, 12.0, 8.0),
    (256, 1900, 3000): (16, 16, 24.0, 16.0),
    (16, 40, 32): (4, 4, 3.0, 2.0),        # the --smoke cell, also mg18's
}
#: Translation's ladder, and the tiny stand-in --smoke uses for it.
TRANSLATION_LADDER = [(256, 475, 750), (256, 950, 1500), (256, 1900, 3000)]
TRANSLATION_SMOKE_LADDER = [(16, 40, 32)]

#: Families measured at a ladder of their own, and the tiny stand-ins --smoke
#: uses for them.  A family belongs here when the shared ladder does not
#: describe its problems: a translation scan has few views and a large
#: detector, so no cell on the shared ladder resembles one.
FAMILY_LADDERS = {'translation': TRANSLATION_LADDER}
FAMILY_SMOKE_LADDERS = {'translation': TRANSLATION_SMOKE_LADDER}

#: Floor families the table has no rows for yet, and the device counts to
#: measure for each.  The FLOORS table gains a family only through a paste of
#: this script's output, so a family that has never been measured cannot be
#: planned from the table: it is named here instead, and the first refresh
#: that clears one prints its first rows.  Remove a family from here in the
#: same change that pastes its rows in, so it is planned from the table
#: afterwards like every other.  The denoiser left this set on 2026-08-16,
#: when the mg16 refresh gave it its first (sentinel) rows; translation
#: entered on 2026-08-17 and left the same day, when the mg22 refresh gave
#: it its first (sentinel) rows and TranslationModel its family declaration.
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
    'translation': ('a translation floor is read in sinogram elements like '
                    'the other projection families, but its probe cells are '
                    'not the shared ladder.  They are production-anchored '
                    'translation scans: a fixed 16x16 translation grid, with '
                    'the detector and the spacing scaled together.  Say so '
                    "in the row's note and name the cell, so nobody reads a "
                    'translation floor as a size on the shared ladder.'),
}


def planned_rows():
    """Every ``(family, count)`` a refresh measures: the table's rows, plus
    the counts named for the families the table has no rows for."""
    rows = set(wf.FLOORS)
    for family, counts in UNTABLED_FAMILIES.items():
        rows.update((family, int(count)) for count in counts)
    return sorted(rows)


def resolve_scope(families_arg):
    """The families this run measures: None for every family, else a sorted
    list.  Raises SystemExit on an unknown name, and on a scope that would
    CARRY a family whose cost inputs moved -- carrying a family asserts its
    costs did not change, and a drifted family fails that assertion, so the
    refusal happens here, before any GPU work.
    """
    all_families = sorted({family for family, _count in planned_rows()})
    stale = wf.stale_families()
    if families_arg is None:
        return None
    if families_arg == 'stale':
        chosen = list(stale)
        if not chosen:
            raise SystemExit(
                'no cost input has moved since the hashes were recorded, so '
                'a stale-scoped refresh has nothing to measure.  Name '
                'families explicitly (--families cone,parallel) for a '
                'deliberate re-measure, or run without --families for the '
                'full refresh.')
    else:
        chosen = sorted({token.strip() for token in families_arg.split(',')
                         if token.strip()})
        unknown = [name for name in chosen if name not in all_families]
        if unknown:
            raise SystemExit(
                'unknown floor famil{}: {}.  The families a refresh can '
                'measure are: {}.'.format(
                    'y' if len(unknown) == 1 else 'ies', ', '.join(unknown),
                    ', '.join(all_families)))
    carried_stale = [name for name in stale if name not in chosen]
    if carried_stale:
        raise SystemExit(
            'refusing the scoped refresh: the cost inputs of {} moved, and '
            'this scope would carry {} rows forward as if their costs were '
            'unchanged.  Widen the scope to include {}, or run the full '
            'refresh.'.format(', '.join(carried_stale),
                              'their' if len(carried_stale) > 1 else 'its',
                              ', '.join(carried_stale)))
    return chosen


def elements(cell):
    return int(cell[0]) * int(cell[1]) * int(cell[2])


def cell_named(target_elements, ladder):
    for cell in ladder:
        if elements(cell) == target_elements:
            return cell
    return None


def family_ladder(family, smoke=False):
    """The cells ``family`` is measured at.  Most families share LADDER; one
    named in FAMILY_LADDERS is measured at its own cells instead."""
    if smoke:
        return FAMILY_SMOKE_LADDERS.get(family, SMOKE_LADDER)
    return FAMILY_LADDERS.get(family, LADDER)


def probe_cells(ladder):
    """The cells a row with no known admission size is measured at: the top of
    that family's ladder.  On LADDER these are exactly SENTINEL_PROBES."""
    return list(ladder[-len(SENTINEL_PROBES):])


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


def build_plan(smoke=False, scope=None):
    """``[{family, count, cell, counts, role}, ...]`` in run order, plus the
    families that have no measured rows at all.  ``scope`` limits the plan to
    the named families; None plans every family."""
    rows = []
    for (family, count) in planned_rows():
        if scope is not None and family not in scope:
            continue
        ladder = family_ladder(family, smoke)
        floor = wf.FLOORS.get((family, count))
        against = comparison_count(family, count)
        arms = sorted({1, against, count})
        if floor is None or floor.elements is None:
            # No admission size is known: a row that has never been measured
            # and a sentinel row are searched the same way, at the probe
            # cells, because neither says where its crossover is.
            cells = probe_cells(ladder)
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


def print_plan(plan, smoke, scope=None):
    ladder = SMOKE_LADDER if smoke else LADDER
    print('widening-floor refresh plan ({}), interpreter {}'.format(
        'SMOKE, CPU' if smoke else 'CUDA', TORCH_PYTHON))
    if scope is not None:
        carried = sorted({family for family, _count in planned_rows()}
                         - set(scope))
        print('  SCOPED: measuring {}; carrying verbatim: {}'.format(
            ', '.join(scope), ', '.join(carried) or 'nothing'))
    print('  ladder: ' + ', '.join(
        '{} ({:,})'.format(c, elements(c)) for c in ladder))
    # A family measured at its own cells prints them too, so the plan never
    # shows a cell that is absent from the ladder printed above it.
    for family in sorted({row['family'] for row in plan} & set(FAMILY_LADDERS)):
        print('  {} ladder: '.format(family) + ', '.join(
            '{} ({:,})'.format(c, elements(c))
            for c in family_ladder(family, smoke)))
    print()
    # The family column is wide enough for the longest family name, so a long
    # one does not push the rest of the row out of alignment.
    header = '{:>11}{:>4}{:>9}{:>20}{:>16}{:>18}'.format(
        'family', 'n', 'against', 'cell', 'sino elements', 'role')
    print(header)
    for row in plan:
        print('{:>11}{:>4}{:>9}{:>20}{:>16,}{:>18}'.format(
            row['family'], row['count'], row['against'], str(tuple(row['cell'])),
            row['cell_elements'], row['role']))
    arms = sum(len(row['counts']) for row in plan)
    cells = {(row['family'], tuple(row['cell'])) for row in plan}
    print('\n{} rows, {} distinct cells, {} timed arms + {} generators'.format(
        len(plan), len(cells), arms, len(cells)))
    for (family, count) in sorted({(r['family'], r['count']) for r in plan}):
        floor = wf.FLOORS.get((family, count))
        # Read against the family's OWN ladder: a family measured elsewhere
        # would otherwise be judged against a top cell it never runs.
        top = elements(family_ladder(family, smoke)[-1])
        # Count the cells this row actually got, rather than assuming the
        # shared probe count: a short ladder yields fewer.
        probes = sum(1 for r in plan
                     if r['family'] == family and r['count'] == count)
        if floor is None:
            print('  NOTE {} n={}: the FLOORS table has no row for this '
                  'family yet, so this run is looking for its first '
                  'admission size at the {} cells above.  Until a row is '
                  'pasted in, the {} floors govern it.'.format(
                      family, count, probes, wf.DEFAULT_FAMILY))
        elif floor.elements is None:
            print('  NOTE {} n={}: SENTINEL.  Measured at the {} cells '
                  'above; sizes above largest_tested ({:,} sinogram '
                  'elements) are NOT measured by this plan.'.format(
                      family, count, probes, floor.largest_tested))
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
    elif family == 'multiaxis':
        # Two angles per view: azimuth around the object, elevation (tilt)
        # out of the plane.  These are the geometry's own test defaults, and
        # mg18 measured this family with them -- azimuths evenly spaced over
        # half a turn, elevations swept across +/- 0.5 radians.  The
        # elevation range sets the recon shape: the automatic geometry
        # divides the detector height by the smallest |cos(elevation)| and
        # clamps that divisor at 0.1, so a range wide enough to reach the
        # clamp would inflate the slice count roughly tenfold.  0.5 radians
        # is far from the clamp.
        azimuth = np.linspace(0, np.pi, num_views, endpoint=False)
        elevation = np.linspace(-0.5, 0.5, num_views)
        model = mbirtorch.MultiAxisParallelModel(
            tuple(cell), np.stack([azimuth, elevation], axis=1))
    elif family == 'translation':
        # A translation scan moves the object across a fixed source and
        # detector on a grid, so its views are grid positions rather than
        # angles.  The cell does not say what that grid is, so the grid and
        # the spacing come from TRANSLATION_SPECS.  The construction below is
        # mg18's, including the source distances.
        spec = TRANSLATION_SPECS.get(tuple(cell))
        if spec is None:
            raise ValueError(
                'refresh_widening_floors has no translation grid recorded '
                'for cell {}.  Add it to TRANSLATION_SPECS before measuring '
                'that cell.'.format(tuple(cell)))
        num_x, num_z, x_spacing, z_spacing = spec
        vectors = mbirtorch.gen_translation_vectors(
            num_x, num_z, x_spacing=x_spacing, z_spacing=z_spacing)
        if vectors.shape[0] != num_views:
            raise ValueError(
                'translation cell {} has {} views, but its {}x{} grid gives '
                '{} translations.'.format(tuple(cell), num_views, num_x,
                                          num_z, vectors.shape[0]))
        # Both distances are half the smaller detector extent, as in mg18.
        source_dist = min(cell[1], cell[2]) / 2
        model = mbirtorch.TranslationModel(
            tuple(cell), vectors, source_detector_dist=source_dist,
            source_iso_dist=source_dist)
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
    # The shepp-logan builder places its ellipsoids as fractions of the
    # volume, and on a volume only a few voxels deep every one of them can
    # miss, leaving the phantom all zeros.  The --smoke translation cell
    # reconstructs a volume that shallow.  An all-zero phantom forward
    # projects to an all-zero sinogram, so every arm of that cell would time
    # a reconstruction of nothing.  A seeded uniform volume has the same
    # shape and a comparable dynamic range, and the generator row records
    # that it was used.
    phantom_fallback = None
    if float(np.max(phantom)) == 0.0:
        phantom = np.asarray(np.random.RandomState(SEED).rand(*recon_shape),
                             dtype=np.float32)
        phantom_fallback = 'seeded uniform (shepp-logan returned all zeros)'
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
                sino_shape=list(staged.shape),
                phantom_fallback=phantom_fallback)


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
        elif row.get('phantom_fallback'):
            print('    phantom: {}'.format(row['phantom_fallback']))

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

    A cell wins only under the coarse admission rule: the speedup reaches
    ``wf.ADMISSION_MARGIN`` AND clears 1.0x by MORE than that cell's warm
    spread.  The margin is what makes a floor survive hardware and shape
    variation; the spread condition keeps one noisy cell from moving a
    floor.  A win that clears the spread but not the margin is recorded as
    ``thin`` so the printout can say what the margin rounded away.
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
            clears_spread = ratio is not None and ratio - 1.0 > spread
            wins = clears_spread and ratio >= wf.ADMISSION_MARGIN
            rows.append((cell, ratio, spread, wins))
            if wins and winner is None:
                winner = cell
        thin = [cell for cell, ratio, spread, wins in rows
                if not wins and ratio is not None and ratio - 1.0 > spread
                and (winner is None or elements(cell) < elements(winner))]
        out[(family, count)] = dict(against=against, rows=rows, winner=winner,
                                    thin=thin)
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
            if wins:
                label = 'WINS'
            elif ratio - 1.0 > spread and ratio < wf.ADMISSION_MARGIN:
                label = ('thin: clears the spread, under the {:.2f}x '
                         'margin'.format(wf.ADMISSION_MARGIN))
            else:
                label = 'loses'
            print('  {:>20} {:>16,}   {:.3f}x  spread {:.1%}   {}'.format(
                str(cell), elements(cell), ratio, spread, label))
        largest = max(elements(c) for c, _r, _s, _w in record['rows'])
        winner = record['winner']
        proposed = elements(winner) if winner else None
        # A family with no table row is in the same state as a sentinel: no
        # admission size is known for it.  The two are reported separately so
        # the reader can tell a first measurement from a re-measurement.
        if winner is None and record['thin']:
            print('  NO CELL CLEARS THE {:.2f}x MARGIN, but {} clear the '
                  'spread: the coarse rule rounds a thin win up one class, '
                  'and the class above the largest measured cell is '
                  'unmeasured.  This run cannot place the floor; re-run '
                  'with a larger ladder, or a person decides what the row '
                  'becomes.'.format(wf.ADMISSION_MARGIN,
                                    ', '.join(str(c) for c in record['thin'])))
        elif winner is None and floor is None:
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
        if winner is not None and record['thin']:
            print('  the {:.2f}x margin rounded away thin wins at: '
                  '{}.'.format(wf.ADMISSION_MARGIN,
                               ', '.join(str(c) for c in record['thin'])))


_ROW_TEMPLATE = """    ('{family}', {count}): Floor(
        family='{family}', count={count}, elements={elements}, cell={cell},
        against={against},
        bracket=Bracket(losing_cell={low_cell}, losing_speedup={low_x},
                        winning_cell={cell}, winning_speedup={win_x}),
        spread={spread:.4g}, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='{today}', commit='{commit}',
        largest_tested={largest},
        note='...'),"""


def _source_chunks(text, width=60):
    """``text`` split at spaces into pieces at most ``width`` characters,
    each keeping its trailing space, so joining the pieces reproduces the
    text exactly.  Used to print a carried row's note as wrapped source."""
    chunks, current = [], ''
    for word in text.split(' '):
        candidate = word if not current else current + ' ' + word
        if current and len(candidate) > width:
            chunks.append(current + ' ')
            current = word
        else:
            current = candidate
    chunks.append(current)
    assert ''.join(chunks) == text
    return chunks


def _render_carried_row(family, count, floor):
    """One CURRENT table row, printed back in source form.  Every field
    passes through verbatim -- elements, bracket, spread, dates, commit, and
    the note -- so a scoped run's paste never claims a measurement that did
    not happen."""
    gpu = ('MEASURED_GPU' if floor.gpu == wf.MEASURED_GPU
           else repr(floor.gpu))
    config = ('MEASURED_CONFIG' if floor.config == wf.MEASURED_CONFIG
              else repr(floor.config))
    elements_text = ('{:_}'.format(floor.elements)
                     if floor.elements is not None else 'None')
    largest_text = ('{:_}'.format(floor.largest_tested)
                    if floor.largest_tested is not None else 'None')
    lines = [
        "    ('{}', {}): Floor(".format(family, count),
        "        family='{}', count={}, elements={}, cell={},".format(
            family, count, elements_text, floor.cell),
        "        against={},".format(floor.against),
        "        bracket=Bracket(losing_cell={}, losing_speedup={},".format(
            floor.bracket.losing_cell, floor.bracket.losing_speedup),
        "                        winning_cell={}, winning_speedup={}),".format(
            floor.bracket.winning_cell, floor.bracket.winning_speedup),
        "        spread={}, gpu={}, config={},".format(
            floor.spread, gpu, config),
        "        measured={!r}, commit={!r},".format(
            floor.measured, floor.commit),
        "        largest_tested={},".format(largest_text),
    ]
    chunks = _source_chunks(floor.note)
    lines.append("        note={!r}".format(chunks[0])
                 + ('' if len(chunks) == 1 else ''))
    for chunk in chunks[1:]:
        lines.append("             {!r}".format(chunk))
    lines[-1] += '),'
    return '\n'.join(lines)


def print_table(verdicts, commit, carried_families=()):
    """The paste-ready replacement for FLOORS, provenance included.  Measured
    families print from this run's verdicts with their notes left to write;
    carried families print their CURRENT rows verbatim, so the paste is
    always the whole table."""
    today = datetime.date.today().isoformat()
    print('\n' + '=' * 78)
    print('PASTE INTO mbirtorch/_widening_floors.py (FLOORS, then the three')
    print('bound constants printed by --bless).  All of it, or none of it.')
    if carried_families:
        print('measured this run: {}.  carried verbatim (their cost inputs '
              'are unmoved): {}.'.format(
                  ', '.join(sorted({f for f, _c in verdicts})),
                  ', '.join(sorted(carried_families))))
    print('=' * 78 + '\nFLOORS = {')
    # Carried rows keep the CURRENT table's own ordering, printed first so
    # the paste groups this run's measured rows together at the end.
    for (family, count), floor in wf.FLOORS.items():
        if family in carried_families:
            print(_render_carried_row(family, count, floor))
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


def print_bless_paste_help(stale):
    """How to hand-paste the block --bless just printed, and how to check it.

    Nothing here writes _widening_floors.py.  Every value in that file sits
    beside a hand-written comment recording what measured it and when, and a
    writer that rewrote the file would have to preserve those comments or
    silently drop them.  So the paste is by hand, and these are the steps.
    """
    print('\n' + '-' * 78)
    print('HOW TO PASTE')
    print('-' * 78)
    print('In mbirtorch/_widening_floors.py, replace three things with the')
    print('block above:')
    print()
    print('  1. the whole BLESSED_COST_HASHES = {...} dict.  Usually only a')
    print('     few entries differ, but replacing the dict whole is always')
    print('     right and takes the same time;')
    print('  2. the STALE_SINCE = ... line;')
    print('  3. the TABLE_CHECKSUM = ... value.  The file wraps it after the')
    print("     '=' with a backslash; either form is fine.")
    print()
    print('Leave FLOORS alone.  --bless measures nothing, so no row and no')
    print('row note moves.  A full refresh prints its own FLOORS block to')
    print('paste beside these three.')
    print()
    print('Then extend the comment above BLESSED_COST_HASHES to say WHEN you')
    print('re-recorded and WHY.  That comment is the only place a reader can')
    print('learn whether a hash moved because the floors were re-measured or')
    print('because someone edited a comment in a priced file.  No hash')
    print('distinguishes the two, and the difference decides whether the')
    print('measurements above still describe the code.')
    if stale:
        print('  Here it needs to cover: {}.'.format(
            ', '.join(name for name, _w, _g in stale)))
    print()
    print('Finally, check that the paste took:')
    print()
    print('    python -c "import mbirtorch._widening_floors as wf; '
          'print(wf.stale_note())"')
    print('    python -m pytest tests/test_widening_floors.py')
    print()
    print('The first should print None.  A string instead means the hashes')
    print('did not all land.  A test failure naming TABLE_CHECKSUM means the')
    print('checksum and the values it binds disagree, which is exactly what')
    print('that check exists to catch: paste all three, or none of them.')


def do_bless():
    """``--bless``: print fresh values for the three bound constants."""
    stale = wf.stale_cost_inputs()
    print('projection-cost inputs that moved since the hashes were last '
          'recorded: {}'.format(
              ', '.join(name for name, _w, _g in stale) or 'none'))
    if stale:
        print('floor families those inputs price: {}'.format(
            ', '.join(wf.stale_families())))
    print('\nRecording the current hashes after a measurement run.  If you')
    print('have NOT just re-measured, doing this hides a real change from')
    print('the automatic staleness note; leave the old hashes in place')
    print('instead and let the note stand until the floors are')
    print('re-measured.\n')
    print('=' * 78)
    print('PASTE INTO mbirtorch/_widening_floors.py.  All of it, or none of')
    print('it: the checksum binds these three constants together.')
    print('=' * 78)
    print(wf.bless_lines(stale_since=None))
    print_bless_paste_help(stale)
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
    parser.add_argument('--families', nargs='?', const='stale', default=None,
                        metavar='NAMES',
                        help='measure only these floor families (comma-'
                             'separated); with no value, exactly the families '
                             'whose recorded cost inputs moved.  Every other '
                             "family's rows are carried into the paste "
                             'verbatim, and the run refuses a scope that '
                             'would carry a family whose inputs moved')
    args = parser.parse_args(argv)

    # Checked before --bless: the pair used to mean "record the hashes
    # without measuring", and someone typing it from memory should be told it
    # is unnecessary rather than quietly re-record the hashes for real.
    if args.accept_stale:
        return do_accept_stale()
    if args.bless:
        return do_bless()

    scope = resolve_scope(args.families)
    carried = (() if scope is None else
               sorted({family for family, _count in planned_rows()}
                      - set(scope)))
    plan = build_plan(smoke=args.smoke, scope=scope)
    if args.plan:
        print_plan(plan, args.smoke, scope)
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

    print_plan(plan, args.smoke, scope)
    print('\nrunning ...')
    started = time.time()
    measured = run_plan(plan, device)
    verdicts = verdict(plan, measured)
    print_verdict(verdicts)
    print_table(verdicts, head_commit(), carried)
    print('\nthen: python dev_scripts/refresh_widening_floors.py --bless')
    print('elapsed {:.1f} min; rows under {}'.format(
        (time.time() - started) / 60, RESULTS_DIR))
    return 0


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == '_worker':
        _worker_main(sys.argv[2], sys.argv[3])
    else:
        sys.exit(main())
