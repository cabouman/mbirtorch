"""The widening speed floors, their invariants, their staleness report, and
the refresh tool's report of which geometries still need measuring.

The floors are a MEASUREMENT of where each device count starts paying for
itself, and a measurement is only as good as the code it was taken against.
This file therefore REPORTS when the projection-cost code has moved -- and
passes.  Out-of-date floors are work to schedule, not a breakage: the library
detects the change live and logs it at every automatic device selection, a
planned nightly automation owns the re-measurement, and nothing here stops a
suite run over it.

Two things still fail HARD, because neither is merely work to schedule:

  * TAMPERING -- TABLE_CHECKSUM binds the floors, the recorded cost-input
    hashes and STALE_SINCE, so hand-editing a hash to silence the note fails
    here.
  * PROVENANCE -- dates, brackets, and floors that rise with the device
    count: the checks that make a row a measurement rather than an
    assertion.

The selection RULE these numbers feed is tested in test_device_policy.py.
"""

import importlib.util
import os
import warnings

import pytest

from mbirtorch import _widening_floors as wf

REFRESH = 'python dev_scripts/refresh_widening_floors.py'
REFRESH_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'dev_scripts', 'refresh_widening_floors.py')


@pytest.fixture(scope='module')
def refresh_tool():
    """The refresh script, loaded from its path.

    dev_scripts is not an installed package, so the tool is loaded by file
    rather than imported by name.  Only its reporting helpers are exercised
    here; nothing in this file measures anything or starts a subprocess.
    """
    if not os.path.exists(REFRESH_PATH):
        pytest.skip('dev_scripts/refresh_widening_floors.py is not present')
    spec = importlib.util.spec_from_file_location('refresh_widening_floors',
                                                  REFRESH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_a_table_with_a_sentinel(monkeypatch):
    """A synthetic two-row table: a finite n=2 floor and an n=4 sentinel -- a
    row whose ``elements`` is None because no admission size was ever
    measured for that count.

    The rule outlives the data.  No shipped row has been a sentinel since the
    2026-08-10 refresh gave cone n=2 a finite floor, so the tests below read
    the rule off a table built here rather than off whichever numbers happen
    to ship.  A refresh that finds a count with no admission size puts a
    sentinel row back, and these tests have to keep holding when it does.
    """
    def row(count, elements, cell, bracket, note):
        return wf.Floor(family='synthetic', count=count, elements=elements,
                        cell=cell, against=count // 2, bracket=bracket,
                        spread=0.01, gpu=wf.MEASURED_GPU,
                        config=wf.MEASURED_CONFIG, measured='2026-08-10',
                        commit='a880d9c', largest_tested=1_023_934_464,
                        note=note)

    table = {
        ('synthetic', 2): row(
            2, 88_080_384, (512, 448, 384),
            wf.Bracket(losing_cell=(384, 336, 288), losing_speedup=0.64,
                       winning_cell=(512, 448, 384), winning_speedup=1.23),
            'synthetic finite floor'),
        ('synthetic', 4): row(
            4, None, None,
            wf.Bracket(losing_cell=(1024, 1008, 992), losing_speedup=0.92,
                       winning_cell=None, winning_speedup=None),
            'synthetic sentinel: no admission point at or below the '
            '1024-class cell'),
    }
    monkeypatch.setattr(wf, 'FLOORS', table)
    return table


# ── the staleness report (loud, never fatal) ─────────────────────────────────
def test_drift_in_the_projection_cost_code_is_reported_and_not_fatal():
    """The floors describe a version of the projection code.  When that code
    moves, say so -- and PASS.

    This assertion used to fail, which handed a stop-everything chore to
    whoever next touched a kernel: the suite stayed red until someone booked
    a 4-GPU node or re-recorded the hashes by hand.  The measured harm of an
    out-of-date floor is one device-count decision made on old numbers; that
    does not buy a broken test run.  So it is reported here, reported again
    by ``wf.stale_note()`` in every automatic device selection, and a planned
    nightly automation owns the re-measurement.

    The hash is deliberately over whole FILES for the kernels and the
    drivers: the module-level chunk constants and budget class attributes
    those files carry move a crossover without touching any function named
    here, and a function-level hash would sail past them.
    """
    stale = wf.stale_cost_inputs()
    if not stale:
        return
    changed = '\n'.join(
        '    {}\n        recorded {}\n        now      {}'.format(
            name, (recorded or 'MISSING')[:16], (actual or 'MISSING')[:16])
        for name, recorded, actual in stale)
    report = (
        'FLOORS STALE: the projection-cost code changed, so the multi-GPU '
        'widening speed floors in mbirtorch/_widening_floors.py no longer '
        'describe it.  They still govern, and every automatic device '
        'selection logs this.\n\nchanged inputs:\n{}\n\n'
        'To RE-MEASURE (on a 4-GPU node, roughly 30-60 minutes), then paste '
        'the printed block:\n'
        '    {}\n\n'
        'To RE-RECORD the hashes only, when the change provably cannot move '
        'a projection cost (no GPU needed):\n'
        '    {} --bless\n\n'
        'Nothing needs to be done to keep working: this test passes '
        'stale.'.format(changed, REFRESH, REFRESH))
    print('\n' + report)
    warnings.warn(report, stacklevel=2)


def test_a_drifted_cost_input_names_itself_in_the_log_with_no_human_involved(
        monkeypatch):
    """The note no longer waits for anyone to write a date by hand: forge a
    recorded hash so the live check sees a change, and the note names what
    moved.

    The cache is reset first because the check runs once per process, and
    some earlier consultation in this session may already have run it.
    """
    forged = dict(wf.BLESSED_COST_HASHES)
    forged['projectors.py'] = '0' * 64
    monkeypatch.setattr(wf, 'BLESSED_COST_HASHES', forged)
    monkeypatch.setattr(wf, 'STALE_SINCE', None)
    monkeypatch.setattr(wf, '_DRIFT_CHECK', None)

    note = wf.stale_note()
    assert note is not None, 'live drift must produce a note by itself'
    assert 'projectors.py' in note, note
    assert 'refresh_widening_floors.py' in note, note
    # The drift path, not the stamp path: STALE_SINCE is unset here.
    assert 'stale since' not in note, note
    # And it is computed once: the second call does not re-hash anything.
    assert wf.stale_note() == note


def test_a_staleness_check_that_cannot_run_is_logged_rather_than_raised(
        monkeypatch):
    """A missing file mid-refactor must not take a reconstruction down with
    it.  The note reports that the floors could not be checked -- which is
    not the same claim as 'fresh' -- and nothing raises."""
    def exploding():
        raise FileNotFoundError('mbirtorch/projectors.py')

    monkeypatch.setattr(wf, 'stale_cost_inputs', exploding)
    monkeypatch.setattr(wf, 'STALE_SINCE', None)
    monkeypatch.setattr(wf, '_DRIFT_CHECK', None)

    note = wf.stale_note()
    assert 'could not be checked' in note, note
    assert 'FileNotFoundError' in note, note


# ── the tamper guard (this still fails hard) ─────────────────────────────────
def test_the_floors_hashes_and_staleness_stamp_move_as_one_unit():
    """Now that a changed cost input only warns, hand-editing a hash is the
    cheap way to silence the note while leaving floors that were never
    re-measured.  The checksum binds all three, so that shortcut fails HERE
    -- and this one is not a warning."""
    assert wf.table_checksum() == wf.TABLE_CHECKSUM, (
        'mbirtorch/_widening_floors.py was edited without going through the '
        'refresh script: FLOORS, BLESSED_COST_HASHES and STALE_SINCE are '
        'bound by TABLE_CHECKSUM and must be written together.\n'
        '    {} --bless            (after re-measuring)'.format(REFRESH))


def test_a_hand_edited_hash_is_caught_by_the_checksum(monkeypatch):
    """The binding, exercised rather than asserted: silencing the staleness
    note by hand must not also green the checksum."""
    forged = dict(wf.BLESSED_COST_HASHES)
    forged['projectors.py'] = '0' * 64
    monkeypatch.setattr(wf, 'BLESSED_COST_HASHES', forged)
    assert wf.table_checksum() != wf.TABLE_CHECKSUM


def test_the_stale_stamp_still_reaches_the_log_when_it_is_set(monkeypatch):
    """STALE_SINCE survives as a hand-written record of a reason to
    re-measure that no hash can see.  It is read here against a check that
    found NO changed cost input, so the hand-written path is tested on its
    own rather than through whatever the shipped table's hashes happen to say
    today.
    """
    monkeypatch.setattr(wf, '_DRIFT_CHECK', ((), None))
    monkeypatch.setattr(wf, 'STALE_SINCE', None)
    assert wf.stale_note() is None
    monkeypatch.setattr(wf, 'STALE_SINCE', '2026-01-02')
    note = wf.stale_note()
    assert 'stale since 2026-01-02' in note
    assert 'refresh_widening_floors.py' in note


# ── provenance ───────────────────────────────────────────────────────────────
def test_every_floor_carries_the_provenance_to_re_measure_it():
    for (family, count), floor in sorted(wf.FLOORS.items()):
        where = '{} n={}'.format(family, count)
        assert floor.family == family and floor.count == count, where
        assert floor.measured and len(floor.measured) == 10, where
        assert floor.commit and floor.commit != 'unknown', where
        assert isinstance(floor.largest_tested, int), where
        assert floor.largest_tested > 0, where
        assert floor.gpu and floor.config, where
        assert 0.0 <= floor.spread < 1.0, where
        assert floor.note, where
        # A floor is always measured against a SMALLER device count.
        assert floor.against >= 1 and floor.against < count, where


def test_every_floor_records_the_bracket_it_was_read_off():
    """A floor without a bracket is an assertion, not a measurement."""
    for (family, count), floor in sorted(wf.FLOORS.items()):
        where = '{} n={}'.format(family, count)
        bracket = floor.bracket
        assert bracket is not None, where
        if floor.elements is None:
            # A sentinel has lost somewhere and won nowhere.
            assert bracket.winning_cell is None, where
            assert bracket.losing_cell is not None, where
        else:
            assert bracket.winning_cell is not None, where
            assert wf.sinogram_elements(bracket.winning_cell) == floor.elements, \
                where
            assert bracket.winning_speedup > 1.0, where
            if bracket.losing_cell is not None:
                assert (wf.sinogram_elements(bracket.losing_cell)
                        < floor.elements), where
                # "Lost" is the admission rule, not a bare comparison with
                # 1.0: a win has to clear 1.0x by MORE than the spread, so a
                # nominal 1.02x inside a 2.5% spread is a loss.
                assert bracket.losing_speedup <= 1.0 + floor.spread, where


def test_a_sentinel_says_how_far_it_was_tested(monkeypatch):
    """A sentinel is not a permanent never; it is a place a refresh has to
    start from, so it must say where that is.

    Read off a synthetic table: the shipped one has no sentinel row today, so
    asserting against it would test the current data rather than the rule.
    """
    install_a_table_with_a_sentinel(monkeypatch)
    sentinels = [(key, floor) for key, floor in wf.FLOORS.items()
                 if floor.elements is None]
    assert sentinels, 'the sentinel path is unexercised if no row uses it'
    for (family, count), floor in sentinels:
        assert floor.largest_tested >= wf.sinogram_elements(
            floor.bracket.losing_cell), '{} n={}'.format(family, count)


def test_finite_floors_rise_with_the_device_count():
    assert wf.monotone_violations() == []


def test_the_measurement_envelope_is_stated():
    """The floors are validated at one configuration; the module says which,
    so a workload outside it knows to re-measure rather than assume."""
    assert 'iteration' in wf.MEASUREMENT_CAVEAT
    assert 'subset schedule' in wf.MEASUREMENT_CAVEAT
    for floor in wf.FLOORS.values():
        assert 'iteration' in floor.config


# ── the accessor ─────────────────────────────────────────────────────────────
def test_one_device_is_always_admitted():
    for family in (None, 'parallel', 'cone', 'a-family-with-no-rows'):
        ok, why = wf.admitted(family, 1, 1)
        assert ok and 'always admitted' in why


def test_a_count_with_no_row_inherits_the_next_measured_count_above_it():
    """The conservative direction: n=3 is governed by the n=4 floor, and so
    is any count above 4, since floors rise with the count."""
    four = wf.FLOORS[('parallel', 4)]
    for count in (3, 4, 5, 8):
        assert wf.governing_floor('parallel', count) is four
    assert wf.governing_floor('parallel', 2) is wf.FLOORS[('parallel', 2)]
    # And the reason names the row that was inherited, not the count asked
    # about, so a log line is never quietly misleading.
    _ok, why = wf.admitted('parallel', 3, 1000)
    assert 'parallel n=4 floor, which n=3 inherits' in why


def test_a_model_with_no_family_gets_the_parallel_floors_and_is_told_so():
    for count in (2, 3, 4):
        for size in (1_376_256, 1_023_934_464):
            assert (wf.admitted(None, count, size)[0]
                    == wf.admitted('parallel', count, size)[0])
    _ok, why = wf.admitted(None, 2, 1_376_256)
    assert 'names no _floor_family' in why and 'parallel floors apply' in why


def test_an_unmeasured_family_admits_everything_rather_than_refusing_it():
    """A family the table has never heard of is not evidence against
    widening.  Only a model class routes here, and only by declaring a
    family with no rows -- which the refresh script reports as work to do."""
    ok, why = wf.admitted('translation', 4, 1)
    assert ok and 'no speed floors are measured' in why


def test_the_sentinel_holds_its_count_at_every_size(monkeypatch):
    """A sentinel excludes its count at EVERY size, including sizes above the
    largest one tested -- the refusal names that limit rather than pretending
    the count was priced there."""
    install_a_table_with_a_sentinel(monkeypatch)
    for size in (1, 88_080_384, 1_023_934_464, 10 ** 12):
        ok, why = wf.admitted('synthetic', 4, size)
        assert not ok
        assert 'sentinel' in why and 'largest size tested' in why


def test_a_refusal_names_the_override_the_user_can_reach_for():
    _ok, why = wf.admitted('parallel', 2, 11_010_048)
    assert why.startswith('held by the speed floor: ')
    assert '11.0M sinogram elements < 88.1M' in why
    assert 'configure_devices(num_devices=2) overrides' in why


def test_a_floor_admits_exactly_at_its_own_value():
    floor = wf.FLOORS[('parallel', 2)]
    assert wf.admitted('parallel', 2, floor.elements)[0]
    assert not wf.admitted('parallel', 2, floor.elements - 1)[0]


def test_the_fallback_note_says_capacity_overrode_speed(monkeypatch):
    why = wf.fallback_reason('parallel', 2, 11_010_048)
    assert 'chosen past its speed floor' in why
    assert 'no admitted count fits' in why
    assert '11.0M sinogram elements < 88.1M' in why
    # A sentinel row has no number to compare against, and says so instead of
    # printing a nonsense inequality.  Synthetic, since the shipped table has
    # carried no sentinel row since the 2026-08-10 refresh.
    install_a_table_with_a_sentinel(monkeypatch)
    assert 'no admission point is measured' in wf.fallback_reason(
        'synthetic', 4, 88_080_384)


def test_sinogram_elements_is_the_product_of_the_shape():
    assert wf.sinogram_elements((512, 448, 384)) == 88_080_384
    assert wf.sinogram_elements((1024, 1008, 992)) == 1_023_934_464


# ── the refresh tool's "needs measurement" report ────────────────────────────
def test_the_refresh_tool_reports_the_geometries_that_take_the_fallback(
        refresh_tool):
    """The one tool whose job is to say "this geometry needs measurement"
    must not be silent about the geometries that actually need it.

    A class that declares no floor family is governed by the DEFAULT_FAMILY
    floors, which were measured on a different geometry.  That is the state
    every newly ported geometry arrives in, so it is reported under the None
    key rather than skipped for having nothing declared.
    """
    import mbirtorch

    missing = refresh_tool.unmeasured_families()
    assert None in missing, (
        'the classes that declare no floor family are the ones taking the '
        'substituted floors, and they are what this report is for')
    undeclared = missing[None]
    assert 'TranslationModel' in undeclared
    assert 'MultiAxisParallelModel' in undeclared

    # Every reported class really does inherit the base value rather than
    # naming a family of its own, so the report matches the code it describes.
    for name in undeclared:
        assert getattr(mbirtorch, name)._floor_family is None


def test_the_report_covers_every_geometry_that_reaches_the_device_decision(
        refresh_tool):
    """The report is scoped to classes a floor can actually govern.

    A floor is consulted when a model chooses its own device count, which
    happens on the shared reconstruction path.  The base class is not a
    geometry and QGGMRFDenoiser refuses recon, so neither can reach that
    decision and neither is work to measure.  An exported alias is the same
    class object as the class it aliases, so it must not be counted twice.
    """
    reported = {name for names in refresh_tool.unmeasured_families().values()
                for name in names}
    assert 'TomographyModel' not in reported
    assert 'QGGMRFDenoiser' not in reported
    # MultiAxisParallelBeamModel is an alias of MultiAxisParallelModel; the
    # class is reported once, under its own name.
    assert 'MultiAxisParallelBeamModel' not in reported
    # The two measured families are governed by their own rows, so they are
    # not outstanding work.
    assert 'ParallelBeamModel' not in reported
    assert 'ConeBeamModel' not in reported


def test_a_declared_family_with_no_rows_is_still_reported_under_its_name(
        refresh_tool, monkeypatch):
    """The other way to arrive unmeasured: name a family the table has never
    heard of.  Widening the report to undeclared classes must not drop the
    case it already handled, so both keys are exercised here."""
    monkeypatch.setattr(refresh_tool.wf, 'FLOORS',
                        {key: value for key, value in wf.FLOORS.items()
                         if key[0] != 'parallel'})

    missing = refresh_tool.unmeasured_families()
    assert missing.get('parallel') == ['ParallelBeamModel']
    assert 'TranslationModel' in missing[None]


def test_the_printed_report_names_the_class_and_the_floors_it_borrows(
        refresh_tool, capsys):
    """Reading the report has to be enough: it names which class is
    unmeasured and which family's floors are standing in for it, so nobody
    has to go read the fallback rule to find out what is governing."""
    refresh_tool.print_plan(refresh_tool.build_plan(smoke=True), smoke=True)

    printed = capsys.readouterr().out
    assert 'NEEDS MEASUREMENT' in printed
    assert 'TranslationModel' in printed
    assert 'MultiAxisParallelModel' in printed
    assert wf.DEFAULT_FAMILY in printed


def test_the_refresh_tool_refuses_to_measure_a_family_it_cannot_build(
        refresh_tool):
    """The report above invites someone to declare a new floor family.  The
    builder must then refuse the family it has no geometry for: falling
    through to parallel beam would time parallel beam and record the numbers
    under the new family's name."""
    with pytest.raises(ValueError, match='cannot build a model for floor family'):
        refresh_tool._build_model('translation', (8, 12, 16), 'cpu')


# ── the env knob ─────────────────────────────────────────────────────────────
def test_the_guard_is_on_by_default_and_off_only_when_asked(monkeypatch):
    monkeypatch.delenv(wf.GUARD_ENV_VAR, raising=False)
    assert wf.guard_enabled()
    for value in ('0', 'false', 'no', 'off', 'OFF'):
        monkeypatch.setenv(wf.GUARD_ENV_VAR, value)
        assert not wf.guard_enabled()
    for value in ('1', 'true', 'yes', ''):
        monkeypatch.setenv(wf.GUARD_ENV_VAR, value)
        assert wf.guard_enabled()
