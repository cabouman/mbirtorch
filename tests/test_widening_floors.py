"""The widening speed floors, their tamper guard, and the admission rule they
feed.

The floors are a MEASUREMENT of where each device count starts paying for
itself.  Two things fail HARD here:

  * TAMPERING -- TABLE_CHECKSUM binds the floors, the recorded cost-input
    hashes and STALE_SINCE, so hand-editing a hash to silence the staleness
    note fails here.
  * PROVENANCE -- floors that rise with the device count: the check that
    makes a row a measurement rather than an assertion.

A staleness check that cannot run is logged rather than raised, so a missing
file mid-refactor never takes a reconstruction down with it.

The selection RULE these numbers feed is tested in test_device_policy.py.
"""

from mbirtorch import _widening_floors as wf

REFRESH = 'python dev_scripts/refresh_widening_floors.py'


def install_a_table_with_a_sentinel(monkeypatch):
    """A synthetic two-row table: a finite n=2 floor and an n=4 sentinel -- a
    row whose ``elements`` is None because no admission size was ever
    measured for that count.

    The rule outlives the data.  The shipped table's sentinel rows come and
    go with refreshes (six ship today), so the tests below read the rule off
    a table built here rather than off whichever numbers happen to ship.
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
def test_the_floors_hashes_and_staleness_stamp_move_as_one_unit(monkeypatch):
    """Now that a changed cost input only warns, hand-editing a hash is the
    cheap way to silence the note while leaving floors that were never
    re-measured.  The checksum binds all three, so that shortcut fails HERE
    -- and this one is not a warning.  The binding is then exercised rather
    than asserted: forging a recorded hash must move the checksum."""
    assert wf.table_checksum() == wf.TABLE_CHECKSUM, (
        'mbirtorch/_widening_floors.py was edited without going through the '
        'refresh script: FLOORS, BLESSED_COST_HASHES and STALE_SINCE are '
        'bound by TABLE_CHECKSUM and must be written together.\n'
        '    {} --bless            (after re-measuring)'.format(REFRESH))

    forged = dict(wf.BLESSED_COST_HASHES)
    forged['projectors.py'] = '0' * 64
    monkeypatch.setattr(wf, 'BLESSED_COST_HASHES', forged)
    assert wf.table_checksum() != wf.TABLE_CHECKSUM


# ── provenance ───────────────────────────────────────────────────────────────
def test_finite_floors_rise_with_the_device_count():
    assert wf.monotone_violations() == []


# ── the accessor ─────────────────────────────────────────────────────────────
def test_multiaxis_floors_split_by_count():
    """Multiaxis admits two devices from the 512-class and four only from
    the 1024-class, and refuses both below their floors.

    The history in brief.  Both rows were sentinels until 2026-08-20,
    when the recompile remedy cleared them and the mg48 refresh placed
    both floors at the 512-class.  The Triton multiaxis kernels then cut
    the one-device wall about four-fold, and the mg56 refresh on that
    tree (with the mg55 1024-class reading) kept the two-device floor at
    the 512-class and RAISED the four-device floor to the 1024-class:
    small cells no longer amortize a four-way fan-out when one device is
    this fast.  The rows' notes and multigpu_findings.md sections 1.45
    and 1.46 in the plans repository carry the measurements.  Counts
    with no row of their own (3, and anything above 4) inherit the n=4
    floor.
    """
    import mbirtorch

    assert mbirtorch.MultiAxisParallelModel._floor_family == 'multiaxis'
    at_the_512_class = wf.sinogram_elements((512, 448, 384))
    at_the_1024_class = wf.sinogram_elements((1024, 1008, 992))
    below_every_floor = wf.sinogram_elements((384, 336, 288))
    ok, why = wf.admitted('multiaxis', 2, at_the_512_class)
    assert ok, why
    for count in (2, 3, 4, 8):
        ok, why = wf.admitted('multiaxis', count, at_the_1024_class)
        assert ok, (count, why)
    for count in (3, 4, 8):
        ok, why = wf.admitted('multiaxis', count, at_the_512_class)
        assert not ok, (count, why)
        assert 'held by the speed floor' in why, why
    for count in (2, 3, 4, 8):
        ok, why = wf.admitted('multiaxis', count, below_every_floor)
        assert not ok, (count, why)
        assert 'held by the speed floor' in why, why
        assert 'configure_devices' in why, why


def test_the_admitted_rule_for_counts_families_and_sentinels(monkeypatch):
    """The admitted() rule in one place.

    One device always passes.  A count with no row of its own inherits the
    next measured count above it, and the reason names the row it inherited
    rather than the count asked about.  A model class that declares no family
    is governed by the parallel floors and is told so.  A family the table has
    never heard of admits everything rather than refusing it.  A sentinel row
    holds its count at every size, including sizes above the largest one
    tested, and the refusal names that limit.
    """
    for family in (None, 'parallel', 'cone', 'a-family-with-no-rows'):
        ok, why = wf.admitted(family, 1, 1)
        assert ok and 'always admitted' in why

    # The conservative direction: n=3 is governed by the n=4 floor, and so
    # is any count above 4, since floors rise with the count.
    four = wf.FLOORS[('parallel', 4)]
    for count in (3, 4, 5, 8):
        assert wf.governing_floor('parallel', count) is four
    assert wf.governing_floor('parallel', 2) is wf.FLOORS[('parallel', 2)]
    _ok, why = wf.admitted('parallel', 3, 1000)
    assert 'parallel n=4 floor, which n=3 inherits' in why

    for count in (2, 3, 4):
        for size in (1_376_256, 1_023_934_464):
            assert (wf.admitted(None, count, size)[0]
                    == wf.admitted('parallel', count, size)[0])
    _ok, why = wf.admitted(None, 2, 1_376_256)
    assert 'names no _floor_family' in why and 'parallel floors apply' in why

    # Only a model class routes to an unmeasured family, and only by
    # declaring a family with no rows -- which the refresh script reports as
    # work to do.  The name below is invented: every shipped family has rows.
    ok, why = wf.admitted('a-family-with-no-rows', 4, 1)
    assert ok and 'no speed floors are measured' in why

    # The sentinel is read off a synthetic table, so the assertion tests the
    # rule rather than whichever rows happen to be sentinels in the shipped
    # table.  This replaces FLOORS, so it comes last.
    install_a_table_with_a_sentinel(monkeypatch)
    for size in (1, 88_080_384, 1_023_934_464, 10 ** 12):
        ok, why = wf.admitted('synthetic', 4, size)
        assert not ok
        assert 'sentinel' in why and 'largest size tested' in why


def test_a_floor_admits_exactly_at_its_own_value():
    floor = wf.FLOORS[('parallel', 2)]
    assert wf.admitted('parallel', 2, floor.elements)[0]
    assert not wf.admitted('parallel', 2, floor.elements - 1)[0]
