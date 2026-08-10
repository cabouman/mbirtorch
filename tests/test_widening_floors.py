"""The widening speed floors, and the tripwire that keeps them honest.

The floors are a MEASUREMENT of where each device count starts paying for
itself.  A measurement is only as good as the code it was taken against, so
this file's first job is to fail the moment the projection-cost code moves --
loudly, with the one command that re-measures it.  Its second job is to hold
the table to its own invariants: every row carries its provenance, and finite
floors rise with the device count.

The selection RULE these numbers feed is tested in test_device_policy.py.
"""

import pytest

from mbirtorch import _widening_floors as wf

REFRESH = 'python dev_scripts/refresh_widening_floors.py'


# ── the tripwire ─────────────────────────────────────────────────────────────
def test_the_projection_cost_code_still_matches_the_measured_floors():
    """The floors describe a version of the projection code.  When that code
    changes, the floors are stale until someone says otherwise.

    This is deliberately a hash over whole FILES for the kernels and the
    drivers: the module-level chunk constants and budget class attributes
    those files carry move a crossover without touching any function named
    here, and a function-level hash would sail past them.
    """
    stale = wf.stale_cost_inputs()
    if stale:
        changed = '\n'.join(
            '    {}\n        blessed {}\n        now     {}'.format(
                name, (blessed or 'MISSING')[:16], (actual or 'MISSING')[:16])
            for name, blessed, actual in stale)
        pytest.fail(
            'the projection-cost code changed, so the multi-GPU widening '
            'speed floors in mbirtorch/_widening_floors.py may no longer '
            'describe it.\n\nchanged inputs:\n{}\n\n'
            'To RE-MEASURE (on a 4-GPU node, roughly 30-60 minutes):\n'
            '    {}\n\n'
            'To RE-BLESS the hashes only, when the change provably cannot '
            'move a projection cost (no GPU needed):\n'
            '    {} --bless\n\n'
            'To acknowledge the debt WITHOUT re-measuring -- the tests pass '
            'and every automatic device selection logs the staleness:\n'
            '    {} --bless --accept-stale'.format(changed, REFRESH, REFRESH,
                                                   REFRESH))


def test_the_floors_hashes_and_staleness_stamp_move_as_one_unit():
    """The cheap way to green the test above is to paste a fresh hash in by
    hand, leaving floors that were never re-measured.  The checksum binds all
    three, so that shortcut fails HERE instead."""
    assert wf.table_checksum() == wf.TABLE_CHECKSUM, (
        'mbirtorch/_widening_floors.py was edited without going through the '
        'refresh script: FLOORS, BLESSED_COST_HASHES and STALE_SINCE are '
        'bound by TABLE_CHECKSUM and must be written together.\n'
        '    {} --bless            (after re-measuring)\n'
        '    {} --bless --accept-stale   (to record the debt instead)'.format(
            REFRESH, REFRESH))


def test_a_hand_edited_hash_is_caught_by_the_checksum(monkeypatch):
    """The binding, exercised rather than asserted: greening the tripwire by
    hand must not also green the checksum."""
    forged = dict(wf.BLESSED_COST_HASHES)
    forged['projectors.py'] = '0' * 64
    monkeypatch.setattr(wf, 'BLESSED_COST_HASHES', forged)
    assert wf.table_checksum() != wf.TABLE_CHECKSUM


def test_the_stale_stamp_reaches_the_log_when_it_is_set(monkeypatch):
    """An acknowledged-stale table passes the tests and pays for it in the
    device-selection log, which is the whole point of the trade.

    Both states are set here rather than read off the shipped table: an
    assertion that today's table is fresh would fail the moment anyone used
    the documented ``--bless --accept-stale`` path, which is the one path
    this test exists to bless.
    """
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
        # The comparison count is the floor_4 rule made explicit.
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


def test_a_sentinel_says_how_far_it_was_tested():
    """A sentinel is not a permanent never; it is a place a refresh has to
    start from, so it must say where that is."""
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


def test_the_sentinel_holds_its_count_at_every_size():
    for size in (1, 88_080_384, 1_023_934_464, 10 ** 12):
        ok, why = wf.admitted('cone', 2, size)
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


def test_the_fallback_note_says_capacity_overrode_speed():
    why = wf.fallback_reason('parallel', 2, 11_010_048)
    assert 'chosen past its speed floor' in why
    assert 'no admitted count fits' in why
    assert '11.0M sinogram elements < 88.1M' in why
    # A sentinel row has no number to compare against, and says so instead of
    # printing a nonsense inequality.
    assert 'no admission point is measured' in wf.fallback_reason(
        'cone', 2, 88_080_384)


def test_sinogram_elements_is_the_product_of_the_shape():
    assert wf.sinogram_elements((512, 448, 384)) == 88_080_384
    assert wf.sinogram_elements((1024, 1008, 992)) == 1_023_934_464


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
