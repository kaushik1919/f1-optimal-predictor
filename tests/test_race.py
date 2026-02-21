"""Tests for Phase 3 / 11A: multi-car stochastic race simulator."""

import numpy as np

from f1_engine.core.car import Car
from f1_engine.core.driver import Driver
from f1_engine.core.race import (
    _PASS_TIME_DELTA,
    RaceResult,
    _apply_overtakes,
    _DriverState,
    simulate_race,
)
from f1_engine.core.team import Team
from f1_engine.core.track import Track

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_track() -> Track:
    return Track(
        name="Test Circuit",
        straight_ratio=0.6,
        overtake_coefficient=0.5,
        energy_harvest_factor=0.7,
        tyre_degradation_factor=0.05,
        downforce_sensitivity=0.50,
    )


def _make_team(name: str, base_speed: float, reliability: float = 0.98) -> Team:
    car = Car(
        team_name=name,
        base_speed=base_speed,
        ers_efficiency=0.80,
        aero_efficiency=0.85,
        tyre_wear_rate=1.0,
        reliability=reliability,
    )
    drivers = [
        Driver(
            name=f"{name}_D1",
            team_name=name,
            skill_offset=0.0,
            consistency=1.0,
        ),
        Driver(
            name=f"{name}_D2",
            team_name=name,
            skill_offset=0.0,
            consistency=1.0,
        ),
    ]
    return Team(name=name, car=car, drivers=drivers)


def _sample_teams(n: int = 4) -> list[Team]:
    """Return *n* teams with slightly varying performance."""
    return [_make_team(f"Team_{i}", base_speed=80.0 + i * 0.3) for i in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_race_runs() -> None:
    """Race must return a classification with length equal to total drivers."""
    track = _sample_track()
    teams = _sample_teams(4)
    result = simulate_race(track, teams, laps=5, seed=42)
    assert isinstance(result, RaceResult)
    assert len(result.final_classification) == len(teams) * 2


def test_seed_determinism() -> None:
    """Same seed must produce identical results."""
    track = _sample_track()
    teams = _sample_teams(4)
    r1 = simulate_race(track, teams, laps=10, noise_std=0.1, seed=123)
    r2 = simulate_race(track, teams, laps=10, noise_std=0.1, seed=123)
    assert r1.final_classification == r2.final_classification
    assert r1.dnf_list == r2.dnf_list
    assert r1.lap_times == r2.lap_times


def test_zero_noise_deterministic() -> None:
    """With noise_std=0, lap times must be identical across two runs."""
    track = _sample_track()
    teams = _sample_teams(3)
    r1 = simulate_race(track, teams, laps=8, noise_std=0.0, seed=99)
    r2 = simulate_race(track, teams, laps=8, noise_std=0.0, seed=99)
    for name in r1.lap_times:
        assert (
            r1.lap_times[name] == r2.lap_times[name]
        ), f"lap times for {name} must be identical with zero noise"


def test_dnf_occurs_with_low_reliability() -> None:
    """Very low reliability should produce at least one DNF across runs."""
    track = _sample_track()
    fragile_teams = [
        _make_team(f"Fragile_{i}", base_speed=80.0, reliability=0.05) for i in range(3)
    ]
    dnf_seen = False
    for attempt in range(10):
        result = simulate_race(
            track, fragile_teams, laps=20, noise_std=0.05, seed=attempt
        )
        if result.dnf_list:
            dnf_seen = True
            break
    assert dnf_seen, "low reliability must produce at least one DNF"


def test_classification_contains_all_drivers() -> None:
    """Every driver must appear exactly once in the final classification."""
    track = _sample_track()
    teams = _sample_teams(5)
    result = simulate_race(track, teams, laps=10, seed=7)
    driver_names = {drv.name for team in teams for drv in team.drivers}
    assert set(result.final_classification) == driver_names


def test_lap_times_dict_keys_match_drivers() -> None:
    """The lap_times dict must have an entry for every driver."""
    track = _sample_track()
    teams = _sample_teams(4)
    result = simulate_race(track, teams, laps=6, seed=55)
    for team in teams:
        for drv in team.drivers:
            assert drv.name in result.lap_times


# ---------------------------------------------------------------------------
# Phase 11A: persistent overtake tests
# ---------------------------------------------------------------------------


def _make_driver_state(name: str, cumulative: float, last_lap: float) -> _DriverState:
    """Build a minimal _DriverState for overtake unit tests."""
    car = Car(
        team_name="T",
        base_speed=80.0,
        ers_efficiency=0.80,
        aero_efficiency=0.85,
        tyre_wear_rate=1.0,
        reliability=0.99,
    )
    drv = Driver(name=name, team_name="T", skill_offset=0.0, consistency=1.0)
    ds = _DriverState(drv, car, deploy_level=0.5, harvest_level=0.5)
    ds.cumulative_time = cumulative
    ds.last_lap_time = last_lap
    return ds


def test_overtake_changes_cumulative_time() -> None:
    """A successful overtake must adjust cumulative times by pass_time_delta.

    We set up two drivers separated by a tiny gap (< 1.0 s) and use a
    large positive delta (trailer slower) to drive pass_prob close to 1.0
    via the logistic model.  After _apply_overtakes the former trailer
    must be swapped ahead with adjusted cumulative times.
    """
    track = _sample_track()
    leader = _make_driver_state("Leader", cumulative=100.0, last_lap=80.0)
    # Trailer is slower (higher last_lap) → delta = 85 - 80 = +5
    # exponent = -3 * 5 * 0.5 = -7.5 → pass_prob ≈ 0.9994
    trailer = _make_driver_state("Trailer", cumulative=100.3, last_lap=85.0)

    ranked = [leader, trailer]
    rng = np.random.default_rng(0)
    _apply_overtakes(ranked, track, rng)

    # After the pass, ranked[0] should be the former trailer.
    assert ranked[0].driver.name == "Trailer"
    assert ranked[1].driver.name == "Leader"

    # Cumulative time of overtaker must be strictly less than overtaken.
    assert ranked[0].cumulative_time < ranked[1].cumulative_time

    # The time adjustment should match the pass_time_delta constant.
    expected_trailer_time = 100.0 - _PASS_TIME_DELTA  # placed ahead of old leader
    expected_leader_time = 100.0 + _PASS_TIME_DELTA  # pushed back
    assert abs(ranked[0].cumulative_time - expected_trailer_time) < 1e-9
    assert abs(ranked[1].cumulative_time - expected_leader_time) < 1e-9


def test_no_instant_reswap() -> None:
    """After an overtake the skip-next logic must prevent immediate re-swap.

    With three drivers A-B-C where all adjacent pairs have high pass_prob,
    if B passes A the loop must skip the (now A, C) comparison on the same
    iteration, so C keeps its original position.
    """
    track = _sample_track()
    a = _make_driver_state("A", cumulative=100.0, last_lap=80.0)
    # B slower → delta = +5 → pass_prob ≈ 0.9994, triggers swap with A.
    b = _make_driver_state("B", cumulative=100.1, last_lap=85.0)
    # C slower → if (A, C) were compared, would also trigger, but
    # the skip-next after (A, B) swap should prevent it.
    c = _make_driver_state("C", cumulative=100.2, last_lap=86.0)

    ranked = [a, b, c]
    rng = np.random.default_rng(0)
    _apply_overtakes(ranked, track, rng)

    # B passes A → ranked becomes [B, A, C].
    # The next comparison (A, C) is skipped, so C stays at index 2.
    assert ranked[0].driver.name == "B"
    assert ranked[1].driver.name == "A"
    assert ranked[2].driver.name == "C"


def test_time_order_consistent_after_pass() -> None:
    """Persistent overtakes must produce a deterministic, self-consistent race.

    This verifies the integration of persistent overtakes into the full
    simulate_race pipeline.  The same seed must always produce the same
    classification, and every finisher/DNF must appear exactly once.
    With persistent time adjustment the classification is sorted by
    the adjusted cumulative time (which includes pass deltas), not
    by the raw sum of lap times.
    """
    track = _sample_track()
    teams = _sample_teams(5)
    r1 = simulate_race(track, teams, laps=15, noise_std=0.1, seed=77)
    r2 = simulate_race(track, teams, laps=15, noise_std=0.1, seed=77)

    # Determinism: identical seed must give identical results.
    assert r1.final_classification == r2.final_classification
    assert r1.dnf_list == r2.dnf_list

    # Completeness: every driver appears exactly once.
    all_drivers = {drv.name for team in teams for drv in team.drivers}
    assert set(r1.final_classification) == all_drivers
    assert len(r1.final_classification) == len(all_drivers)

    # Finishers recorded non-empty lap lists; DNFs may have partial lists.
    finishers = [name for name in r1.final_classification if name not in r1.dnf_list]
    for name in finishers:
        assert len(r1.lap_times[name]) == 15
