"""Tests for Phase 3: multi-car stochastic race simulator."""

from f1_engine.core.car import Car
from f1_engine.core.driver import Driver
from f1_engine.core.race import RaceResult, simulate_race
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
