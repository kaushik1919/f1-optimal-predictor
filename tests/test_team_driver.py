"""Tests for Phase 10: two-driver-per-team modelling with WDC and WCC."""

import pytest

from f1_engine.core.car import Car
from f1_engine.core.driver import Driver
from f1_engine.core.race import simulate_race
from f1_engine.core.season import simulate_season_monte_carlo
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


def _make_team(
    name: str,
    base_speed: float = 80.0,
    skill_offsets: tuple[float, float] = (0.0, 0.0),
    consistencies: tuple[float, float] = (1.0, 1.0),
    reliability: float = 0.98,
) -> Team:
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
            skill_offset=skill_offsets[0],
            consistency=consistencies[0],
        ),
        Driver(
            name=f"{name}_D2",
            team_name=name,
            skill_offset=skill_offsets[1],
            consistency=consistencies[1],
        ),
    ]
    return Team(name=name, car=car, drivers=drivers)


def _sample_teams(n: int = 4) -> list[Team]:
    return [_make_team(f"Team_{i}", base_speed=80.0 + i * 0.3) for i in range(n)]


def _mini_calendar() -> list[Track]:
    base = dict(
        straight_ratio=0.6,
        overtake_coefficient=0.5,
        energy_harvest_factor=0.7,
        tyre_degradation_factor=0.05,
        downforce_sensitivity=0.50,
    )
    return [
        Track(name="Track_A", **base),
        Track(name="Track_B", **base),
        Track(name="Track_C", **base),
    ]


# ---------------------------------------------------------------------------
# Team model tests
# ---------------------------------------------------------------------------


def test_team_requires_two_drivers() -> None:
    """Team must reject driver counts other than 2."""
    car = Car(
        team_name="Solo",
        base_speed=80.0,
        ers_efficiency=0.8,
        aero_efficiency=0.85,
        tyre_wear_rate=1.0,
        reliability=0.95,
    )
    one_driver = [
        Driver(name="D1", team_name="Solo", skill_offset=0.0, consistency=1.0)
    ]
    with pytest.raises(ValueError, match="exactly 2 drivers"):
        Team(name="Solo", car=car, drivers=one_driver)

    three_drivers = [
        Driver(name=f"D{i}", team_name="Solo", skill_offset=0.0, consistency=1.0)
        for i in range(3)
    ]
    with pytest.raises(ValueError, match="exactly 2 drivers"):
        Team(name="Solo", car=car, drivers=three_drivers)


def test_team_rejects_mismatched_driver() -> None:
    """Driver team_name must match the Team name."""
    car = Car(
        team_name="Alpha",
        base_speed=80.0,
        ers_efficiency=0.8,
        aero_efficiency=0.85,
        tyre_wear_rate=1.0,
        reliability=0.95,
    )
    drivers = [
        Driver(name="D1", team_name="Alpha", skill_offset=0.0, consistency=1.0),
        Driver(name="D2", team_name="Beta", skill_offset=0.0, consistency=1.0),
    ]
    with pytest.raises(ValueError, match="does not match"):
        Team(name="Alpha", car=car, drivers=drivers)


# ---------------------------------------------------------------------------
# Driver skill effect tests
# ---------------------------------------------------------------------------


def test_driver_skill_changes_results() -> None:
    """A driver with a large positive skill offset should finish behind
    a teammate with zero offset more often than not."""
    track = _sample_track()
    team = _make_team("Skill", skill_offsets=(-0.5, 0.5))
    teams = [team]

    faster_wins = 0
    runs = 20
    for seed in range(runs):
        result = simulate_race(track, teams, laps=10, noise_std=0.02, seed=seed)
        if result.final_classification[0] == "Skill_D1":
            faster_wins += 1

    # The faster driver (negative offset) should win majority
    assert (
        faster_wins > runs // 2
    ), f"Faster driver should win majority, got {faster_wins}/{runs}"


def test_driver_consistency_affects_variance() -> None:
    """A driver with high consistency (low multiplier) should have lower
    lap-time variance than one with low consistency (high multiplier)."""
    track = _sample_track()
    team = _make_team("Var", consistencies=(0.5, 2.0))
    teams = [team]

    result = simulate_race(track, teams, laps=30, noise_std=0.10, seed=42)
    import numpy as np

    times_d1 = np.array(result.lap_times["Var_D1"])
    times_d2 = np.array(result.lap_times["Var_D2"])

    # Only consider laps where both are active (non-DNF)
    min_len = min(len(times_d1), len(times_d2))
    if min_len > 5:
        var_d1 = float(np.var(times_d1[:min_len]))
        var_d2 = float(np.var(times_d2[:min_len]))
        assert var_d1 < var_d2, (
            f"More consistent driver should have lower variance: "
            f"{var_d1:.6f} vs {var_d2:.6f}"
        )


# ---------------------------------------------------------------------------
# WDC / WCC separation tests
# ---------------------------------------------------------------------------


def test_wdc_and_wcc_not_identical() -> None:
    """WDC probabilities (driver-keyed) should differ from WCC
    probabilities (team-keyed) in structure."""
    calendar = _mini_calendar()
    teams = _sample_teams(3)
    result = simulate_season_monte_carlo(
        calendar, teams, laps_per_race=5, seasons=10, base_seed=0
    )
    # WDC keys are driver names, WCC keys are team names
    wdc_keys = set(result["wdc_probabilities"].keys())
    wcc_keys = set(result["wcc_probabilities"].keys())
    assert wdc_keys != wcc_keys, "WDC and WCC key sets must differ"
    # 3 teams * 2 drivers = 6 driver entries
    assert len(wdc_keys) == 6
    assert len(wcc_keys) == 3


def test_team_points_equal_sum_of_drivers() -> None:
    """Constructor points must equal the sum of both drivers' points."""
    calendar = _mini_calendar()
    teams = _sample_teams(3)
    result = simulate_season_monte_carlo(
        calendar, teams, laps_per_race=5, seasons=5, base_seed=77
    )

    drv_pts = result["expected_driver_points"]
    team_pts = result["expected_team_points"]

    for team in teams:
        d1, d2 = team.drivers[0].name, team.drivers[1].name
        driver_sum = drv_pts[d1] + drv_pts[d2]
        assert abs(driver_sum - team_pts[team.name]) < 1e-9, (
            f"Team {team.name}: driver sum {driver_sum:.4f} != "
            f"team pts {team_pts[team.name]:.4f}"
        )
