"""Tests for Phase 5: full-season Monte Carlo championship simulator."""

from f1_engine.core.car import Car
from f1_engine.core.driver import Driver
from f1_engine.core.season import simulate_season_monte_carlo
from f1_engine.core.team import Team
from f1_engine.core.track import Track

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mini_calendar() -> list[Track]:
    """Return a short 3-race calendar for fast testing."""
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


def _make_team(name: str, base_speed: float) -> Team:
    car = Car(
        team_name=name,
        base_speed=base_speed,
        ers_efficiency=0.80,
        aero_efficiency=0.85,
        tyre_wear_rate=1.0,
        reliability=0.98,
    )
    drivers = [
        Driver(name=f"{name}_D1", team_name=name, skill_offset=0.0, consistency=1.0),
        Driver(name=f"{name}_D2", team_name=name, skill_offset=0.0, consistency=1.0),
    ]
    return Team(name=name, car=car, drivers=drivers)


def _sample_teams() -> list[Team]:
    return [_make_team(f"Team_{i}", 80.0 + i * 0.3) for i in range(4)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_wdc_probabilities_sum_to_one() -> None:
    """WDC probabilities across all drivers must sum to 1.0."""
    calendar = _mini_calendar()
    teams = _sample_teams()
    result = simulate_season_monte_carlo(
        calendar, teams, laps_per_race=5, seasons=10, base_seed=0
    )
    wdc_sum = sum(result["wdc_probabilities"].values())
    assert (
        abs(wdc_sum - 1.0) < 1e-9
    ), f"WDC probabilities must sum to 1.0, got {wdc_sum}"


def test_expected_points_non_negative() -> None:
    """Expected season points for every driver must be >= 0."""
    calendar = _mini_calendar()
    teams = _sample_teams()
    result = simulate_season_monte_carlo(
        calendar, teams, laps_per_race=5, seasons=10, base_seed=42
    )
    for name, pts in result["expected_driver_points"].items():
        assert pts >= 0.0, f"expected points for {name} must be >= 0"


def test_seed_determinism() -> None:
    """Two runs with the same base_seed must produce identical results."""
    calendar = _mini_calendar()
    teams = _sample_teams()
    r1 = simulate_season_monte_carlo(
        calendar, teams, laps_per_race=5, seasons=8, base_seed=77
    )
    r2 = simulate_season_monte_carlo(
        calendar, teams, laps_per_race=5, seasons=8, base_seed=77
    )
    assert r1["wdc_probabilities"] == r2["wdc_probabilities"]
    assert r1["expected_driver_points"] == r2["expected_driver_points"]
    assert r1["driver_standings_distribution"] == r2["driver_standings_distribution"]


def test_single_season_behaviour() -> None:
    """With seasons=1, exactly one driver must be WDC champion (prob 1.0)."""
    calendar = _mini_calendar()
    teams = _sample_teams()
    result = simulate_season_monte_carlo(
        calendar, teams, laps_per_race=5, seasons=1, base_seed=10
    )
    champions = [
        name for name, prob in result["wdc_probabilities"].items() if prob > 0.0
    ]
    assert len(champions) == 1, "exactly one WDC champion with seasons=1"
    assert result["wdc_probabilities"][champions[0]] == 1.0


def test_standings_distribution_sums() -> None:
    """Each driver's standings distribution must sum to 1.0."""
    calendar = _mini_calendar()
    teams = _sample_teams()
    result = simulate_season_monte_carlo(
        calendar, teams, laps_per_race=5, seasons=15, base_seed=55
    )
    for name, dist in result["driver_standings_distribution"].items():
        dist_sum = sum(dist.values())
        assert (
            abs(dist_sum - 1.0) < 1e-9
        ), f"standings distribution for {name} must sum to 1.0, got {dist_sum}"


def test_expected_position_in_range() -> None:
    """Expected final position must be between 1 and the number of drivers."""
    calendar = _mini_calendar()
    teams = _sample_teams()
    n_drivers = len(teams) * 2
    result = simulate_season_monte_carlo(
        calendar, teams, laps_per_race=5, seasons=10, base_seed=33
    )
    for name, dist in result["driver_standings_distribution"].items():
        for pos in dist:
            assert 1 <= pos <= n_drivers, f"position {pos} for {name} out of range"
