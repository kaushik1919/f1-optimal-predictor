"""Tests for Phase 4: Monte Carlo race analytics engine."""

from f1_engine.core.car import Car
from f1_engine.core.driver import Driver
from f1_engine.core.monte_carlo import simulate_race_monte_carlo
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


def test_probabilities_sum_to_one() -> None:
    """Winner probabilities across all drivers must sum to 1.0."""
    track = _sample_track()
    teams = _sample_teams()
    result = simulate_race_monte_carlo(track, teams, laps=5, simulations=20)

    winner_sum = sum(result["winner_probabilities"].values())
    assert (
        abs(winner_sum - 1.0) < 1e-9
    ), f"winner probabilities must sum to 1.0, got {winner_sum}"

    for name, dist in result["finish_distribution"].items():
        dist_sum = sum(dist.values())
        assert (
            abs(dist_sum - 1.0) < 1e-9
        ), f"finish distribution for {name} must sum to 1.0, got {dist_sum}"


def test_deterministic_given_base_seed() -> None:
    """Two runs with the same base_seed must produce identical results."""
    track = _sample_track()
    teams = _sample_teams()
    r1 = simulate_race_monte_carlo(track, teams, laps=5, simulations=10, base_seed=99)
    r2 = simulate_race_monte_carlo(track, teams, laps=5, simulations=10, base_seed=99)
    assert r1["winner_probabilities"] == r2["winner_probabilities"]
    assert r1["expected_points"] == r2["expected_points"]
    assert r1["finish_distribution"] == r2["finish_distribution"]


def test_expected_points_non_negative() -> None:
    """Expected points for every driver must be >= 0."""
    track = _sample_track()
    teams = _sample_teams()
    result = simulate_race_monte_carlo(track, teams, laps=5, simulations=15)
    for name, pts in result["expected_points"].items():
        assert pts >= 0.0, f"expected points for {name} must be >= 0"


def test_more_simulations_stabilize_results() -> None:
    """With more simulations the expected position should remain bounded
    between 1 and the number of drivers."""
    track = _sample_track()
    teams = _sample_teams()
    n_drivers = len(teams) * 2
    result = simulate_race_monte_carlo(
        track, teams, laps=5, simulations=50, base_seed=0
    )
    for name, pos in result["expected_position"].items():
        assert (
            1.0 <= pos <= float(n_drivers)
        ), f"expected position for {name} out of range: {pos}"
