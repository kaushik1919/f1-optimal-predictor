"""Tests for Phase 5: full-season Monte Carlo championship simulator."""

from f1_engine.core.car import Car
from f1_engine.core.season import simulate_season_monte_carlo
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
        downforce_sensitivity=2.0,
    )
    return [
        Track(name="Track_A", **base),
        Track(name="Track_B", **base),
        Track(name="Track_C", **base),
    ]


def _sample_cars() -> list[Car]:
    return [
        Car(
            team_name=f"Team_{i}",
            base_speed=80.0 + i * 0.3,
            ers_efficiency=0.80,
            aero_efficiency=0.85,
            tyre_wear_rate=1.0,
            reliability=0.98,
        )
        for i in range(4)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_wdc_probabilities_sum_to_one() -> None:
    """WDC probabilities across all teams must sum to 1.0."""
    calendar = _mini_calendar()
    cars = _sample_cars()
    result = simulate_season_monte_carlo(
        calendar, cars, laps_per_race=5, seasons=10, base_seed=0
    )
    wdc_sum = sum(result["wdc_probabilities"].values())
    assert (
        abs(wdc_sum - 1.0) < 1e-9
    ), f"WDC probabilities must sum to 1.0, got {wdc_sum}"


def test_expected_points_non_negative() -> None:
    """Expected season points for every team must be >= 0."""
    calendar = _mini_calendar()
    cars = _sample_cars()
    result = simulate_season_monte_carlo(
        calendar, cars, laps_per_race=5, seasons=10, base_seed=42
    )
    for name, pts in result["expected_points"].items():
        assert pts >= 0.0, f"expected points for {name} must be >= 0"


def test_seed_determinism() -> None:
    """Two runs with the same base_seed must produce identical results."""
    calendar = _mini_calendar()
    cars = _sample_cars()
    r1 = simulate_season_monte_carlo(
        calendar, cars, laps_per_race=5, seasons=8, base_seed=77
    )
    r2 = simulate_season_monte_carlo(
        calendar, cars, laps_per_race=5, seasons=8, base_seed=77
    )
    assert r1["wdc_probabilities"] == r2["wdc_probabilities"]
    assert r1["expected_points"] == r2["expected_points"]
    assert r1["standings_distribution"] == r2["standings_distribution"]


def test_single_season_behaviour() -> None:
    """With seasons=1, exactly one team must be champion (probability 1.0)."""
    calendar = _mini_calendar()
    cars = _sample_cars()
    result = simulate_season_monte_carlo(
        calendar, cars, laps_per_race=5, seasons=1, base_seed=10
    )
    champions = [
        name for name, prob in result["wdc_probabilities"].items() if prob > 0.0
    ]
    assert len(champions) == 1, "exactly one champion with seasons=1"
    assert result["wdc_probabilities"][champions[0]] == 1.0


def test_standings_distribution_sums() -> None:
    """Each team's standings distribution must sum to 1.0."""
    calendar = _mini_calendar()
    cars = _sample_cars()
    result = simulate_season_monte_carlo(
        calendar, cars, laps_per_race=5, seasons=15, base_seed=55
    )
    for name, dist in result["standings_distribution"].items():
        dist_sum = sum(dist.values())
        assert (
            abs(dist_sum - 1.0) < 1e-9
        ), f"standings distribution for {name} must sum to 1.0, got {dist_sum}"


def test_expected_position_in_range() -> None:
    """Expected final position must be between 1 and the number of cars."""
    calendar = _mini_calendar()
    cars = _sample_cars()
    n_cars = len(cars)
    result = simulate_season_monte_carlo(
        calendar, cars, laps_per_race=5, seasons=10, base_seed=33
    )
    for name, pos in result["expected_final_position"].items():
        assert (
            1.0 <= pos <= float(n_cars)
        ), f"expected final position for {name} out of range: {pos}"
