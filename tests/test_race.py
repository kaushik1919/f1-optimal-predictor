"""Tests for Phase 3: multi-car stochastic race simulator."""

from f1_engine.core.car import Car
from f1_engine.core.race import RaceResult, simulate_race
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
        downforce_sensitivity=2.0,
    )


def _sample_cars(n: int = 4) -> list[Car]:
    """Return *n* cars with slightly varying performance."""
    cars: list[Car] = []
    for i in range(n):
        cars.append(
            Car(
                team_name=f"Team_{i}",
                base_speed=80.0 + i * 0.3,
                ers_efficiency=0.80,
                aero_efficiency=0.85,
                tyre_wear_rate=1.0,
                reliability=0.98,
            )
        )
    return cars


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_race_runs() -> None:
    """Race must return a classification with length equal to the car count."""
    track = _sample_track()
    cars = _sample_cars(4)
    result = simulate_race(track, cars, laps=5, seed=42)
    assert isinstance(result, RaceResult)
    assert len(result.final_classification) == len(cars)


def test_seed_determinism() -> None:
    """Same seed must produce identical results."""
    track = _sample_track()
    cars = _sample_cars(4)
    r1 = simulate_race(track, cars, laps=10, noise_std=0.1, seed=123)
    r2 = simulate_race(track, cars, laps=10, noise_std=0.1, seed=123)
    assert r1.final_classification == r2.final_classification
    assert r1.dnf_list == r2.dnf_list
    assert r1.lap_times == r2.lap_times


def test_zero_noise_deterministic() -> None:
    """With noise_std=0, lap times must be identical across two runs."""
    track = _sample_track()
    cars = _sample_cars(3)
    r1 = simulate_race(track, cars, laps=8, noise_std=0.0, seed=99)
    r2 = simulate_race(track, cars, laps=8, noise_std=0.0, seed=99)
    for name in r1.lap_times:
        assert (
            r1.lap_times[name] == r2.lap_times[name]
        ), f"lap times for {name} must be identical with zero noise"


def test_dnf_occurs_with_low_reliability() -> None:
    """Very low reliability should produce at least one DNF across runs."""
    track = _sample_track()
    fragile_cars = [
        Car(
            team_name=f"Fragile_{i}",
            base_speed=80.0,
            ers_efficiency=0.8,
            aero_efficiency=0.85,
            tyre_wear_rate=1.0,
            reliability=0.05,
        )
        for i in range(6)
    ]
    dnf_seen = False
    for attempt in range(10):
        result = simulate_race(
            track, fragile_cars, laps=20, noise_std=0.05, seed=attempt
        )
        if result.dnf_list:
            dnf_seen = True
            break
    assert dnf_seen, "low reliability must produce at least one DNF"


def test_classification_contains_all_teams() -> None:
    """Every team must appear exactly once in the final classification."""
    track = _sample_track()
    cars = _sample_cars(5)
    result = simulate_race(track, cars, laps=10, seed=7)
    team_names = {c.team_name for c in cars}
    assert set(result.final_classification) == team_names


def test_lap_times_dict_keys_match_cars() -> None:
    """The lap_times dict must have an entry for every car."""
    track = _sample_track()
    cars = _sample_cars(4)
    result = simulate_race(track, cars, laps=6, seed=55)
    for car in cars:
        assert car.team_name in result.lap_times
