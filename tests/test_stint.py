"""Tests for Phase 2: energy model, tyre model, stint simulation, strategy search."""

from f1_engine.core.car import Car
from f1_engine.core.energy import EnergyState
from f1_engine.core.stint import find_best_constant_deploy, simulate_stint
from f1_engine.core.strategy import Strategy
from f1_engine.core.track import Track
from f1_engine.core.tyre import TyreState

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


def _sample_car() -> Car:
    return Car(
        team_name="Test Racing",
        base_speed=80.0,
        ers_efficiency=0.8,
        aero_efficiency=0.85,
        tyre_wear_rate=1.0,
        reliability=0.95,
    )


# ---------------------------------------------------------------------------
# Energy model tests
# ---------------------------------------------------------------------------


def test_energy_bounds() -> None:
    """Energy must never go negative and never exceed max_charge."""
    energy = EnergyState(max_charge=4.0, current_charge=2.0)

    # Deplete beyond what is available
    deployed = energy.deploy(10.0)
    assert deployed == 2.0
    assert energy.current_charge == 0.0, "charge must not go negative"

    # Harvest beyond capacity
    harvested = energy.harvest(100.0)
    assert harvested == 4.0
    assert energy.current_charge == 4.0, "charge must not exceed max_charge"


def test_energy_deploy_returns_actual() -> None:
    """deploy() must return the actual amount deployed, not the request."""
    energy = EnergyState(max_charge=4.0, current_charge=0.3)
    actual = energy.deploy(1.0)
    assert actual == 0.3
    assert energy.current_charge == 0.0


def test_energy_harvest_returns_actual() -> None:
    """harvest() must return the actual amount harvested, not the request."""
    energy = EnergyState(max_charge=4.0, current_charge=3.8)
    actual = energy.harvest(1.0)
    assert abs(actual - 0.2) < 1e-9, "harvest must return actual amount"
    assert energy.current_charge == 4.0


# ---------------------------------------------------------------------------
# Tyre model tests
# ---------------------------------------------------------------------------


def test_tyre_increment() -> None:
    """Tyre age must increment by exactly one per call."""
    tyre = TyreState(age=0, wear_rate_multiplier=1.0)
    tyre.increment_age()
    assert tyre.age == 1
    tyre.increment_age()
    assert tyre.age == 2


# ---------------------------------------------------------------------------
# Stint simulation tests
# ---------------------------------------------------------------------------


def test_stint_deterministic_repeatability() -> None:
    """Two identical stint simulations must produce the same total_time."""
    track = _sample_track()
    car = _sample_car()
    strategy = Strategy(deploy_level=0.5, harvest_level=1.0)

    r1 = simulate_stint(track, car, strategy, laps=10)
    r2 = simulate_stint(track, car, strategy, laps=10)
    assert r1["total_time"] == r2["total_time"], "stint must be deterministic"
    assert r1["lap_times"] == r2["lap_times"]
    assert r1["energy_trace"] == r2["energy_trace"]
    assert r1["tyre_trace"] == r2["tyre_trace"]


def test_stint_trace_lengths() -> None:
    """All trace lists must have length equal to the number of laps."""
    track = _sample_track()
    car = _sample_car()
    strategy = Strategy(deploy_level=0.4, harvest_level=0.8)
    laps = 12
    result = simulate_stint(track, car, strategy, laps=laps)
    assert len(result["lap_times"]) == laps
    assert len(result["energy_trace"]) == laps
    assert len(result["tyre_trace"]) == laps


def test_stint_energy_never_negative() -> None:
    """Battery level must never drop below zero during a stint."""
    track = _sample_track()
    car = _sample_car()
    strategy = Strategy(deploy_level=0.8, harvest_level=0.2)
    result = simulate_stint(track, car, strategy, laps=20)
    for level in result["energy_trace"]:
        assert level >= 0.0, "energy must never be negative"


def test_stint_energy_never_above_max() -> None:
    """Battery level must never exceed max_charge during a stint."""
    track = _sample_track()
    car = _sample_car()
    strategy = Strategy(deploy_level=0.1, harvest_level=1.0)
    max_charge = 4.0
    result = simulate_stint(track, car, strategy, laps=20, max_charge=max_charge)
    for level in result["energy_trace"]:
        assert level <= max_charge, "energy must not exceed max_charge"


# ---------------------------------------------------------------------------
# Strategy search tests
# ---------------------------------------------------------------------------


def test_moderate_deploy_faster_than_zero() -> None:
    """A moderate deploy level (0.4) must produce a faster stint than 0.0."""
    track = _sample_track()
    car = _sample_car()
    laps = 15

    zero_strat = Strategy(deploy_level=0.0, harvest_level=1.0)
    moderate_strat = Strategy(deploy_level=0.4, harvest_level=1.0)

    r_zero = simulate_stint(track, car, zero_strat, laps=laps)
    r_mod = simulate_stint(track, car, moderate_strat, laps=laps)

    assert (
        r_mod["total_time"] < r_zero["total_time"]
    ), "moderate deploy must be faster than zero deploy"


def test_find_best_constant_deploy_returns_strategy() -> None:
    """The strategy search must return a valid Strategy and a finite time."""
    track = _sample_track()
    car = _sample_car()
    result = find_best_constant_deploy(track, car, laps=10)
    assert isinstance(result["best_strategy"], Strategy)
    assert isinstance(result["best_time"], float)
    assert result["best_time"] > 0.0
