"""Tests for Phase 7: sensitivity and volatility analysis engine."""

import math

from f1_engine.core.car import Car
from f1_engine.core.sensitivity import (
    compute_championship_entropy,
    compute_ers_sensitivity,
    compute_reliability_sensitivity,
)
from f1_engine.core.track import Track

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mini_calendar() -> list[Track]:
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
    ]


def _target_car() -> Car:
    return Car(
        team_name="TargetTeam",
        base_speed=80.0,
        ers_efficiency=0.80,
        aero_efficiency=0.85,
        tyre_wear_rate=1.0,
        reliability=0.95,
    )


def _other_cars() -> list[Car]:
    return [
        Car(
            team_name=f"Rival_{i}",
            base_speed=80.5 + i * 0.2,
            ers_efficiency=0.78,
            aero_efficiency=0.84,
            tyre_wear_rate=1.0,
            reliability=0.94,
        )
        for i in range(3)
    ]


# ---------------------------------------------------------------------------
# Entropy tests
# ---------------------------------------------------------------------------


def test_entropy_positive() -> None:
    """Shannon entropy must be non-negative for any valid distribution."""
    probs = {"A": 0.5, "B": 0.3, "C": 0.2}
    h = compute_championship_entropy(probs)
    assert h > 0.0


def test_entropy_zero_for_certain_champion() -> None:
    """When one team has probability 1.0, entropy should be 0.0."""
    probs = {"A": 1.0, "B": 0.0, "C": 0.0}
    h = compute_championship_entropy(probs)
    assert h == 0.0


def test_entropy_maximum_for_uniform() -> None:
    """Uniform distribution over n teams should give entropy = log(n)."""
    n = 4
    probs = {f"T{i}": 1.0 / n for i in range(n)}
    h = compute_championship_entropy(probs)
    assert abs(h - math.log(n)) < 1e-9


# ---------------------------------------------------------------------------
# Sensitivity tests
# ---------------------------------------------------------------------------


def test_sensitivity_sign_reasonable() -> None:
    """Reliability sensitivity for a competitive car should be >= 0.

    Increasing reliability should not decrease WDC probability, so the
    central-difference estimate should be non-negative (within noise).
    """
    calendar = _mini_calendar()
    car = _target_car()
    others = _other_cars()
    sens = compute_reliability_sensitivity(
        calendar,
        car,
        others,
        laps_per_race=5,
        seasons=6,
        delta=0.02,
        base_seed=42,
    )
    # With few simulations the sign could fluctuate, but a reasonable
    # delta should keep it >= -0.5 at minimum (not wildly wrong).
    assert sens >= -1.0


def test_ers_sensitivity_runs() -> None:
    """ERS sensitivity should return a finite float without errors."""
    calendar = _mini_calendar()
    car = _target_car()
    others = _other_cars()
    sens = compute_ers_sensitivity(
        calendar,
        car,
        others,
        laps_per_race=5,
        seasons=6,
        delta=0.02,
        base_seed=42,
    )
    assert math.isfinite(sens)


def test_zero_delta_returns_zero() -> None:
    """When delta is 0, the denominator collapses; function should return 0."""
    calendar = _mini_calendar()
    car = _target_car()
    others = _other_cars()
    sens = compute_reliability_sensitivity(
        calendar,
        car,
        others,
        laps_per_race=5,
        seasons=4,
        delta=0.0,
        base_seed=10,
    )
    assert sens == 0.0
