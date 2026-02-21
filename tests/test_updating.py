"""Tests for Phase 6: latent performance updating engine."""

from f1_engine.core.car import Car
from f1_engine.core.updating import (
    PerformanceState,
    apply_updated_state,
    update_performance_state,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _base_state() -> PerformanceState:
    return PerformanceState(
        base_speed=80.0,
        ers_efficiency=0.80,
        reliability=0.95,
    )


def _sample_car() -> Car:
    return Car(
        team_name="TestTeam",
        base_speed=80.0,
        ers_efficiency=0.80,
        aero_efficiency=0.85,
        tyre_wear_rate=1.0,
        reliability=0.95,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_update_improves_performance_after_positive_error() -> None:
    """Positive error (scored more than expected) should make the car faster.

    Faster means: lower base_speed, higher ers_efficiency, higher reliability.
    """
    prior = _base_state()
    updated = update_performance_state(
        prior, observed_points=25.0, expected_points=10.0
    )

    assert updated.base_speed < prior.base_speed
    assert updated.ers_efficiency > prior.ers_efficiency
    assert updated.reliability > prior.reliability


def test_update_reduces_performance_after_negative_error() -> None:
    """Negative error (scored less than expected) should make the car slower.

    Slower means: higher base_speed, lower ers_efficiency, lower reliability.
    """
    prior = _base_state()
    updated = update_performance_state(prior, observed_points=0.0, expected_points=18.0)

    assert updated.base_speed > prior.base_speed
    assert updated.ers_efficiency < prior.ers_efficiency
    assert updated.reliability < prior.reliability


def test_reliability_clamped() -> None:
    """Reliability must remain in [0.0, 1.0] regardless of error magnitude."""
    # Push reliability above 1.0 with a huge positive error.
    high = update_performance_state(
        PerformanceState(base_speed=80.0, ers_efficiency=0.8, reliability=0.99),
        observed_points=500.0,
        expected_points=0.0,
        learning_rate=1.0,
    )
    assert high.reliability <= 1.0

    # Push reliability below 0.0 with a huge negative error.
    low = update_performance_state(
        PerformanceState(base_speed=80.0, ers_efficiency=0.8, reliability=0.01),
        observed_points=0.0,
        expected_points=500.0,
        learning_rate=1.0,
    )
    assert low.reliability >= 0.0


def test_deterministic_update() -> None:
    """Same inputs must always produce the exact same output."""
    prior = _base_state()
    r1 = update_performance_state(prior, observed_points=12.0, expected_points=8.0)
    r2 = update_performance_state(prior, observed_points=12.0, expected_points=8.0)

    assert r1.base_speed == r2.base_speed
    assert r1.ers_efficiency == r2.ers_efficiency
    assert r1.reliability == r2.reliability


def test_apply_updated_state_preserves_non_performance() -> None:
    """apply_updated_state must carry over team_name, aero, tyre attributes."""
    car = _sample_car()
    state = PerformanceState(base_speed=78.0, ers_efficiency=0.85, reliability=0.97)
    new_car = apply_updated_state(car, state)

    assert new_car.team_name == car.team_name
    assert new_car.aero_efficiency == car.aero_efficiency
    assert new_car.tyre_wear_rate == car.tyre_wear_rate
    assert new_car.base_speed == state.base_speed
    assert new_car.ers_efficiency == state.ers_efficiency
    assert new_car.reliability == state.reliability


def test_zero_error_no_change() -> None:
    """When observed equals expected, parameters should not change."""
    prior = _base_state()
    updated = update_performance_state(
        prior, observed_points=10.0, expected_points=10.0
    )
    assert updated.base_speed == prior.base_speed
    assert updated.ers_efficiency == prior.ers_efficiency
    assert updated.reliability == prior.reliability
