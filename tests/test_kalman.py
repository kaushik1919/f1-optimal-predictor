"""Tests for Phase 11C: formal Kalman filter performance updating."""

import numpy as np

from f1_engine.core.car import Car
from f1_engine.core.driver import Driver
from f1_engine.core.kalman_update import (
    KalmanPerformanceState,
    apply_kalman_state_to_team,
    initialize_kalman_state,
    kalman_update,
)
from f1_engine.core.team import Team
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
        downforce_sensitivity=0.50,
    )
    return [Track(name="T1", **base), Track(name="T2", **base)]


def _make_team(
    name: str,
    base_speed: float = 80.0,
    ers: float = 0.80,
    reliability: float = 0.95,
) -> Team:
    car = Car(
        team_name=name,
        base_speed=base_speed,
        ers_efficiency=ers,
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


def _target_team() -> Team:
    return _make_team("Target")


def _other_teams() -> list[Team]:
    return [_make_team(f"Rival_{i}", base_speed=80.0 + i * 0.3) for i in range(2)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_kalman_initialization() -> None:
    """initialize_kalman_state must produce correct theta and P."""
    car = Car(
        team_name="Init",
        base_speed=82.0,
        ers_efficiency=0.75,
        aero_efficiency=0.85,
        tyre_wear_rate=1.0,
        reliability=0.90,
    )
    state = initialize_kalman_state(car)

    assert state.theta.shape == (3,)
    assert state.P.shape == (3, 3)
    assert float(state.theta[0]) == 82.0
    assert float(state.theta[1]) == 0.75
    assert float(state.theta[2]) == 0.90
    # Covariance is diagonal with known values.
    assert float(state.P[0, 0]) == 0.1
    assert float(state.P[1, 1]) == 0.05
    assert float(state.P[2, 2]) == 0.01
    # Off-diagonals are zero.
    assert float(state.P[0, 1]) == 0.0
    assert float(state.P[1, 2]) == 0.0


def test_kalman_update_changes_theta() -> None:
    """After an update with non-zero residual, theta must change."""
    team = _target_team()
    others = _other_teams()
    calendar = _mini_calendar()
    state = initialize_kalman_state(team.car)

    # Observed points much higher than expected → push towards better.
    updated = kalman_update(
        state=state,
        team=team,
        driver_name="Target_D1",
        observed_points=100.0,
        expected_points=40.0,
        calendar=calendar,
        other_teams=others,
        laps_per_race=5,
        base_seed=42,
        gradient_seasons=20,
    )

    assert not np.array_equal(
        updated.theta, state.theta
    ), "theta must change after a non-zero residual update"


def test_covariance_reduces_after_update() -> None:
    """The trace of P must not increase after a Kalman update.

    A measurement always adds information, so tr(P_new) <= tr(P_old).
    """
    team = _target_team()
    others = _other_teams()
    calendar = _mini_calendar()
    state = initialize_kalman_state(team.car)

    updated = kalman_update(
        state=state,
        team=team,
        driver_name="Target_D1",
        observed_points=50.0,
        expected_points=45.0,
        calendar=calendar,
        other_teams=others,
        laps_per_race=5,
        base_seed=7,
        gradient_seasons=20,
    )

    assert (
        np.trace(updated.P) <= np.trace(state.P) + 1e-12
    ), "Covariance trace must not increase after an update"


def test_reliability_clamped() -> None:
    """Reliability in theta must stay within [0, 1] after an update."""
    team = _make_team("Clamp", reliability=0.99)
    others = _other_teams()
    calendar = _mini_calendar()
    state = initialize_kalman_state(team.car)

    # Large positive residual to push reliability upward.
    updated = kalman_update(
        state=state,
        team=team,
        driver_name="Clamp_D1",
        observed_points=200.0,
        expected_points=10.0,
        calendar=calendar,
        other_teams=others,
        laps_per_race=5,
        base_seed=99,
        gradient_seasons=20,
    )

    assert (
        0.0 <= float(updated.theta[2]) <= 1.0
    ), f"reliability must be in [0, 1], got {updated.theta[2]}"
    assert (
        0.0 <= float(updated.theta[1]) <= 1.0
    ), f"ers_efficiency must be in [0, 1], got {updated.theta[1]}"


def test_deterministic_given_same_inputs() -> None:
    """Two identical kalman_update calls must produce identical output."""
    team = _target_team()
    others = _other_teams()
    calendar = _mini_calendar()
    state = initialize_kalman_state(team.car)

    kwargs = dict(
        state=state,
        team=team,
        driver_name="Target_D1",
        observed_points=60.0,
        expected_points=50.0,
        calendar=calendar,
        other_teams=others,
        laps_per_race=5,
        base_seed=123,
        gradient_seasons=20,
    )

    r1 = kalman_update(**kwargs)
    r2 = kalman_update(**kwargs)

    np.testing.assert_array_equal(r1.theta, r2.theta)
    np.testing.assert_array_equal(r1.P, r2.P)


def test_apply_kalman_state_to_team() -> None:
    """apply_kalman_state_to_team must produce a Team with updated Car."""
    team = _target_team()
    state = KalmanPerformanceState(
        theta=np.array([79.5, 0.85, 0.92]),
        P=np.diag([0.08, 0.04, 0.008]),
    )

    new_team = apply_kalman_state_to_team(state, team)

    assert new_team.car.base_speed == 79.5
    assert new_team.car.ers_efficiency == 0.85
    assert new_team.car.reliability == 0.92
    # Unchanged attributes preserved.
    assert new_team.car.aero_efficiency == team.car.aero_efficiency
    assert new_team.car.tyre_wear_rate == team.car.tyre_wear_rate
    assert new_team.name == team.name
    assert len(new_team.drivers) == 2
