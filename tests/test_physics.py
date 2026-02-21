"""Tests for the deterministic physics module."""

from f1_engine.core.car import Car
from f1_engine.core.physics import lap_time
from f1_engine.core.track import Track


def _sample_track() -> Track:
    """Return a representative test track."""
    return Track(
        name="Test Circuit",
        straight_ratio=0.6,
        overtake_coefficient=0.5,
        energy_harvest_factor=0.7,
        tyre_degradation_factor=0.05,
        downforce_sensitivity=2.0,
    )


def _sample_car() -> Car:
    """Return a representative test car."""
    return Car(
        team_name="Test Racing",
        base_speed=80.0,
        ers_efficiency=0.8,
        aero_efficiency=0.85,
        tyre_wear_rate=1.0,
        reliability=0.95,
    )


def test_lap_time_positive() -> None:
    """Lap time must return a float greater than zero."""
    track = _sample_track()
    car = _sample_car()
    result = lap_time(track, car, tyre_age=5.0, deploy_level=0.5)
    assert isinstance(result, float), "lap_time must return a float"
    assert result > 0.0, "lap_time must be greater than zero"


def test_lap_time_deterministic() -> None:
    """Identical inputs must always produce the same lap time."""
    track = _sample_track()
    car = _sample_car()
    t1 = lap_time(track, car, tyre_age=10.0, deploy_level=0.7)
    t2 = lap_time(track, car, tyre_age=10.0, deploy_level=0.7)
    assert t1 == t2, "lap_time must be deterministic"


def test_lap_time_tyre_degradation_increases_time() -> None:
    """Higher tyre age must produce a longer lap time."""
    track = _sample_track()
    car = _sample_car()
    fresh = lap_time(track, car, tyre_age=0.0, deploy_level=0.5)
    worn = lap_time(track, car, tyre_age=20.0, deploy_level=0.5)
    assert worn > fresh, "worn tyres must produce a slower lap time"


def test_lap_time_ers_deployment_reduces_time() -> None:
    """Higher ERS deployment must produce a shorter lap time."""
    track = _sample_track()
    car = _sample_car()
    low_deploy = lap_time(track, car, tyre_age=5.0, deploy_level=0.0)
    high_deploy = lap_time(track, car, tyre_age=5.0, deploy_level=1.0)
    assert high_deploy < low_deploy, "higher ERS deploy must reduce lap time"
