"""Latent performance updating engine for the F1 2026 simulation engine.

This module implements a simple Bayesian-style adjustment mechanism that
shifts latent car performance parameters after comparing observed race
results against prior expectations.  The update rule is gradient-free:
the error signal (observed minus expected points) is scaled by a learning
rate and applied to each parameter with a domain-specific sensitivity
coefficient.

All state objects are immutable; every update returns a fresh instance.
"""

from __future__ import annotations

from dataclasses import dataclass

from f1_engine.core.car import Car

# ---------------------------------------------------------------------------
# Performance state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerformanceState:
    """Latent true-performance parameters for a single car.

    These values represent the engine's current belief about a car's
    underlying capability.  They are updated after each observed race
    via :func:`update_performance_state`.

    Attributes:
        base_speed: Believed baseline lap time in seconds (lower is faster).
        ers_efficiency: Believed ERS deployment/recovery effectiveness
            (0.0--1.0).
        reliability: Believed mechanical reliability factor (0.0--1.0).
    """

    base_speed: float
    ers_efficiency: float
    reliability: float


# ---------------------------------------------------------------------------
# Update function
# ---------------------------------------------------------------------------


def update_performance_state(
    prior: PerformanceState,
    observed_points: float,
    expected_points: float,
    learning_rate: float = 0.05,
) -> PerformanceState:
    """Return an updated performance state given observed vs expected points.

    The error signal ``observed_points - expected_points`` drives three
    independent parameter adjustments:

    * ``base_speed  -= learning_rate * error * 0.01``
      (positive error means the car did better than expected, so its true
      base speed is likely faster -- i.e. a lower value).
    * ``ers_efficiency += learning_rate * error * 0.005``
      (positive error nudges ERS efficiency upward).
    * ``reliability += learning_rate * error * 0.001``
      (positive error nudges reliability upward).

    Reliability is clamped to the interval ``[0.0, 1.0]`` after the update.

    Args:
        prior: Current performance belief.
        observed_points: Points actually scored in the latest race.
        expected_points: Points the model predicted before the race.
        learning_rate: Step-size multiplier controlling update magnitude.

    Returns:
        A new :class:`PerformanceState` with adjusted parameters.
    """
    error: float = observed_points - expected_points

    new_base_speed: float = prior.base_speed - learning_rate * error * 0.01
    new_ers_efficiency: float = prior.ers_efficiency + learning_rate * error * 0.005
    new_reliability: float = prior.reliability + learning_rate * error * 0.001

    # Clamp reliability to [0, 1].
    new_reliability = max(0.0, min(1.0, new_reliability))

    return PerformanceState(
        base_speed=new_base_speed,
        ers_efficiency=new_ers_efficiency,
        reliability=new_reliability,
    )


# ---------------------------------------------------------------------------
# Integration helper
# ---------------------------------------------------------------------------


def apply_updated_state(car: Car, state: PerformanceState) -> Car:
    """Return a new :class:`Car` with performance parameters from *state*.

    Non-performance attributes (``team_name``, ``aero_efficiency``,
    ``tyre_wear_rate``) are carried over unchanged from the original car.

    Args:
        car: Original car instance.
        state: Updated performance belief to apply.

    Returns:
        A new :class:`Car` reflecting the updated latent performance.
    """
    return Car(
        team_name=car.team_name,
        base_speed=state.base_speed,
        ers_efficiency=state.ers_efficiency,
        aero_efficiency=car.aero_efficiency,
        tyre_wear_rate=car.tyre_wear_rate,
        reliability=state.reliability,
    )
