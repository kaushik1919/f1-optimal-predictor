"""Formal Kalman filter performance updating engine.

Replaces the heuristic gradient-free updater in ``updating.py`` with a
state-space formulation that maintains a full covariance matrix over the
latent performance vector.

State vector (theta, shape ``(3,)``)::

    [base_speed, ers_efficiency, reliability]

Covariance matrix (P, shape ``(3, 3)``):
    Tracks the estimated uncertainty of ``theta``.

The measurement model maps the state to a scalar *expected championship
points* for a specified driver.  Because this mapping is non-linear (it
involves a full Monte Carlo season simulation), the measurement gradient
``H`` is obtained via numerical central differences, making this an
*Extended Kalman Filter* (EKF).

All Monte Carlo calls use small replication counts and deterministic seeds
so that the gradient computation remains tractable and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from f1_engine.core.car import Car
from f1_engine.core.season import simulate_season_monte_carlo
from f1_engine.core.team import Team
from f1_engine.core.track import Track

# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------


@dataclass
class KalmanPerformanceState:
    """Kalman filter state for a single car's performance parameters.

    Attributes:
        theta: Mean state vector ``[base_speed, ers_efficiency, reliability]``.
        P: Covariance matrix of the state estimate, shape ``(3, 3)``.
    """

    theta: NDArray[np.float64]
    P: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.theta.shape != (3,):
            raise ValueError(f"theta must have shape (3,), got {self.theta.shape}.")
        if self.P.shape != (3, 3):
            raise ValueError(f"P must have shape (3, 3), got {self.P.shape}.")


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def initialize_kalman_state(car: Car) -> KalmanPerformanceState:
    """Create a Kalman state from a :class:`Car`'s current parameters.

    The initial covariance is diagonal with conservative uncertainty
    values reflecting the typical scale of each parameter:

    * ``base_speed``     -- sigma^2 = 0.10
    * ``ers_efficiency`` -- sigma^2 = 0.05
    * ``reliability``    -- sigma^2 = 0.01

    Args:
        car: Car whose attributes seed the state vector.

    Returns:
        Initialised :class:`KalmanPerformanceState`.
    """
    theta = np.array(
        [car.base_speed, car.ers_efficiency, car.reliability],
        dtype=np.float64,
    )
    P = np.diag([0.1, 0.05, 0.01]).astype(np.float64)  # noqa: N806
    return KalmanPerformanceState(theta=theta, P=P)


# ---------------------------------------------------------------------------
# Numerical measurement gradient
# ---------------------------------------------------------------------------


def _build_perturbed_team(
    team: Team,
    theta: NDArray[np.float64],
) -> Team:
    """Return a new Team whose Car has parameters from *theta*."""
    car = Car(
        team_name=team.car.team_name,
        base_speed=float(theta[0]),
        ers_efficiency=float(np.clip(theta[1], 0.0, 1.0)),
        aero_efficiency=team.car.aero_efficiency,
        tyre_wear_rate=team.car.tyre_wear_rate,
        reliability=float(np.clip(theta[2], 0.0, 1.0)),
    )
    return Team(name=team.name, car=car, drivers=team.drivers)


def compute_measurement_gradient(
    team: Team,
    driver_name: str,
    calendar: list[Track],
    other_teams: list[Team],
    laps_per_race: int,
    base_seed: int,
    seasons: int = 100,
    delta: float = 1e-3,
) -> NDArray[np.float64]:
    """Estimate the measurement Jacobian via central differences.

    For each of the three state dimensions, perturbs ``theta`` by
    ``+/- delta`` and evaluates the expected championship points of
    *driver_name* using a lightweight Monte Carlo season simulation.
    The partial derivative is approximated as::

        dh/d(theta_i) = (points_plus - points_minus) / (2 * delta)

    Args:
        team: The team whose car parameters are being linearised.
        driver_name: Name of the driver whose expected WDC points
            are the scalar measurement.
        calendar: Season calendar.
        other_teams: All other teams (unchanged across perturbations).
        laps_per_race: Laps per race.
        base_seed: Seed for reproducibility.
        seasons: Number of Monte Carlo replications per perturbation
            (kept small for computational efficiency).
        delta: Perturbation magnitude for finite differences.

    Returns:
        Row vector ``H`` of shape ``(1, 3)``.
    """
    theta = np.array(
        [team.car.base_speed, team.car.ers_efficiency, team.car.reliability],
        dtype=np.float64,
    )

    H = np.zeros((1, 3), dtype=np.float64)  # noqa: N806

    for i in range(3):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += delta
        theta_minus[i] -= delta

        team_plus = _build_perturbed_team(team, theta_plus)
        team_minus = _build_perturbed_team(team, theta_minus)

        all_teams_plus = [team_plus] + list(other_teams)
        all_teams_minus = [team_minus] + list(other_teams)

        result_plus = simulate_season_monte_carlo(
            calendar,
            all_teams_plus,
            laps_per_race,
            seasons,
            base_seed=base_seed,
        )
        result_minus = simulate_season_monte_carlo(
            calendar,
            all_teams_minus,
            laps_per_race,
            seasons,
            base_seed=base_seed,
        )

        pts_plus: float = result_plus["expected_driver_points"].get(driver_name, 0.0)
        pts_minus: float = result_minus["expected_driver_points"].get(driver_name, 0.0)

        H[0, i] = (pts_plus - pts_minus) / (2.0 * delta)

    return H


# ---------------------------------------------------------------------------
# Kalman update step
# ---------------------------------------------------------------------------


def kalman_update(
    state: KalmanPerformanceState,
    team: Team,
    driver_name: str,
    observed_points: float,
    expected_points: float,
    calendar: list[Track],
    other_teams: list[Team],
    laps_per_race: int,
    base_seed: int,
    measurement_variance: float = 10.0,
    gradient_seasons: int = 100,
    gradient_delta: float = 1e-3,
) -> KalmanPerformanceState:
    """Perform one Kalman filter update given an observation.

    Steps:

    1. Compute the innovation (residual)::

           y = observed_points - expected_points

    2. Compute the measurement gradient ``H`` via central differences.
    3. Compute the innovation covariance::

           S = H @ P @ H^T + R

    4. Compute the Kalman gain::

           K = P @ H^T @ inv(S)

    5. Update the state::

           theta_new = theta + K.flatten() * y

    6. Update the covariance::

           P_new = (I - K @ H) @ P

    7. Clamp ``reliability`` to ``[0, 1]`` and ``ers_efficiency``
       to ``[0, 1]``.

    Args:
        state: Prior Kalman state.
        team: Team being updated (used for gradient computation).
        driver_name: Driver whose expected points form the measurement.
        observed_points: Actually observed championship points.
        expected_points: Model-predicted championship points under
            the current state.
        calendar: Season calendar (for gradient computation).
        other_teams: Remaining teams on the grid (unchanged).
        laps_per_race: Laps per race.
        base_seed: Seed for reproducibility.
        measurement_variance: Scalar observation noise variance R.
        gradient_seasons: Monte Carlo replications for the gradient.
        gradient_delta: Perturbation step for numerical gradient.

    Returns:
        Updated :class:`KalmanPerformanceState` with new theta and P.
    """
    # 1. Innovation
    y: float = observed_points - expected_points

    # 2. Measurement gradient  H: (1, 3)
    H: NDArray[np.float64] = compute_measurement_gradient(  # noqa: N806
        team=team,
        driver_name=driver_name,
        calendar=calendar,
        other_teams=other_teams,
        laps_per_race=laps_per_race,
        base_seed=base_seed,
        seasons=gradient_seasons,
        delta=gradient_delta,
    )

    theta = state.theta.copy()
    P = state.P.copy()  # noqa: N806

    # 3. Innovation covariance  S: (1, 1)
    R: float = measurement_variance  # noqa: N806
    S: NDArray[np.float64] = H @ P @ H.T + R  # noqa: N806

    # Guard against degenerate S (should never be zero with R > 0)
    s_val: float = float(S[0, 0])
    if abs(s_val) < 1e-15:
        return KalmanPerformanceState(theta=theta, P=P)

    # 4. Kalman gain  K: (3, 1)
    K: NDArray[np.float64] = (P @ H.T) / s_val  # noqa: N806

    # 5. State update
    theta_new: NDArray[np.float64] = theta + K.flatten() * y

    # 6. Covariance update  (Joseph form is more stable but standard
    #    form is sufficient here with well-conditioned P and scalar S)
    I3 = np.eye(3, dtype=np.float64)  # noqa: N806
    P_new: NDArray[np.float64] = (I3 - K @ H) @ P  # noqa: N806

    # 7. Clamp bounded parameters
    theta_new[1] = float(np.clip(theta_new[1], 0.0, 1.0))  # ers
    theta_new[2] = float(np.clip(theta_new[2], 0.0, 1.0))  # reliability

    return KalmanPerformanceState(theta=theta_new, P=P_new)


# ---------------------------------------------------------------------------
# Apply state to team
# ---------------------------------------------------------------------------


def apply_kalman_state_to_team(
    state: KalmanPerformanceState,
    team: Team,
) -> Team:
    """Return a new Team with Car parameters drawn from *state*.

    Parameters not tracked by the Kalman filter (``aero_efficiency``,
    ``tyre_wear_rate``) are carried forward unchanged.

    Args:
        state: Current Kalman performance state.
        team: Original team to update.

    Returns:
        New :class:`Team` with an updated :class:`Car`.
    """
    return _build_perturbed_team(team, state.theta)
