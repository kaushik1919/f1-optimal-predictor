"""Sensitivity and volatility analysis engine for the F1 2026 simulation engine.

This module provides tools for measuring how sensitive championship outcomes
are to small perturbations in car parameters (elasticity via central
differences) and for quantifying the overall unpredictability of a
championship (Shannon entropy of WDC probability distributions).

All Monte Carlo calls are fully seeded for reproducibility.

Phase 10 operates on teams (with two drivers each).  Sensitivity functions
perturb the *car* attached to a target team and evaluate the WDC probability
of a specified driver from that team.
"""

from __future__ import annotations

import math
from dataclasses import replace

from f1_engine.core.car import Car
from f1_engine.core.season import simulate_season_monte_carlo
from f1_engine.core.team import Team
from f1_engine.core.track import Track

# ---------------------------------------------------------------------------
# Reliability sensitivity
# ---------------------------------------------------------------------------


def compute_reliability_sensitivity(
    calendar: list[Track],
    team: Team,
    other_teams: list[Team],
    driver_name: str,
    laps_per_race: int,
    seasons: int,
    delta: float = 0.01,
    base_seed: int = 200,
) -> float:
    """Estimate the elasticity of WDC probability w.r.t. reliability.

    Uses a central-difference approximation::

        sensitivity = (WDC_plus - WDC_minus) / (2 * delta)

    where ``WDC_plus`` and ``WDC_minus`` are the WDC probabilities for
    *driver_name* when the team car's reliability is perturbed by ``+delta``
    and ``-delta`` respectively.

    The perturbed reliability values are clamped to ``[0.0, 1.0]``.

    Args:
        calendar: Season calendar (list of tracks).
        team: The team whose car reliability is being perturbed.
        other_teams: All other teams in the field (unchanged).
        driver_name: The driver whose WDC probability is evaluated.
        laps_per_race: Laps per race.
        seasons: Monte Carlo replications.
        delta: Perturbation magnitude.
        base_seed: Seed for reproducibility.

    Returns:
        Central-difference elasticity estimate (float).
    """
    rel_plus: float = min(1.0, team.car.reliability + delta)
    rel_minus: float = max(0.0, team.car.reliability - delta)

    car_plus: Car = replace(team.car, reliability=rel_plus)
    car_minus: Car = replace(team.car, reliability=rel_minus)

    team_plus: Team = Team(name=team.name, car=car_plus, drivers=team.drivers)
    team_minus: Team = Team(name=team.name, car=car_minus, drivers=team.drivers)

    teams_plus: list[Team] = [team_plus] + list(other_teams)
    teams_minus: list[Team] = [team_minus] + list(other_teams)

    result_plus = simulate_season_monte_carlo(
        calendar, teams_plus, laps_per_race, seasons, base_seed=base_seed
    )
    result_minus = simulate_season_monte_carlo(
        calendar, teams_minus, laps_per_race, seasons, base_seed=base_seed
    )

    wdc_plus: float = result_plus["wdc_probabilities"].get(driver_name, 0.0)
    wdc_minus: float = result_minus["wdc_probabilities"].get(driver_name, 0.0)

    actual_delta: float = rel_plus - rel_minus
    if actual_delta == 0.0:
        return 0.0

    return (wdc_plus - wdc_minus) / actual_delta


# ---------------------------------------------------------------------------
# ERS efficiency sensitivity
# ---------------------------------------------------------------------------


def compute_ers_sensitivity(
    calendar: list[Track],
    team: Team,
    other_teams: list[Team],
    driver_name: str,
    laps_per_race: int,
    seasons: int,
    delta: float = 0.01,
    base_seed: int = 200,
) -> float:
    """Estimate the elasticity of WDC probability w.r.t. ERS efficiency.

    Uses a central-difference approximation::

        sensitivity = (WDC_plus - WDC_minus) / (2 * delta)

    where ``WDC_plus`` and ``WDC_minus`` are the WDC probabilities for
    *driver_name* when the team car's ``ers_efficiency`` is perturbed by
    ``+delta`` and ``-delta`` respectively.

    The perturbed ERS efficiency values are clamped to ``[0.0, 1.0]``.

    Args:
        calendar: Season calendar (list of tracks).
        team: The team whose car ERS efficiency is being perturbed.
        other_teams: All other teams in the field (unchanged).
        driver_name: The driver whose WDC probability is evaluated.
        laps_per_race: Laps per race.
        seasons: Monte Carlo replications.
        delta: Perturbation magnitude.
        base_seed: Seed for reproducibility.

    Returns:
        Central-difference elasticity estimate (float).
    """
    ers_plus: float = min(1.0, team.car.ers_efficiency + delta)
    ers_minus: float = max(0.0, team.car.ers_efficiency - delta)

    car_plus: Car = replace(team.car, ers_efficiency=ers_plus)
    car_minus: Car = replace(team.car, ers_efficiency=ers_minus)

    team_plus: Team = Team(name=team.name, car=car_plus, drivers=team.drivers)
    team_minus: Team = Team(name=team.name, car=car_minus, drivers=team.drivers)

    teams_plus: list[Team] = [team_plus] + list(other_teams)
    teams_minus: list[Team] = [team_minus] + list(other_teams)

    result_plus = simulate_season_monte_carlo(
        calendar, teams_plus, laps_per_race, seasons, base_seed=base_seed
    )
    result_minus = simulate_season_monte_carlo(
        calendar, teams_minus, laps_per_race, seasons, base_seed=base_seed
    )

    wdc_plus: float = result_plus["wdc_probabilities"].get(driver_name, 0.0)
    wdc_minus: float = result_minus["wdc_probabilities"].get(driver_name, 0.0)

    actual_delta: float = ers_plus - ers_minus
    if actual_delta == 0.0:
        return 0.0

    return (wdc_plus - wdc_minus) / actual_delta


# ---------------------------------------------------------------------------
# Championship volatility (Shannon entropy)
# ---------------------------------------------------------------------------


def compute_championship_entropy(wdc_probabilities: dict[str, float]) -> float:
    """Compute the Shannon entropy of a WDC probability distribution.

    Entropy quantifies championship unpredictability.  A single dominant
    team yields entropy close to 0; a perfectly balanced field with *n*
    teams yields ``log(n)``.

    The formula is::

        H = -sum(p * log(p))  for all p > 0

    Zero-probability entries are skipped (``0 * log(0)`` is treated as 0
    by convention).

    Args:
        wdc_probabilities: Mapping from team name to WDC probability.

    Returns:
        Shannon entropy (non-negative float, in nats).
    """
    entropy: float = 0.0
    for prob in wdc_probabilities.values():
        if prob > 0.0:
            entropy -= prob * math.log(prob)
    return entropy
