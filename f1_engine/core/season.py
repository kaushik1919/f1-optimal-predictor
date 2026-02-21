"""Full-season Monte Carlo championship simulator for the F1 2026 engine.

Simulates many complete 24-race seasons and aggregates the results into
World Drivers' Championship (WDC) and World Constructors' Championship (WCC)
probability distributions, expected points, and standings histograms.

Phase 10 extends this module to operate at the driver level, tracking
individual driver points for WDC and summing them per constructor for WCC.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from f1_engine.core.race import simulate_race
from f1_engine.core.team import Team
from f1_engine.core.track import Track

# Standard F1 points for positions 1-10.
_POINTS_TABLE: list[int] = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]


def simulate_season_monte_carlo(
    calendar: list[Track],
    teams: list[Team],
    laps_per_race: int,
    seasons: int,
    base_seed: int = 100,
) -> dict[str, Any]:
    """Run a Monte Carlo ensemble of full-season championship simulations.

    Each simulated season consists of every race on the *calendar* run in
    order.  Seeding follows a two-level scheme so that each race within
    each season has a unique, reproducible seed::

        race_seed = (base_seed + season_index) + race_index * 1000

    After every race, FIA championship points are awarded to the top 10
    finishers using the standard table ``[25, 18, 15, 12, 10, 8, 6, 4, 2, 1]``.
    Driver points determine the WDC; the sum of both drivers' points per
    team determine the WCC.

    Collected statistics:
      - **WDC probability** -- fraction of seasons in which the driver won
        the drivers' championship.
      - **WCC probability** -- fraction of seasons in which the team won
        the constructors' championship.
      - **Expected driver season points** -- arithmetic mean of season
        points across all simulated seasons (per driver).
      - **Expected team season points** -- arithmetic mean of season
        points across all simulated seasons (per constructor).
      - **Driver standings distribution** -- for each possible championship
        position, the probability of finishing there (per driver).
      - **Team standings distribution** -- for each possible championship
        position, the probability of finishing there (per constructor).

    Args:
        calendar: Ordered list of tracks forming the season.
        teams: List of participating teams (each with 2 drivers).
        laps_per_race: Number of laps per race (>= 1).
        seasons: Number of Monte Carlo season replications (>= 1).
        base_seed: Starting seed value.

    Returns:
        Dictionary with keys:
            wdc_probabilities            -- ``{driver_name: float}``
            wcc_probabilities            -- ``{team_name: float}``
            expected_driver_points       -- ``{driver_name: float}``
            expected_team_points         -- ``{team_name: float}``
            driver_standings_distribution -- ``{driver_name: {pos: float}}``
            team_standings_distribution  -- ``{team_name: {pos: float}}``

    Raises:
        ValueError: If seasons < 1 or calendar is empty.
    """
    if seasons < 1:
        raise ValueError("seasons must be >= 1.")
    if not calendar:
        raise ValueError("calendar must not be empty.")

    # Collect names
    driver_names: list[str] = []
    team_names: list[str] = []
    driver_to_team: dict[str, str] = {}
    for team in teams:
        team_names.append(team.name)
        for drv in team.drivers:
            driver_names.append(drv.name)
            driver_to_team[drv.name] = team.name

    # WDC accumulators
    wdc_counts: dict[str, int] = defaultdict(int)
    drv_points_sums: dict[str, float] = defaultdict(float)
    drv_standings_counts: dict[str, dict[int, int]] = {
        name: defaultdict(int) for name in driver_names
    }

    # WCC accumulators
    wcc_counts: dict[str, int] = defaultdict(int)
    team_points_sums: dict[str, float] = defaultdict(float)
    team_standings_counts: dict[str, dict[int, int]] = {
        name: defaultdict(int) for name in team_names
    }

    for season_index in range(seasons):
        season_seed: int = base_seed + season_index

        # Per-season accumulators
        drv_season_pts: dict[str, float] = {n: 0.0 for n in driver_names}
        team_season_pts: dict[str, float] = {n: 0.0 for n in team_names}

        for race_index, track in enumerate(calendar):
            race_seed: int = season_seed + race_index * 1000

            result = simulate_race(track, teams, laps_per_race, seed=race_seed)

            # Award points for top-10 finishers
            for pos_idx, drv_name in enumerate(result.final_classification):
                if pos_idx < len(_POINTS_TABLE):
                    pts = _POINTS_TABLE[pos_idx]
                    drv_season_pts[drv_name] += pts
                    team_season_pts[driver_to_team[drv_name]] += pts

        # -- WDC ranking (drivers) -------------------------------------------
        drv_ranked: list[tuple[str, float]] = sorted(
            drv_season_pts.items(), key=lambda x: x[1], reverse=True
        )
        wdc_champion: str = drv_ranked[0][0]
        wdc_counts[wdc_champion] += 1

        for pos_idx, (name, pts) in enumerate(drv_ranked):
            position: int = pos_idx + 1
            drv_points_sums[name] += pts
            drv_standings_counts[name][position] += 1

        # -- WCC ranking (constructors) --------------------------------------
        team_ranked: list[tuple[str, float]] = sorted(
            team_season_pts.items(), key=lambda x: x[1], reverse=True
        )
        wcc_champion: str = team_ranked[0][0]
        wcc_counts[wcc_champion] += 1

        for pos_idx, (name, pts) in enumerate(team_ranked):
            position = pos_idx + 1
            team_points_sums[name] += pts
            team_standings_counts[name][position] += 1

    # -- Normalise to probabilities -------------------------------------------
    inv: float = 1.0 / seasons

    wdc_probabilities: dict[str, float] = {
        name: wdc_counts[name] * inv for name in driver_names
    }
    wcc_probabilities: dict[str, float] = {
        name: wcc_counts[name] * inv for name in team_names
    }
    expected_driver_points: dict[str, float] = {
        name: drv_points_sums[name] * inv for name in driver_names
    }
    expected_team_points: dict[str, float] = {
        name: team_points_sums[name] * inv for name in team_names
    }
    driver_standings_distribution: dict[str, dict[int, float]] = {}
    for name in driver_names:
        driver_standings_distribution[name] = {
            pos: count * inv
            for pos, count in sorted(drv_standings_counts[name].items())
        }
    team_standings_distribution: dict[str, dict[int, float]] = {}
    for name in team_names:
        team_standings_distribution[name] = {
            pos: count * inv
            for pos, count in sorted(team_standings_counts[name].items())
        }

    return {
        "wdc_probabilities": wdc_probabilities,
        "wcc_probabilities": wcc_probabilities,
        "expected_driver_points": expected_driver_points,
        "expected_team_points": expected_team_points,
        "driver_standings_distribution": driver_standings_distribution,
        "team_standings_distribution": team_standings_distribution,
    }
