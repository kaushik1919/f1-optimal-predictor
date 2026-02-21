"""Full-season Monte Carlo championship simulator for the F1 2026 engine.

Simulates many complete 24-race seasons and aggregates the results into
World Drivers' Championship probability distributions, expected points,
and standings histograms.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from f1_engine.core.car import Car
from f1_engine.core.race import simulate_race
from f1_engine.core.track import Track

# Standard F1 points for positions 1-10.
_POINTS_TABLE: list[int] = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]


def simulate_season_monte_carlo(
    calendar: list[Track],
    cars: list[Car],
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
    At the end of a season the standings are ranked by total points and
    the champion is recorded.

    Collected statistics per team:
      - **WDC probability** -- fraction of seasons in which the team won
        the championship.
      - **Expected total season points** -- arithmetic mean of season
        points across all simulated seasons.
      - **Expected final championship position** -- arithmetic mean of
        the end-of-season standing.
      - **Standings distribution** -- for each possible championship
        position, the probability of finishing there.

    Args:
        calendar: Ordered list of tracks forming the season.
        cars: List of participating cars (consistent across all races).
        laps_per_race: Number of laps per race (>= 1).
        seasons: Number of Monte Carlo season replications (>= 1).
        base_seed: Starting seed value.

    Returns:
        Dictionary with keys:
            wdc_probabilities        -- ``{team_name: float}``
            expected_points          -- ``{team_name: float}``
            expected_final_position  -- ``{team_name: float}``
            standings_distribution   -- ``{team_name: {position: float}}``

    Raises:
        ValueError: If seasons < 1 or calendar is empty.
    """
    if seasons < 1:
        raise ValueError("seasons must be >= 1.")
    if not calendar:
        raise ValueError("calendar must not be empty.")

    team_names: list[str] = [c.team_name for c in cars]

    # Accumulators
    wdc_counts: dict[str, int] = defaultdict(int)
    points_sums: dict[str, float] = defaultdict(float)
    position_sums: dict[str, int] = defaultdict(int)
    standings_counts: dict[str, dict[int, int]] = {
        name: defaultdict(int) for name in team_names
    }

    for season_index in range(seasons):
        season_seed: int = base_seed + season_index

        # Per-season points accumulator
        season_points: dict[str, float] = {name: 0.0 for name in team_names}

        for race_index, track in enumerate(calendar):
            race_seed: int = season_seed + race_index * 1000

            result = simulate_race(track, cars, laps_per_race, seed=race_seed)

            # Award points for top-10 finishers
            for pos_idx, name in enumerate(result.final_classification):
                if pos_idx < len(_POINTS_TABLE):
                    season_points[name] += _POINTS_TABLE[pos_idx]

        # Rank by total season points (descending), stable sort preserves
        # insertion order for ties.
        ranked: list[tuple[str, float]] = sorted(
            season_points.items(), key=lambda x: x[1], reverse=True
        )

        # Record champion
        champion_name: str = ranked[0][0]
        wdc_counts[champion_name] += 1

        # Record positions and points
        for pos_idx, (name, pts) in enumerate(ranked):
            position: int = pos_idx + 1  # 1-based
            points_sums[name] += pts
            position_sums[name] += position
            standings_counts[name][position] += 1

    # -- Normalise to probabilities -------------------------------------------
    inv: float = 1.0 / seasons

    wdc_probabilities: dict[str, float] = {
        name: wdc_counts[name] * inv for name in team_names
    }
    expected_points: dict[str, float] = {
        name: points_sums[name] * inv for name in team_names
    }
    expected_final_position: dict[str, float] = {
        name: position_sums[name] * inv for name in team_names
    }
    standings_distribution: dict[str, dict[int, float]] = {}
    for name in team_names:
        standings_distribution[name] = {
            pos: count * inv for pos, count in sorted(standings_counts[name].items())
        }

    return {
        "wdc_probabilities": wdc_probabilities,
        "expected_points": expected_points,
        "expected_final_position": expected_final_position,
        "standings_distribution": standings_distribution,
    }
