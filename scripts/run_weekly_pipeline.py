#!/usr/bin/env python
"""Weekly automated calibration and season simulation pipeline.

This script orchestrates the full weekly workflow:

1. Calibrate car parameters from the latest completed real-world race
   using FastF1.
2. Build ``Car`` instances from the calibrated parameters.
3. Run a season Monte Carlo simulation (default: 500 seasons) over the
   2026 calendar.
4. Save results to ``results/latest_weekly_simulation.json``.
5. Print a structured summary.

Usage
-----
::

    python scripts/run_weekly_pipeline.py

Requirements
------------
- ``fastf1>=3.0.0``, ``pandas>=2.0.0``, ``numpy>=1.26``,
  ``pyyaml>=6.0`` must be installed.
- Internet access is required on the first FastF1 load.
"""

from __future__ import annotations

import json
import os
import sys

# Ensure the project root is on the import path.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from f1_engine.config import load_calendar  # noqa: E402
from f1_engine.core.car import Car  # noqa: E402
from f1_engine.core.driver import Driver  # noqa: E402
from f1_engine.core.season import simulate_season_monte_carlo  # noqa: E402
from f1_engine.core.team import Team  # noqa: E402
from f1_engine.data_ingestion.fastf1_loader import (  # noqa: E402
    estimate_team_parameters,
    load_session_data,
)

# Import the calibration script's helpers.
from scripts.calibrate_from_testing import (  # noqa: E402
    _detect_latest_event,
    _detect_season,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEASONS: int = 500
LAPS_PER_RACE: int = 57
BASE_SEED: int = 2026
RESULTS_DIR: str = os.path.join(_project_root, "results")
OUTPUT_PATH: str = os.path.join(RESULTS_DIR, "latest_weekly_simulation.json")
PARAMS_PATH: str = os.path.join(RESULTS_DIR, "calibrated_parameters.json")

# Default non-calibrated attributes shared by all cars.
_DEFAULT_AERO: float = 0.85
_DEFAULT_TYRE_WEAR: float = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_teams(params: dict[str, dict[str, float]]) -> list[Team]:
    """Convert calibrated parameter dict into a list of Team instances.

    Each team gets two drivers with default skill_offset=0.0 and
    consistency=1.0, named ``<team> Driver 1`` and ``<team> Driver 2``.
    """
    teams: list[Team] = []
    for team_name, attrs in sorted(params.items()):
        car = Car(
            team_name=team_name,
            base_speed=attrs["base_speed"],
            ers_efficiency=min(max(attrs["ers_efficiency"], 0.01), 1.0),
            aero_efficiency=_DEFAULT_AERO,
            tyre_wear_rate=_DEFAULT_TYRE_WEAR,
            reliability=min(max(attrs["reliability"], 0.01), 1.0),
        )
        drivers = [
            Driver(
                name=f"{team_name} Driver 1",
                team_name=team_name,
                skill_offset=0.0,
                consistency=1.0,
            ),
            Driver(
                name=f"{team_name} Driver 2",
                team_name=team_name,
                skill_offset=0.0,
                consistency=1.0,
            ),
        ]
        teams.append(Team(name=team_name, car=car, drivers=drivers))
    return teams


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full weekly calibration and simulation pipeline."""
    print("=" * 60)
    print("WEEKLY CALIBRATION AND SIMULATION PIPELINE")
    print("=" * 60)
    print()

    # -- Step 1: Calibrate from latest race ----------------------------------
    year: int = _detect_season()
    event: str = _detect_latest_event(year)
    session: str = "R"

    print(f"[1/4] Calibrating from {year} {event} ({session})")
    laps_df = load_session_data(year, event, session)
    print(f"      Loaded {len(laps_df)} lap records.")
    parameters = estimate_team_parameters(laps_df)

    # Save calibrated parameters.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(PARAMS_PATH, "w", encoding="utf-8") as fh:
        json.dump(parameters, fh, indent=2, sort_keys=True)
    print(f"      Parameters saved to {PARAMS_PATH}")
    print()

    # -- Step 2: Build Team instances -----------------------------------------
    print("[2/4] Building Team instances from calibrated parameters")
    teams = _build_teams(parameters)
    print(f"      {len(teams)} teams loaded.")
    print()

    # -- Step 3: Load calendar and run season Monte Carlo --------------------
    print(f"[3/4] Running season Monte Carlo ({SEASONS} seasons)")
    calendar = load_calendar()
    result = simulate_season_monte_carlo(
        calendar,
        teams,
        laps_per_race=LAPS_PER_RACE,
        seasons=SEASONS,
        base_seed=BASE_SEED,
    )
    print("      Simulation complete.")
    print()

    # -- Step 4: Save and summarise ------------------------------------------
    print("[4/4] Saving results")

    output: dict[str, object] = {
        "metadata": {
            "calibration_year": year,
            "calibration_event": event,
            "seasons_simulated": SEASONS,
            "laps_per_race": LAPS_PER_RACE,
            "base_seed": BASE_SEED,
        },
        "wdc_probabilities": result["wdc_probabilities"],
        "wcc_probabilities": result["wcc_probabilities"],
        "expected_driver_points": result["expected_driver_points"],
        "expected_team_points": result["expected_team_points"],
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, sort_keys=True)
    print(f"      Results saved to {OUTPUT_PATH}")
    print()

    # -- Structured summary --------------------------------------------------
    print("=" * 60)
    print("WDC (DRIVER) PROBABILITY SUMMARY")
    print("=" * 60)
    sorted_drivers = sorted(
        result["wdc_probabilities"].items(),
        key=lambda x: x[1],
        reverse=True,
    )
    for rank, (drv, prob) in enumerate(sorted_drivers, start=1):
        exp_pts = result["expected_driver_points"][drv]
        print(
            f"  {rank:2d}. {drv:<30s}  " f"WDC: {prob:.3f}  " f"E[pts]: {exp_pts:.1f}"
        )
    print()
    print("=" * 60)
    print("WCC (CONSTRUCTOR) PROBABILITY SUMMARY")
    print("=" * 60)
    sorted_teams = sorted(
        result["wcc_probabilities"].items(),
        key=lambda x: x[1],
        reverse=True,
    )
    for rank, (team, prob) in enumerate(sorted_teams, start=1):
        exp_pts = result["expected_team_points"][team]
        print(
            f"  {rank:2d}. {team:<25s}  " f"WCC: {prob:.3f}  " f"E[pts]: {exp_pts:.1f}"
        )
    print()
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
