#!/usr/bin/env python
"""Calibrate simulation car parameters from real-world session data.

This script uses FastF1 to download lap data for the latest completed
race event, estimates per-team performance parameters, and writes the
results to ``results/calibrated_parameters.json``.

The target season and event are detected automatically:

1. The current year is tried first (``datetime.now().year``).
2. If no events are found, the previous year is used as a fallback.
3. The latest event whose ``EventDate <= today`` is selected.

Usage
-----
::

    python scripts/calibrate_from_testing.py

Requirements
------------
- ``fastf1>=3.0.0`` and ``pandas>=2.0.0`` must be installed.
- Internet access is required on the first run (data is cached locally
  in ``fastf1_cache/`` afterward).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import fastf1  # type: ignore[import-untyped]
import pandas as pd

# Ensure the project root is on the import path when running as a script.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from f1_engine.data_ingestion.fastf1_loader import (  # noqa: E402
    estimate_team_parameters,
    load_session_data,
)

RESULTS_DIR: str = os.path.join(_project_root, "results")
OUTPUT_PATH: str = os.path.join(RESULTS_DIR, "calibrated_parameters.json")


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------


def _detect_season() -> int:
    """Return the current year, falling back to the previous year.

    FastF1's ``get_event_schedule`` is queried for the current year.  If
    the schedule is empty (season data not yet available), the previous
    year is returned instead.
    """
    fastf1.Cache.enable_cache("fastf1_cache")
    year: int = datetime.now().year
    try:
        schedule = fastf1.get_event_schedule(year)
        if schedule.empty:
            raise ValueError("empty schedule")
    except Exception:
        year -= 1
    return year


def _detect_latest_event(year: int) -> str:
    """Return the name of the latest completed race event for *year*.

    An event is considered completed if its ``EventDate`` is on or before
    today.  Testing events are excluded so that only race weekends are
    considered.

    Raises:
        RuntimeError: If no completed events are found.
    """
    fastf1.Cache.enable_cache("fastf1_cache")
    schedule = fastf1.get_event_schedule(year)

    today = pd.Timestamp(datetime.now().date())

    # Filter to conventional race weekends (exclude testing).
    races = schedule[schedule["EventFormat"] != "testing"]

    # Keep only events that have already started.
    completed = races[races["EventDate"] <= today]

    if completed.empty:
        raise RuntimeError(f"No completed race events found for {year}.")

    latest = completed.iloc[-1]
    return str(latest["EventName"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Auto-detect latest session, estimate parameters, and save results."""
    year: int = _detect_season()
    print(f"Detected season: {year}")

    event: str = _detect_latest_event(year)
    session: str = "R"  # Race session for best calibration data.
    print(f"Latest completed event: {event}")
    print(f"Session: {session}")
    print("(First run requires internet access; subsequent runs use cache.)")
    print()

    laps_df = load_session_data(year, event, session)
    print(f"Loaded {len(laps_df)} lap records.")
    print()

    parameters = estimate_team_parameters(laps_df)

    # ---- Print structured output -------------------------------------------
    print("=" * 60)
    print("CALIBRATED TEAM PARAMETERS")
    print("=" * 60)
    for team, attrs in sorted(parameters.items()):
        print(f"\n  {team}:")
        print(f"    base_speed     = {attrs['base_speed']:.4f} s")
        print(f"    reliability    = {attrs['reliability']:.4f}")
        print(f"    ers_efficiency = {attrs['ers_efficiency']:.6f}")
    print()

    # ---- Save to JSON ------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(parameters, fh, indent=2, sort_keys=True)
    print(f"Results written to {OUTPUT_PATH}")

    return parameters  # type: ignore[return-value]


if __name__ == "__main__":
    main()
