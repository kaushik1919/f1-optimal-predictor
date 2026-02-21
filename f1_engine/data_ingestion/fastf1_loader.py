"""FastF1-based data loader and parameter calibration for the F1 2026 engine.

This module provides functions to:

1. Load real-world session data (practice, qualifying, race, or testing)
   from the FastF1 API.
2. Estimate car performance parameters from the retrieved lap data so
   that the simulation engine can be calibrated against observed reality.

FastF1 caches data locally after the first download.  Internet access is
required on the initial load of any session.
"""

from __future__ import annotations

from typing import Any

import fastf1  # type: ignore[import-untyped]
import pandas as pd

# ---------------------------------------------------------------------------
# Session loader
# ---------------------------------------------------------------------------


def load_session_data(
    year: int,
    event: str,
    session: str,
) -> pd.DataFrame:
    """Load lap-level data for a given session via FastF1.

    Enables the local disk cache on first call (``fastf1_cache/``).

    Args:
        year: Season year (e.g. ``2023``).
        event: Grand Prix or testing event name (e.g. ``"Bahrain"``).
        session: Session identifier accepted by FastF1 (e.g.
            ``"FP1"``, ``"Q"``, ``"R"``, ``"Practice 1"``).

    Returns:
        A :class:`pandas.DataFrame` of lap records as returned by
        ``session.laps``.
    """
    fastf1.Cache.enable_cache("fastf1_cache")

    sess = fastf1.get_session(year, event, session)
    sess.load()

    return sess.laps


# ---------------------------------------------------------------------------
# Parameter estimation
# ---------------------------------------------------------------------------


def estimate_team_parameters(
    laps_df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Estimate simulation parameters per team from observed lap data.

    For each team present in *laps_df* the following mapping is produced:

    - ``base_speed`` -- mean lap time in seconds.  This directly maps to
      the ``Car.base_speed`` attribute (lower is faster).
    - ``reliability`` -- ``1 - (retirements / total_laps)`` where a
      retirement is approximated by any lap whose ``LapTime`` is NaT
      (missing).
    - ``ers_efficiency`` -- a proxy computed as the inverse of lap-time
      standard deviation.  Teams with more consistent (lower-variance)
      lap times are assumed to manage ERS deployment more effectively.

    Args:
        laps_df: DataFrame of race laps, expected to contain at least
            the columns ``Team``, ``LapTime``, and ``LapNumber``.

    Returns:
        Nested dictionary ``{team_name: {"base_speed": …, "reliability": …,
        "ers_efficiency": …}}``.
    """
    results: dict[str, dict[str, float]] = {}

    # Convert LapTime to total-seconds if it is a timedelta.
    if not laps_df.empty and pd.api.types.is_timedelta64_dtype(laps_df["LapTime"]):
        laps_df = laps_df.copy()
        laps_df["LapTimeSec"] = laps_df["LapTime"].dt.total_seconds()
    elif "LapTimeSec" not in laps_df.columns:
        laps_df = laps_df.copy()
        laps_df["LapTimeSec"] = pd.to_numeric(laps_df["LapTime"], errors="coerce")

    teams: list[Any] = sorted(laps_df["Team"].dropna().unique().tolist())

    for team in teams:
        team_laps = laps_df[laps_df["Team"] == team]
        total_laps: int = len(team_laps)

        valid = team_laps["LapTimeSec"].dropna()
        retirements: int = total_laps - len(valid)

        if valid.empty:
            mean_time = 0.0
            std_time = 1.0
        else:
            mean_time = float(valid.mean())
            std_time = float(valid.std()) if len(valid) > 1 else 1.0

        # Clamp std away from zero to avoid division-by-zero.
        if std_time < 1e-6:
            std_time = 1e-6

        reliability: float = 1.0 - retirements / total_laps if total_laps > 0 else 1.0
        reliability = max(0.0, min(1.0, reliability))

        ers_efficiency: float = 1.0 / std_time
        ers_efficiency = max(0.0, min(1.0, ers_efficiency))

        results[str(team)] = {
            "base_speed": mean_time,
            "reliability": reliability,
            "ers_efficiency": ers_efficiency,
        }

    return results
