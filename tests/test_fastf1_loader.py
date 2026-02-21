"""Tests for Phase 8: FastF1 data ingestion and parameter calibration.

These tests exercise the calibration logic using synthetic DataFrames
so that no network access or FastF1 installation is required to run
the core test suite.
"""

from __future__ import annotations

import pandas as pd

from f1_engine.data_ingestion.fastf1_loader import estimate_team_parameters

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_laps(
    teams: list[str],
    times: list[list[float | None]],
) -> pd.DataFrame:
    """Build a minimal laps DataFrame from teams and per-lap times."""
    rows: list[dict[str, object]] = []
    for team, lap_times in zip(teams, times):
        for i, t in enumerate(lap_times, start=1):
            rows.append(
                {
                    "Team": team,
                    "LapNumber": i,
                    "LapTime": t,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mean_lap_time_as_base_speed() -> None:
    """base_speed should equal the mean of valid lap times."""
    df = _make_laps(["TeamA"], [[90.0, 92.0, 88.0]])
    result = estimate_team_parameters(df)
    expected_mean = (90.0 + 92.0 + 88.0) / 3
    assert abs(result["TeamA"]["base_speed"] - expected_mean) < 1e-9


def test_reliability_reflects_missing_laps() -> None:
    """Reliability should decrease when laps have missing times."""
    df = _make_laps(["TeamB"], [[85.0, None, 86.0, None]])
    result = estimate_team_parameters(df)
    # 2 out of 4 laps are missing -> reliability = 1 - 2/4 = 0.5
    assert abs(result["TeamB"]["reliability"] - 0.5) < 1e-9


def test_ers_efficiency_bounded() -> None:
    """ers_efficiency proxy must be clamped to [0, 1]."""
    df = _make_laps(["TeamC"], [[100.0, 100.0, 100.0]])
    result = estimate_team_parameters(df)
    ers = result["TeamC"]["ers_efficiency"]
    assert 0.0 <= ers <= 1.0


def test_multiple_teams_returned() -> None:
    """All teams present in the DataFrame must appear in the output."""
    df = _make_laps(
        ["Alpha", "Bravo", "Charlie"],
        [[80.0, 81.0], [90.0, 91.0], [85.0, 86.0]],
    )
    result = estimate_team_parameters(df)
    assert set(result.keys()) == {"Alpha", "Bravo", "Charlie"}


def test_full_reliability_when_no_missing() -> None:
    """If no laps are missing, reliability should be 1.0."""
    df = _make_laps(["TeamD"], [[90.0, 91.0, 92.0]])
    result = estimate_team_parameters(df)
    assert result["TeamD"]["reliability"] == 1.0


def test_timedelta_lap_times() -> None:
    """estimate_team_parameters should handle timedelta LapTime columns."""
    df = pd.DataFrame(
        {
            "Team": ["TeamE"] * 3,
            "LapNumber": [1, 2, 3],
            "LapTime": pd.to_timedelta([90, 91, 92], unit="s"),
        }
    )
    result = estimate_team_parameters(df)
    expected_mean = (90.0 + 91.0 + 92.0) / 3
    assert abs(result["TeamE"]["base_speed"] - expected_mean) < 1e-9
