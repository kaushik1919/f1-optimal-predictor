"""Tests for Phase 9: full 24-track parameterisation and calendar loading."""

from f1_engine.config import load_calendar
from f1_engine.core.track import Track

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_PARAM_FIELDS: list[str] = [
    "straight_ratio",
    "overtake_coefficient",
    "energy_harvest_factor",
    "tyre_degradation_factor",
    "downforce_sensitivity",
]


def test_calendar_loads_all_24_tracks() -> None:
    """The 2026 calendar must contain exactly 24 Track objects."""
    calendar = load_calendar()
    assert len(calendar) == 24
    for track in calendar:
        assert isinstance(track, Track)


def test_all_tracks_have_unique_parameters() -> None:
    """Every track must have a unique parameter vector (no copy-paste)."""
    calendar = load_calendar()
    vectors: list[tuple[float, ...]] = []
    for track in calendar:
        vec = (
            track.straight_ratio,
            track.overtake_coefficient,
            track.energy_harvest_factor,
            track.tyre_degradation_factor,
            track.downforce_sensitivity,
        )
        vectors.append(vec)
    # All vectors must be distinct.
    assert len(set(vectors)) == len(
        vectors
    ), "Duplicate parameter vectors detected among tracks"


def test_track_parameters_in_bounds() -> None:
    """All numeric track parameters must be in [0.0, 1.0]."""
    calendar = load_calendar()
    for track in calendar:
        for field_name in _PARAM_FIELDS:
            value = getattr(track, field_name)
            assert (
                0.0 <= value <= 1.0
            ), f"{track.name}.{field_name} = {value} is out of [0, 1]"


def test_track_names_are_unique() -> None:
    """Every track in the calendar must have a unique name."""
    calendar = load_calendar()
    names = [t.name for t in calendar]
    assert len(set(names)) == len(names)


def test_track_names_non_empty() -> None:
    """No track should have an empty name."""
    calendar = load_calendar()
    for track in calendar:
        assert track.name, "Track name must not be empty"
