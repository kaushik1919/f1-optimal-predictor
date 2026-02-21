"""Configuration loader for the F1 2026 simulation engine."""

from pathlib import Path

import yaml

from f1_engine.core.track import Track

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
CALENDAR_PATH: Path = DATA_DIR / "calendar_2026.yaml"

_REQUIRED_FIELDS: tuple[str, ...] = (
    "name",
    "straight_ratio",
    "overtake_coefficient",
    "energy_harvest_factor",
    "tyre_degradation_factor",
    "downforce_sensitivity",
)

_NUMERIC_FIELDS: tuple[str, ...] = _REQUIRED_FIELDS[1:]  # all except name


def load_calendar(path: Path | None = None) -> list[Track]:
    """Load the 2026 race calendar from a YAML file.

    Each entry is validated and converted into a :class:`Track` instance.

    Args:
        path: Optional override for the calendar file path.

    Returns:
        List of :class:`Track` objects for the season.

    Raises:
        FileNotFoundError: If the calendar file does not exist.
        ValueError: If any race entry is missing fields or has
            out-of-range parameter values.
    """
    calendar_path = path or CALENDAR_PATH
    if not calendar_path.exists():
        raise FileNotFoundError(f"Calendar file not found: {calendar_path}")

    with open(calendar_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    races: list[dict] = data["races"]
    tracks: list[Track] = []

    for idx, entry in enumerate(races):
        # --- Validate required fields ---
        for field in _REQUIRED_FIELDS:
            if field not in entry:
                raise ValueError(
                    f"Race entry {idx} ({entry.get('name', '<unknown>')}) "
                    f"is missing required field '{field}'"
                )

        # --- Validate numeric ranges [0, 1] ---
        for field in _NUMERIC_FIELDS:
            val = entry[field]
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"Race entry {idx} ({entry['name']}): "
                    f"'{field}' must be numeric, got {type(val).__name__}"
                )
            if not 0.0 <= float(val) <= 1.0:
                raise ValueError(
                    f"Race entry {idx} ({entry['name']}): "
                    f"'{field}' must be in [0, 1], got {val}"
                )

        tracks.append(
            Track(
                name=str(entry["name"]),
                straight_ratio=float(entry["straight_ratio"]),
                overtake_coefficient=float(entry["overtake_coefficient"]),
                energy_harvest_factor=float(entry["energy_harvest_factor"]),
                tyre_degradation_factor=float(entry["tyre_degradation_factor"]),
                downforce_sensitivity=float(entry["downforce_sensitivity"]),
            )
        )

    return tracks
