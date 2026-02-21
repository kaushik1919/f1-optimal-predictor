"""Configuration loader for the F1 2026 simulation engine."""

from pathlib import Path
from typing import Any

import yaml

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
CALENDAR_PATH: Path = DATA_DIR / "calendar_2026.yaml"


def load_calendar(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the 2026 race calendar from a YAML file.

    Args:
        path: Optional override for the calendar file path.

    Returns:
        List of race entries from the calendar file.

    Raises:
        FileNotFoundError: If the calendar file does not exist.
    """
    calendar_path = path or CALENDAR_PATH
    if not calendar_path.exists():
        raise FileNotFoundError(f"Calendar file not found: {calendar_path}")
    with open(calendar_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data["races"]
