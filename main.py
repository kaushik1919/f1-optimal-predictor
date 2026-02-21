"""CLI entrypoint for the F1 2026 Strategy Simulation Engine."""

from __future__ import annotations

import sys

from f1_engine import __version__
from f1_engine.config import load_calendar
from f1_engine.core.car import Car
from f1_engine.core.physics import lap_time
from f1_engine.core.track import Track


def main() -> None:
    """Run a demonstration of the deterministic simulation core."""
    print(f"F1 2026 Strategy Simulation Engine v{__version__}")
    print("=" * 56)

    # -- Load calendar --------------------------------------------------------
    calendar = load_calendar()
    print(f"\n2026 Calendar: {len(calendar)} races loaded")
    for i, race in enumerate(calendar, start=1):
        print(f"  R{i:02d}: {race['name']}")

    # -- Sample track and car -------------------------------------------------
    track = Track(
        name="Bahrain International Circuit",
        straight_ratio=0.55,
        overtake_coefficient=0.65,
        energy_harvest_factor=0.70,
        tyre_degradation_factor=0.06,
        downforce_sensitivity=2.5,
    )

    car = Car(
        team_name="Scuderia Example",
        base_speed=78.5,
        ers_efficiency=0.82,
        aero_efficiency=0.88,
        tyre_wear_rate=1.05,
        reliability=0.96,
    )

    print(f"\nTrack : {track.name}")
    print(f"Car   : {car.team_name}")
    print("-" * 56)

    # -- Simulate a stint -----------------------------------------------------
    stint_laps = 15
    deploy = 0.6
    print(f"\nSimulating {stint_laps}-lap stint (ERS deploy={deploy}):\n")
    print(f"  {'Lap':>3}  {'Tyre Age':>8}  {'Lap Time (s)':>12}")
    print(f"  {'---':>3}  {'--------':>8}  {'------------':>12}")

    for lap_num in range(1, stint_laps + 1):
        tyre_age = float(lap_num)
        t = lap_time(track, car, tyre_age=tyre_age, deploy_level=deploy)
        print(f"  {lap_num:3d}  {tyre_age:8.1f}  {t:12.4f}")

    print("\nPhase 1 simulation complete.")


if __name__ == "__main__":
    sys.exit(main() or 0)
