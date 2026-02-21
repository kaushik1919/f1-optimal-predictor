"""Tests for Phase 13: finite-horizon dynamic programming pit optimisation."""

from f1_engine.core.car import Car
from f1_engine.core.pit_dp import PIT_LOSS, _lap_cost, compute_optimal_strategy_dp
from f1_engine.core.strategy import Strategy
from f1_engine.core.track import Track
from f1_engine.core.tyre import MEDIUM

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_track() -> Track:
    return Track(
        name="DP Circuit",
        straight_ratio=0.6,
        overtake_coefficient=0.5,
        energy_harvest_factor=0.7,
        tyre_degradation_factor=0.05,
        downforce_sensitivity=0.50,
    )


def _sample_car() -> Car:
    return Car(
        team_name="DPTeam",
        base_speed=80.0,
        ers_efficiency=0.80,
        aero_efficiency=0.85,
        tyre_wear_rate=1.0,
        reliability=0.98,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dp_returns_strategy() -> None:
    """compute_optimal_strategy_dp must return a valid Strategy."""
    track = _sample_track()
    car = _sample_car()
    result = compute_optimal_strategy_dp(track, car, total_laps=10)

    assert isinstance(result, Strategy)
    # compound_sequence length must be len(pit_laps) + 1
    assert len(result.compound_sequence) == len(result.pit_laps) + 1
    # deploy and harvest are zero (DP ignores ERS)
    assert result.deploy_level == 0.0
    assert result.harvest_level == 0.0
    # pit_laps must be sorted and 1-based
    assert list(result.pit_laps) == sorted(result.pit_laps)
    for lap in result.pit_laps:
        assert 1 <= lap <= 10


def test_dp_strategy_valid_compounds() -> None:
    """Every compound in the returned strategy must be one of SOFT/MEDIUM/HARD."""
    track = _sample_track()
    car = _sample_car()
    result = compute_optimal_strategy_dp(track, car, total_laps=15)

    valid_names = {"SOFT", "MEDIUM", "HARD"}
    for compound in result.compound_sequence:
        assert compound.name in valid_names


def test_dp_beats_naive_no_stop_small_case() -> None:
    """On a high-degradation track, the DP strategy must be at least as
    fast as a naive zero-stop strategy over a moderate race distance.

    We use a track with high tyre degradation to make pit stops worthwhile.
    """
    track = Track(
        name="High Deg",
        straight_ratio=0.5,
        overtake_coefficient=0.4,
        energy_harvest_factor=0.6,
        tyre_degradation_factor=0.40,  # very high degradation
        downforce_sensitivity=0.50,
    )
    car = _sample_car()
    total_laps = 20

    # DP optimal strategy
    dp_strat = compute_optimal_strategy_dp(
        track, car, total_laps, starting_compound=MEDIUM
    )

    # Naive: no pit stop, stay on MEDIUM for the entire race
    naive_time: float = 0.0
    for age in range(total_laps):
        naive_time += _lap_cost(track, car, age, MEDIUM)

    # Evaluate DP strategy total time
    dp_time: float = 0.0
    pit_set = set(dp_strat.pit_laps)
    stint_idx = 0
    tyre_age = 0
    for lap_1based in range(1, total_laps + 1):
        compound = dp_strat.compound_sequence[stint_idx]
        if lap_1based in pit_set:
            dp_time += PIT_LOSS
            stint_idx += 1
            compound = dp_strat.compound_sequence[stint_idx]
            tyre_age = 0

        dp_time += _lap_cost(track, car, tyre_age, compound)
        tyre_age += 1

    assert (
        dp_time <= naive_time + 1e-6
    ), f"DP ({dp_time:.2f}s) should be <= naive ({naive_time:.2f}s)"
