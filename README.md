# F1 2026 Strategy Simulation Engine

A production-grade, modular simulation engine for Formula 1 2026 hybrid race strategy analysis.

---

## Project Overview

This engine provides a deterministic, physics-based model for simulating Formula 1 lap times under the 2026 technical regulations. It is designed to be extensible, testable, and deployable, serving as the foundation for advanced race strategy optimisation tools.

The simulation captures the interplay between car performance, circuit characteristics, tyre degradation, and ERS (Energy Recovery System) deployment to produce realistic lap time estimates.

---

## Phase 1 Scope

Phase 1 delivers the deterministic simulation core:

- **Track model** -- Immutable dataclass representing circuit-specific properties such as straight ratio, overtaking coefficient, energy harvest factor, tyre degradation factor, and downforce sensitivity.
- **Car model** -- Immutable dataclass representing constructor-specific performance attributes including base speed, ERS efficiency, aerodynamic efficiency, tyre wear rate, and reliability.
- **Deterministic lap time function** -- A physics-based calculation combining base speed, aerodynamic effects, tyre degradation, and ERS deployment into a single lap time value.
- **2026 calendar configuration** -- YAML-based listing of all 24 scheduled races.
- **CLI entrypoint** -- Demonstration script exercising the core simulation.
- **Test skeleton** -- Pytest-based test suite validating core physics.
- **CI pipeline** -- GitHub Actions workflow running linting (ruff, black) and tests on every push and pull request.
- **Docker support** -- Production-ready container image for reproducible execution.

Stochastic elements, strategy optimisation, and multi-car simulation are deferred to later phases.

---

## Architecture

```
f1_engine/
    __init__.py          -- Package root; version metadata.
    config.py            -- YAML calendar loader and path constants.
    core/
        __init__.py      -- Public API re-exports.
        track.py         -- Track dataclass with validation.
        car.py           -- Car dataclass with validation.
        physics.py       -- Deterministic lap time calculation.
        energy.py        -- ERS battery state model (Phase 2).
        tyre.py          -- Tyre wear state model (Phase 2).
        strategy.py      -- Constant strategy dataclass (Phase 2).
        stint.py         -- Stint simulation and strategy search (Phase 2).
        race.py          -- Multi-car stochastic race simulator (Phase 3).
        monte_carlo.py   -- Monte Carlo race analytics engine (Phase 4).
        season.py        -- Full-season Monte Carlo championship simulator (Phase 5).

data/
    calendar_2026.yaml   -- 2026 season race calendar.

tests/
    test_physics.py      -- Core physics test suite.
    test_stint.py        -- Energy, tyre, stint, and strategy tests (Phase 2).
    test_race.py         -- Race simulator tests (Phase 3).
    test_monte_carlo.py  -- Monte Carlo analytics tests (Phase 4).
    test_season.py       -- Season championship simulator tests (Phase 5).

main.py                  -- CLI entrypoint.
```

### Lap Time Formula

```
lap_time = base_component + aero_component + tyre_component - ers_component
```

Where:

- `base_component = car.base_speed`
- `aero_component = track.downforce_sensitivity * (1 - car.aero_efficiency)`
- `tyre_component = tyre_age * track.tyre_degradation_factor * car.tyre_wear_rate`
- `ers_component = deploy_level * car.ers_efficiency`

---

## Phase 2 Scope

Phase 2 adds deterministic energy state modelling and stint simulation on top of the Phase 1 core. All behaviour remains single-car and fully deterministic.

### Energy Model

`EnergyState` tracks the ERS battery over a stint:

- **deploy(amount)** -- Deploys `min(amount, current_charge)` and reduces the battery level. Returns actual energy deployed.
- **harvest(amount)** -- Increases the battery level by the requested amount, capped at `max_charge`. Returns actual energy harvested.

Energy harvest per lap is computed as:

```
harvest_amount = track.energy_harvest_factor * strategy.harvest_level
```

The battery is bounded: charge is always in the interval `[0, max_charge]`.

### Tyre Model

`TyreState` tracks integer tyre age (laps completed) and carries a `wear_rate_multiplier` forwarded from the car configuration.

### Strategy Model

`Strategy` is an immutable dataclass holding two scalar multipliers, each in `[0.0, 1.0]`:

- `deploy_level` -- fraction of ERS energy to deploy per lap.
- `harvest_level` -- fraction of maximum harvestable energy to recover per lap.

### Stint Simulation

`simulate_stint(track, car, strategy, laps)` executes the following loop for each lap:

1. Harvest energy (scaled by track and strategy).
2. Deploy energy (bounded by battery state).
3. Compute lap time via the physics model using actual deployment.
4. Increment tyre age.

Returns a dictionary containing `total_time`, `lap_times`, `energy_trace`, and `tyre_trace`.

### Constant Strategy Search

`find_best_constant_deploy(track, car, laps)` evaluates deploy levels `[0.0, 0.2, 0.4, 0.6, 0.8]` with harvest fixed at `1.0` and returns the strategy yielding the lowest total stint time.

---

## Phase 3 Scope

Phase 3 introduces a multi-car stochastic race simulator built on top of the deterministic core from Phases 1 and 2. All randomness is seeded and reproducible.

### Stochastic Lap Time Model

Each lap time begins with the deterministic value from the Phase 1 physics model. A Gaussian noise term is then added:

```
observed_lap_time = deterministic_lap_time + N(0, noise_std)
```

Setting `noise_std = 0.0` recovers fully deterministic behaviour. The random generator is created per call via `numpy.random.default_rng(seed)`, ensuring no global state is modified.

### Reliability Hazard Model

Each lap, every active car faces a probability of mechanical retirement (DNF). The hazard rate per lap is derived from the car's reliability attribute:

```
hazard = 1 - exp(-(1 - car.reliability))
```

A car with `reliability = 1.0` has zero hazard. Lower reliability values produce progressively higher per-lap retirement probabilities.

### Overtake Logistic Model

After each lap, adjacent cars in the running order are evaluated for position swaps. An overtake is attempted when the cumulative time gap is less than 1.0 second. The pass probability follows a logistic function:

```
delta = trailing_last_lap - leading_last_lap
pass_prob = 1 / (1 + exp(-3.0 * delta * track.overtake_coefficient))
```

This means a trailing car that posted a faster lap than the leader is more likely to complete the overtake, modulated by the circuit's overtaking difficulty.

### Race Output

`simulate_race` returns a `RaceResult` dataclass containing:

- `final_classification` -- all team names ordered by finishing position (DNFs appended).
- `dnf_list` -- team names that retired.
- `lap_times` -- per-car list of recorded lap times.

---

## Phase 4 Scope

Phase 4 adds a Monte Carlo analytics engine that runs many seeded replications of the Phase 3 race simulator and aggregates results into probability distributions.

### Monte Carlo Methodology

`simulate_race_monte_carlo(track, cars, laps, simulations, base_seed)` executes `simulations` independent race replications.  Replication *i* uses `seed = base_seed + i`, ensuring:

- Full reproducibility when the same `base_seed` and `simulations` count are provided.
- No global random state is modified.
- Each replication is statistically independent.

After all replications complete, the following statistics are computed per team:

- **Winner probability** -- fraction of simulations in which the team finished first.
- **Podium probability** -- fraction of simulations in which the team finished in the top 3.
- **Expected finishing position** -- arithmetic mean of finishing positions.
- **Expected championship points** -- arithmetic mean of points awarded per the standard F1 table `[25, 18, 15, 12, 10, 8, 6, 4, 2, 1]`.
- **Finish distribution** -- for each possible finishing position, the probability of ending up there.

### Statistical Interpretation

All probabilities are empirical frequencies.  Increasing `simulations` improves the precision of every estimate.  Winner probabilities across all teams sum to exactly 1.0.  Each team's finish distribution also sums to 1.0.

### Seed Handling

The `base_seed` parameter controls reproducibility at the ensemble level.  Individual race randomness (Gaussian noise, reliability hazard, overtake draws) is governed by `numpy.random.default_rng(seed)` inside each `simulate_race` call, inheriting the Phase 3 seeding contract.

---

## Phase 5 Scope

Phase 5 adds a full-season Monte Carlo championship simulator that aggregates race-level results across every round of the calendar into season-long championship probability distributions.

### Season Monte Carlo Methodology

`simulate_season_monte_carlo(calendar, cars, laps_per_race, seasons, base_seed)` simulates `seasons` complete championships.  For each simulated season, every race on the calendar is run using the Phase 3 race simulator.  Points are accumulated across all rounds using the standard F1 table `[25, 18, 15, 12, 10, 8, 6, 4, 2, 1]`.

After all seasons complete, the following statistics are computed per team:

- **WDC probability** -- fraction of simulated seasons in which the team accumulated the most points.
- **Expected season points** -- arithmetic mean of total championship points.
- **Expected final position** -- arithmetic mean of final standings position.
- **Standings distribution** -- for each possible final position, the probability of finishing there.

### Seed Strategy

Each individual race within a season is seeded deterministically:

```
season_seed = base_seed + season_index
race_seed   = season_seed + race_index * 1000
```

This two-level scheme ensures:

- Full reproducibility given the same `base_seed` and `seasons` count.
- Statistical independence between races within a season.
- Statistical independence between simulated seasons.
- No global random state is modified.

### Computational Complexity

Total race simulations executed: `seasons * len(calendar)`.  For a 24-race calendar with 1000 seasons, the engine runs 24000 individual race simulations.  Each race inherits the O(laps * cars) cost from Phase 3.

### Championship Resolution

WDC probabilities across all teams sum to exactly 1.0.  Ties in season points are broken by the order returned from NumPy's argsort (lowest index first), consistent across runs due to seeded randomness.

---

## How to Run Locally

### Prerequisites

- Python 3.11 or later
- pip

### Setup

```bash
pip install -r requirements.txt
```

### Run the simulation

```bash
python main.py
```

### Run tests

```bash
pytest -v
```

### Run linters

```bash
ruff check .
black --check .
```

---

## How to Run with Docker

### Build the image

```bash
docker build -t f1-2026-engine .
```

### Run the container

```bash
docker run --rm f1-2026-engine
```

---

## CI Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and pull request to `main`. It executes the following steps:

1. Checkout the repository.
2. Set up Python 3.11.
3. Install project dependencies.
4. Run `ruff check .` for linting.
5. Run `black --check .` for formatting verification.
6. Run `pytest -v` for the test suite.

Any failure in linting, formatting, or tests will fail the pipeline.

---

## License

This project is provided as-is for research and simulation purposes.
