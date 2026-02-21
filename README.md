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
        updating.py      -- Latent performance updating engine (Phase 6).
        sensitivity.py   -- Sensitivity and volatility analysis engine (Phase 7).
        driver.py        -- Driver frozen dataclass (Phase 10).
        team.py          -- Team model pairing a Car with two Drivers (Phase 10).

    data_ingestion/
        __init__.py      -- Data ingestion package.
        fastf1_loader.py -- FastF1 session loader and parameter calibration (Phase 8).

data/
    calendar_2026.yaml   -- 2026 season race calendar (full per-track parameterisation).

tests/
    test_physics.py      -- Core physics test suite.
    test_stint.py        -- Energy, tyre, stint, and strategy tests (Phase 2).
    test_race.py         -- Race simulator tests (Phase 3).
    test_monte_carlo.py  -- Monte Carlo analytics tests (Phase 4).
    test_season.py       -- Season championship simulator tests (Phase 5).
    test_updating.py     -- Performance updating engine tests (Phase 6).
    test_sensitivity.py  -- Sensitivity and volatility analysis tests (Phase 7).
    test_fastf1_loader.py -- Data ingestion and calibration tests (Phase 8).
    test_calendar.py     -- Calendar loading and track validation tests (Phase 9).
    test_team_driver.py  -- Driver and team modelling tests (Phase 10).

scripts/
    calibrate_from_testing.py -- CLI script to calibrate from real sessions (Phase 8).
    run_weekly_pipeline.py   -- Automated weekly calibration and simulation (Phase 8).

results/
    calibrated_parameters.json -- Output of calibration script (gitignored).

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

## Phase 6 Scope

Phase 6 introduces a continuous updating engine that adjusts the latent car performance parameters after observed race results are compared against the model's prior expectations.  This closes the feedback loop between simulation and reality.

### Latent vs Observed Performance

The engine maintains a distinction between **latent** (true, hidden) performance and **observed** (noisy, race-day) outcomes.  The `PerformanceState` dataclass holds the engine's current belief about a car's underlying capability:

- `base_speed` -- believed baseline lap time.
- `ers_efficiency` -- believed ERS effectiveness.
- `reliability` -- believed mechanical reliability.

These are not directly measured; they are inferred from the gap between predicted and actual race points.

### Simple Bayesian-Style Update

`update_performance_state(prior, observed_points, expected_points, learning_rate)` computes a scalar error signal and applies proportional corrections:

```
error = observed_points - expected_points

base_speed  -= learning_rate * error * 0.01
ers_efficiency += learning_rate * error * 0.005
reliability += learning_rate * error * 0.001
```

A positive error (the car scored more than predicted) causes:

- `base_speed` to decrease (faster).
- `ers_efficiency` to increase.
- `reliability` to increase.

A negative error reverses the direction.  Reliability is clamped to `[0.0, 1.0]` after every update.

### Learning Rate Interpretation

The `learning_rate` parameter (default `0.05`) controls how aggressively the model revises its beliefs.  A higher value responds quickly to new evidence but is more sensitive to noise.  A lower value smooths out volatility but reacts slowly to genuine performance shifts.  The three sensitivity coefficients (`0.01`, `0.005`, `0.001`) encode the relative responsiveness of each parameter to the same unit of error, ensuring that base speed adjusts most and reliability adjusts least.

### Integration

`apply_updated_state(car, state)` produces a new `Car` instance with the updated latent parameters while preserving non-performance attributes (`team_name`, `aero_efficiency`, `tyre_wear_rate`).  Because `Car` is a frozen dataclass, no mutation occurs.

---

## Phase 7 Scope

Phase 7 adds a sensitivity and volatility analysis engine that quantifies how championship outcomes respond to small parameter perturbations and how unpredictable the title race is overall.

### Elasticity Approximation

Parameter sensitivity is estimated using a central-difference numerical derivative.  For a given parameter (reliability or ERS efficiency):

1. The parameter is increased by ``+delta``.
2. The parameter is decreased by ``-delta``.
3. A full season Monte Carlo simulation is run for each perturbed configuration.
4. The WDC probability for the target car is extracted from both runs.
5. The elasticity is computed as::

```
sensitivity = (WDC_plus - WDC_minus) / (2 * delta)
```

Perturbed values are clamped to valid ranges (``[0.0, 1.0]`` for both reliability and ERS efficiency).  If the effective delta collapses to zero (e.g. parameter already at a boundary), the function returns ``0.0``.

### Central Difference Method

The central-difference scheme is preferred over forward or backward differences because it achieves second-order accuracy -- the truncation error is proportional to ``delta^2`` rather than ``delta``.  This yields more stable elasticity estimates with the same computational budget (two season simulations per parameter).

### Entropy as Volatility Measure

`compute_championship_entropy(wdc_probabilities)` computes the Shannon entropy of the WDC probability distribution::

```
H = -sum(p * log(p))   for all p > 0
```

Entropy provides a single scalar summary of championship unpredictability:

- **H = 0** -- one team is certain to win; no volatility.
- **H = log(n)** -- all *n* teams are equally likely to win; maximum volatility.

Entropy is measured in nats (natural logarithm).  Zero-probability entries are skipped by convention (``0 * log(0) = 0``).

### Available Functions

- `compute_reliability_sensitivity(calendar, car, other_cars, ...)` -- elasticity of WDC probability with respect to reliability.
- `compute_ers_sensitivity(calendar, car, other_cars, ...)` -- elasticity of WDC probability with respect to ERS efficiency.
- `compute_championship_entropy(wdc_probabilities)` -- Shannon entropy of the championship distribution.

All Monte Carlo calls use seeded randomness for full reproducibility.

---

## Phase 8 Scope

Phase 8 adds external data ingestion using the FastF1 library to calibrate car parameters from real-world Formula 1 testing or race sessions.

### Data Source

The engine uses [FastF1](https://theoehrly.github.io/Fast-F1/) (``fastf1>=3.0.0``) to download official timing data from the FIA.  Session data is cached locally in ``fastf1_cache/`` after the first download, so subsequent runs do not require internet access.

``load_session_data(year, event, session)`` loads any available session (practice, qualifying, race, or pre-season testing) and returns a ``pandas.DataFrame`` of lap records.

### Parameter Estimation

``estimate_team_parameters(laps_df)`` processes the lap data and produces per-team parameter estimates:

- **base_speed** -- the arithmetic mean of valid lap times in seconds.  This maps directly to ``Car.base_speed`` (lower is faster).
- **reliability** -- computed as ``1 - (retirements / total_laps)`` where a retirement is approximated by any lap with a missing ``LapTime``.  Clamped to ``[0.0, 1.0]``.
- **ers_efficiency** -- a proxy computed as the inverse of lap-time standard deviation.  Teams with lower variance are assumed to manage ERS deployment more effectively.  Clamped to ``[0.0, 1.0]``.

The output is a nested dictionary ``{team_name: {"base_speed": ..., "reliability": ..., "ers_efficiency": ...}}``.

### Calibration Script

``scripts/calibrate_from_testing.py`` automatically detects the latest season and most recent completed race:

1. The current year is tried first (``datetime.now().year``).
2. If no events are found, the previous year is used as a fallback.
3. The latest event whose ``EventDate <= today`` is selected.
4. Estimate parameters for every team.
5. Print a structured summary.
6. Write results to ``results/calibrated_parameters.json``.

### Limitations

- **Proxy quality** -- ERS efficiency is approximated via lap-time consistency, not direct energy telemetry.  True ERS calibration would require power-unit data not available through FastF1.
- **Retirement detection** -- Missing lap times are used as a retirement signal, but they can also indicate pit stops or deleted laps.  More sophisticated filtering (e.g. excluding in-laps and out-laps) would improve accuracy.
- **Session specificity** -- Parameters estimated from a single session may not generalise.  Averaging across multiple sessions or weighting by recency is recommended for production use.
- **Network dependency** -- The first load of any session requires internet access.  Subsequent runs use the local cache.

---

## Phase 9 -- Track Environment Modelling

Every circuit on the 2026 calendar is now described by a five-element numerical vector stored in ``data/calendar_2026.yaml``:

| Parameter                | What it captures                              | Range   |
|--------------------------|-----------------------------------------------|---------|
| ``straight_ratio``       | Proportion of the lap spent on full-throttle straights | [0, 1] |
| ``overtake_coefficient`` | Ease of completing an overtake                | [0, 1] |
| ``energy_harvest_factor``| ERS energy recovery potential per lap         | [0, 1] |
| ``tyre_degradation_factor`` | Circuit-specific tyre wear severity        | [0, 1] |
| ``downforce_sensitivity``| Impact of downforce level on lap time         | [0, 1] |

Tracks are grouped into five archetypes:

- **H -- High-speed** (Monza, Spa, Jeddah, Austrian, Las Vegas) -- dominant straights, high harvest, low downforce demand.
- **S -- Street circuit** (Monaco, Baku, Singapore) -- tight corners, very low overtaking probability, high downforce requirement.
- **D -- High degradation** (Bahrain, Barcelona, COTA, Qatar) -- abrasive surfaces or demanding layouts that punish tyre management.
- **T -- Technical / high-downforce** (Suzuka, Imola, Hungary, Zandvoort) -- complex corner sequences rewarding aerodynamic efficiency.
- **B -- Balanced** (Melbourne, Shanghai, Miami, Montreal, Silverstone, Mexico City, Interlagos, Abu Dhabi) -- moderate values across all five parameters.

### Why heterogeneous tracks matter

When all races share identical parameters championship outcomes are driven almost entirely by car performance.  Introducing per-track variation makes the simulator sensitive to car-track interactions: a high-ERS-efficiency car gains disproportionately at harvest-rich circuits, while a low-wear-rate car is rewarded at high-degradation venues.  This increases championship volatility and produces more realistic probability distributions.

### Loader changes

``f1_engine/config.py`` now returns ``list[Track]`` instead of ``list[dict]``.  Each YAML entry is validated on load:

1. All six fields (name plus five numeric parameters) must be present.
2. Every numeric parameter must lie in the closed interval [0, 1].
3. A ``ValueError`` is raised immediately if any entry fails validation.

Downstream consumers (``main.py``, ``scripts/run_weekly_pipeline.py``) receive fully constructed, immutable ``Track`` dataclass instances with no further parsing required.

---

## Phase 10 -- Driver-Level Modelling and Constructor Championship

Phase 10 introduces individual driver modelling and separates the World Drivers' Championship (WDC) from the World Constructors' Championship (WCC).

### Driver Model

``Driver`` is a frozen dataclass with four fields:

| Field           | Type    | Description                                          |
|-----------------|---------|------------------------------------------------------|
| ``name``        | ``str`` | Unique driver identifier.                            |
| ``team_name``   | ``str`` | Must match the parent ``Team.name``.                 |
| ``skill_offset``| ``float``| Added to every lap time (lower is faster).          |
| ``consistency`` | ``float``| Multiplier on lap-time noise (> 0; 1.0 = baseline).|

### Team Model

``Team`` pairs a ``Car`` with exactly two ``Driver`` instances.  Validation enforces:

1. Exactly two drivers per team.
2. Each driver's ``team_name`` matches the team ``name``.
3. Team ``name`` is non-empty.

The car defines shared mechanical performance (base speed, ERS, aero, reliability).  Drivers add individual variation through ``skill_offset`` and ``consistency``.

### Race-Level Changes

``simulate_race`` now accepts ``teams: list[Team]`` instead of ``cars: list[Car]``.  Each driver maintains independent ``EnergyState`` and ``TyreState``.  Lap time for a driver is:

```
lap_time = physics_lap_time(car, track, ...) + driver.skill_offset + noise
```

where ``noise ~ N(0, noise_std * driver.consistency)``.  Reliability failures remain car-based: the hazard rate derives from ``car.reliability``, applied independently to each driver.

### Championship Separation

The season simulator now returns two distinct probability distributions:

- **WDC probabilities** -- keyed by driver name; each driver accumulates individual race points.
- **WCC probabilities** -- keyed by team name; constructor points equal the sum of both drivers' race points.

Sensitivity functions perturb the car attached to a target team and evaluate the WDC probability of a specified driver from that team.

### Why Two Drivers Matter

With a single entity per team the championship is entirely determined by car performance.  Adding a second driver with distinct skill and consistency parameters introduces intra-team competition, making WDC outcomes more volatile than WCC outcomes.  This matches real-world behaviour where constructor standings tend to be more stable than driver standings across a season.

---

## Phase 11A -- Persistent Overtake Modelling

Prior to Phase 11A, a successful overtake only swapped positions in the ranking list without altering cumulative race times.  This meant the overtaken driver could immediately re-pass on the very next comparison, producing unrealistic back-and-forth oscillation and failing to reflect the physical reality of an on-track pass.

### Why Time Continuity Matters

In a real race, when one car overtakes another it physically occupies the space ahead.  Simply swapping list indices does not encode this: both drivers retain their original cumulative times, so the next sort can reverse the move.  By adjusting cumulative times at the moment of the pass, the overtake becomes *persistent* -- the new leader has a genuine time advantage and the passed driver must drive faster across subsequent laps to recover the gap.

### Pass Delta Parameter

A module-level constant ``_PASS_TIME_DELTA = 0.2`` seconds governs the time transfer on each successful pass:

```
trailer.cumulative_time = leader.cumulative_time - pass_time_delta
leader.cumulative_time += pass_time_delta
```

Cumulative times are clamped so they never become negative.  After the adjustment the two entries are swapped in the ranking list and the next adjacent comparison is skipped, preventing immediate re-swap oscillation within the same lap evaluation.

### Impact on Championship Variance

Persistent overtakes increase the influence of race-day events on final standings.  A driver who executes several overtakes accumulates a meaningful time buffer, making position gains "stick" through to the chequered flag.  Across a full season of Monte Carlo simulations, this produces wider championship probability distributions -- especially among closely matched midfield teams -- because single-race position swings compound over multiple rounds.

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

## Automated Weekly Calibration

The engine includes a fully automated weekly pipeline that keeps simulation parameters aligned with the latest real-world performance data.

### How It Works

1. **Data ingestion** -- The ``scripts/run_weekly_pipeline.py`` script uses FastF1 to download lap data from the most recently completed Formula 1 race.  Season and event detection is automatic: the current year is tried first, falling back to the previous year if no events are available.
2. **Parameter estimation** -- Per-team ``base_speed``, ``reliability``, and ``ers_efficiency`` values are computed from the observed lap data and saved to ``results/calibrated_parameters.json``.
3. **Season simulation** -- A 500-season Monte Carlo championship simulation is executed using the 2026 calendar and the freshly calibrated car parameters.  Results (WDC probabilities, expected points, expected positions) are saved to ``results/latest_weekly_simulation.json``.
4. **Auto-commit** -- If the results differ from the previous run, the GitHub Action commits and pushes the updated files.

### GitHub Action

The workflow is defined in ``.github/workflows/weekly_calibration.yml``:

- **Scheduled trigger** -- runs every Monday at 03:00 UTC.
- **Manual trigger** -- can be dispatched on demand via ``workflow_dispatch``.
- **Environment** -- ``PYTHONUNBUFFERED: 1`` ensures real-time log output.
- **No secrets required** -- the workflow uses only public FastF1 data and the default ``GITHUB_TOKEN``.

### How Results Update Automatically

After each pipeline run, the workflow stages the ``results/`` directory and checks for differences.  If any calibrated parameters or simulation outputs have changed, a commit is created with the message ``"Weekly auto-calibration update"`` and pushed to ``main``.  If nothing changed, the push step is skipped.

This means the ``results/`` directory in the repository always reflects the latest available calibration, updated weekly without manual intervention.

### Running Locally

To execute the pipeline manually:

```bash
python scripts/run_weekly_pipeline.py
```

Internet access is required on the first run to download session data.  Subsequent runs use the local FastF1 cache.

---

## License

This project is licensed under the MIT License.  See [LICENSE](LICENSE) for details.
