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
        __init__.py      -- Public API re-exports (Track, Car, lap_time).
        track.py         -- Track dataclass with validation.
        car.py           -- Car dataclass with validation.
        physics.py       -- Deterministic lap time calculation.

data/
    calendar_2026.yaml   -- 2026 season race calendar.

tests/
    test_physics.py      -- Core physics test suite.

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
