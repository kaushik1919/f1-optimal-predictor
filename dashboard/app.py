"""F1 2026 Season Intelligence Dashboard.

Interactive analytics dashboard built with Streamlit and Plotly.
Provides Monte Carlo championship simulation, sensitivity analysis,
Safety Car frequency display, and pit strategy distribution summaries.

Launch with::

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from f1_engine.core.car import Car
from f1_engine.core.driver import Driver
from f1_engine.core.pit_dp import compute_optimal_strategy_dp
from f1_engine.core.season import simulate_season_monte_carlo
from f1_engine.core.sensitivity import (
    compute_championship_entropy,
    compute_ers_sensitivity,
    compute_reliability_sensitivity,
)
from f1_engine.core.team import Team
from f1_engine.core.track import Track

# ---------------------------------------------------------------------------
# Default grid -- representative 2026 field
# ---------------------------------------------------------------------------

_DEFAULT_TEAMS: list[dict] = [
    {
        "name": "Red Bull Racing",
        "base_speed": 79.5,
        "ers": 0.88,
        "aero": 0.90,
        "wear": 0.95,
        "rel": 0.96,
        "d1": "Max Verstappen",
        "d2": "Yuki Tsunoda",
    },
    {
        "name": "McLaren",
        "base_speed": 79.8,
        "ers": 0.86,
        "aero": 0.88,
        "wear": 1.00,
        "rel": 0.95,
        "d1": "Lando Norris",
        "d2": "Oscar Piastri",
    },
    {
        "name": "Ferrari",
        "base_speed": 79.7,
        "ers": 0.87,
        "aero": 0.87,
        "wear": 1.02,
        "rel": 0.93,
        "d1": "Charles Leclerc",
        "d2": "Lewis Hamilton",
    },
    {
        "name": "Mercedes",
        "base_speed": 80.0,
        "ers": 0.85,
        "aero": 0.89,
        "wear": 0.98,
        "rel": 0.97,
        "d1": "George Russell",
        "d2": "Andrea Kimi Antonelli",
    },
    {
        "name": "Aston Martin",
        "base_speed": 80.3,
        "ers": 0.83,
        "aero": 0.86,
        "wear": 1.01,
        "rel": 0.94,
        "d1": "Fernando Alonso",
        "d2": "Lance Stroll",
    },
]

_DEFAULT_CALENDAR: list[dict] = [
    {
        "name": "Bahrain",
        "straight_ratio": 0.50,
        "overtake_coefficient": 0.60,
        "energy_harvest_factor": 0.65,
        "tyre_degradation_factor": 0.55,
        "downforce_sensitivity": 0.50,
        "safety_car_lambda": 0.08,
        "safety_car_resume_lambda": 0.40,
    },
    {
        "name": "Saudi Arabia",
        "straight_ratio": 0.60,
        "overtake_coefficient": 0.40,
        "energy_harvest_factor": 0.70,
        "tyre_degradation_factor": 0.45,
        "downforce_sensitivity": 0.55,
        "safety_car_lambda": 0.15,
        "safety_car_resume_lambda": 0.45,
    },
    {
        "name": "Australia",
        "straight_ratio": 0.55,
        "overtake_coefficient": 0.50,
        "energy_harvest_factor": 0.60,
        "tyre_degradation_factor": 0.50,
        "downforce_sensitivity": 0.55,
        "safety_car_lambda": 0.12,
        "safety_car_resume_lambda": 0.40,
    },
    {
        "name": "Japan",
        "straight_ratio": 0.45,
        "overtake_coefficient": 0.35,
        "energy_harvest_factor": 0.55,
        "tyre_degradation_factor": 0.40,
        "downforce_sensitivity": 0.70,
        "safety_car_lambda": 0.06,
        "safety_car_resume_lambda": 0.50,
    },
    {
        "name": "China",
        "straight_ratio": 0.55,
        "overtake_coefficient": 0.55,
        "energy_harvest_factor": 0.65,
        "tyre_degradation_factor": 0.55,
        "downforce_sensitivity": 0.50,
        "safety_car_lambda": 0.08,
        "safety_car_resume_lambda": 0.45,
    },
    {
        "name": "Monaco",
        "straight_ratio": 0.30,
        "overtake_coefficient": 0.10,
        "energy_harvest_factor": 0.40,
        "tyre_degradation_factor": 0.30,
        "downforce_sensitivity": 0.80,
        "safety_car_lambda": 0.20,
        "safety_car_resume_lambda": 0.35,
    },
    {
        "name": "Spain",
        "straight_ratio": 0.50,
        "overtake_coefficient": 0.45,
        "energy_harvest_factor": 0.60,
        "tyre_degradation_factor": 0.60,
        "downforce_sensitivity": 0.55,
        "safety_car_lambda": 0.07,
        "safety_car_resume_lambda": 0.45,
    },
    {
        "name": "Great Britain",
        "straight_ratio": 0.55,
        "overtake_coefficient": 0.50,
        "energy_harvest_factor": 0.60,
        "tyre_degradation_factor": 0.50,
        "downforce_sensitivity": 0.60,
        "safety_car_lambda": 0.06,
        "safety_car_resume_lambda": 0.50,
    },
    {
        "name": "Italy",
        "straight_ratio": 0.70,
        "overtake_coefficient": 0.65,
        "energy_harvest_factor": 0.75,
        "tyre_degradation_factor": 0.35,
        "downforce_sensitivity": 0.35,
        "safety_car_lambda": 0.05,
        "safety_car_resume_lambda": 0.55,
    },
    {
        "name": "Abu Dhabi",
        "straight_ratio": 0.55,
        "overtake_coefficient": 0.55,
        "energy_harvest_factor": 0.65,
        "tyre_degradation_factor": 0.45,
        "downforce_sensitivity": 0.50,
        "safety_car_lambda": 0.07,
        "safety_car_resume_lambda": 0.45,
    },
]

_LAPS_PER_RACE: int = 57


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_teams(specs: list[dict]) -> list[Team]:
    """Construct Team objects from specification dicts."""
    teams: list[Team] = []
    for s in specs:
        car = Car(
            team_name=s["name"],
            base_speed=s["base_speed"],
            ers_efficiency=s["ers"],
            aero_efficiency=s["aero"],
            tyre_wear_rate=s["wear"],
            reliability=s["rel"],
        )
        drivers = [
            Driver(
                name=s["d1"],
                team_name=s["name"],
                skill_offset=0.0,
                consistency=1.0,
            ),
            Driver(
                name=s["d2"],
                team_name=s["name"],
                skill_offset=0.0,
                consistency=1.0,
            ),
        ]
        teams.append(Team(name=s["name"], car=car, drivers=drivers))
    return teams


def _build_calendar(
    specs: list[dict],
    sc_enabled: bool = True,
) -> list[Track]:
    """Construct Track objects, optionally zeroing SC probabilities."""
    tracks: list[Track] = []
    for s in specs:
        sc_lambda = s.get("safety_car_lambda", 0.0) if sc_enabled else 0.0
        sc_resume = s.get("safety_car_resume_lambda", 0.0) if sc_enabled else 0.0
        tracks.append(
            Track(
                name=s["name"],
                straight_ratio=s["straight_ratio"],
                overtake_coefficient=s["overtake_coefficient"],
                energy_harvest_factor=s["energy_harvest_factor"],
                tyre_degradation_factor=s["tyre_degradation_factor"],
                downforce_sensitivity=s["downforce_sensitivity"],
                safety_car_lambda=sc_lambda,
                safety_car_resume_lambda=sc_resume,
            )
        )
    return tracks


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901
    """Entry point for the Streamlit dashboard."""
    st.set_page_config(
        page_title="F1 2026 Season Intelligence",
        layout="wide",
    )

    st.title("F1 2026 Season Intelligence Dashboard")

    # ── Sidebar ──────────────────────────────────────────────────────────
    st.sidebar.header("Simulation Parameters")

    n_seasons: int = st.sidebar.slider(
        "Monte Carlo seasons",
        min_value=50,
        max_value=5000,
        value=500,
        step=50,
    )

    sc_enabled: bool = st.sidebar.toggle("Safety Car enabled", value=True)

    measurement_variance: float = st.sidebar.slider(
        "Measurement variance (R)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
    )

    teams = _build_teams(_DEFAULT_TEAMS)
    all_drivers: list[str] = []
    for team in teams:
        for d in team.drivers:
            all_drivers.append(d.name)

    selected_driver: str = st.sidebar.selectbox(
        "Driver for sensitivity analysis",
        options=all_drivers,
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Measurement variance R = {measurement_variance:.1f} "
        f"(used by Kalman filter, shown for reference)"
    )

    # ── Section 1: Run simulation ────────────────────────────────────────
    st.header("1 -- Season Simulation")

    run_clicked: bool = st.button("Run Season Simulation")

    if run_clicked:
        calendar = _build_calendar(_DEFAULT_CALENDAR, sc_enabled=sc_enabled)

        with st.spinner("Running Monte Carlo season simulation..."):
            results = simulate_season_monte_carlo(
                calendar,
                teams,
                _LAPS_PER_RACE,
                n_seasons,
                base_seed=42,
            )

        st.session_state["results"] = results
        st.session_state["calendar"] = calendar
        st.session_state["teams"] = teams
        st.session_state["n_seasons"] = n_seasons
        st.session_state["sc_enabled"] = sc_enabled
        st.session_state["selected_driver"] = selected_driver
        st.session_state["measurement_variance"] = measurement_variance

    # Guard: only show results if available
    if "results" not in st.session_state:
        st.info(
            "Configure parameters in the sidebar, then press "
            '"Run Season Simulation".'
        )
        return

    results = st.session_state["results"]
    calendar = st.session_state["calendar"]
    teams = st.session_state["teams"]
    selected_driver = st.session_state.get("selected_driver", all_drivers[0])

    # ── Section 2: Championship probability charts ───────────────────────
    st.header("2 -- Championship Probabilities")

    col_wdc, col_wcc = st.columns(2)

    # WDC bar chart
    wdc_probs: dict[str, float] = results["wdc_probabilities"]
    wdc_sorted = sorted(wdc_probs.items(), key=lambda x: x[1], reverse=True)
    wdc_names = [x[0] for x in wdc_sorted]
    wdc_vals = [x[1] for x in wdc_sorted]

    with col_wdc:
        fig_wdc = go.Figure(
            go.Bar(
                x=wdc_vals,
                y=wdc_names,
                orientation="h",
                marker_color="#e10600",
            )
        )
        fig_wdc.update_layout(
            title="WDC Probability",
            xaxis_title="Probability",
            yaxis=dict(autorange="reversed"),
            height=400,
            margin=dict(l=160),
        )
        st.plotly_chart(fig_wdc, use_container_width=True)

    # WCC bar chart
    wcc_probs: dict[str, float] = results["wcc_probabilities"]
    wcc_sorted = sorted(wcc_probs.items(), key=lambda x: x[1], reverse=True)
    wcc_names = [x[0] for x in wcc_sorted]
    wcc_vals = [x[1] for x in wcc_sorted]

    with col_wcc:
        fig_wcc = go.Figure(
            go.Bar(
                x=wcc_vals,
                y=wcc_names,
                orientation="h",
                marker_color="#1e1e1e",
            )
        )
        fig_wcc.update_layout(
            title="WCC Probability",
            xaxis_title="Probability",
            yaxis=dict(autorange="reversed"),
            height=400,
            margin=dict(l=160),
        )
        st.plotly_chart(fig_wcc, use_container_width=True)

    # ── Section 3: Championship entropy ──────────────────────────────────
    st.header("3 -- Championship Entropy")

    entropy: float = compute_championship_entropy(wdc_probs)
    col_e1, col_e2, col_e3 = st.columns(3)
    col_e1.metric("Shannon Entropy (bits)", f"{entropy:.3f}")
    col_e2.metric(
        "Max Entropy (uniform)",
        f"{__import__('math').log2(len(wdc_probs)):.3f}",
    )
    col_e3.metric(
        "Competitiveness Ratio",
        f"{entropy / max(1e-12, __import__('math').log2(len(wdc_probs))):.2%}",
    )

    # ── Section 4: Sensitivity analysis ──────────────────────────────────
    st.header("4 -- Sensitivity Analysis")

    # Identify which team the selected driver belongs to
    target_team: Team | None = None
    other_teams: list[Team] = []
    for t in teams:
        driver_names = [d.name for d in t.drivers]
        if selected_driver in driver_names:
            target_team = t
        else:
            other_teams.append(t)

    if target_team is None:
        st.warning(f"Driver '{selected_driver}' not found in any team.")
    else:
        st.write(
            f"Sensitivity of **{selected_driver}** "
            f"({target_team.name}) WDC probability:"
        )

        sens_seasons = min(st.session_state.get("n_seasons", 500), 200)

        col_s1, col_s2 = st.columns(2)

        with st.spinner("Computing ERS sensitivity..."):
            ers_sens = compute_ers_sensitivity(
                calendar=calendar,
                team=target_team,
                other_teams=other_teams,
                driver_name=selected_driver,
                laps_per_race=_LAPS_PER_RACE,
                seasons=sens_seasons,
            )

        with st.spinner("Computing reliability sensitivity..."):
            rel_sens = compute_reliability_sensitivity(
                calendar=calendar,
                team=target_team,
                other_teams=other_teams,
                driver_name=selected_driver,
                laps_per_race=_LAPS_PER_RACE,
                seasons=sens_seasons,
            )

        col_s1.metric("ERS Efficiency Elasticity", f"{ers_sens:+.4f}")
        col_s2.metric("Reliability Elasticity", f"{rel_sens:+.4f}")

    # ── Section 5: Safety Car & pit strategy ─────────────────────────────
    st.header("5 -- Safety Car & Pit Strategy")

    col_sc, col_pit = st.columns(2)

    with col_sc:
        st.subheader("Safety Car Frequency")
        sc_state = st.session_state.get("sc_enabled", True)
        if not sc_state:
            st.write("Safety Car was **disabled** for this simulation.")
        else:
            sc_data = []
            for trk in calendar:
                expected_sc_laps = (
                    (
                        _LAPS_PER_RACE
                        * trk.safety_car_lambda
                        / max(1e-9, trk.safety_car_resume_lambda)
                    )
                    if trk.safety_car_lambda > 0
                    else 0.0
                )
                sc_data.append(
                    {"Track": trk.name, "Expected SC laps": expected_sc_laps}
                )

            fig_sc = go.Figure(
                go.Bar(
                    x=[d["Track"] for d in sc_data],
                    y=[d["Expected SC laps"] for d in sc_data],
                    marker_color="#ffa500",
                )
            )
            fig_sc.update_layout(
                title="Expected Safety Car Laps per Race",
                xaxis_title="Grand Prix",
                yaxis_title="Expected SC Laps",
                height=350,
            )
            st.plotly_chart(fig_sc, use_container_width=True)

    with col_pit:
        st.subheader("Optimal Pit Strategy (DP)")
        pit_data = []
        for t in teams:
            strat = compute_optimal_strategy_dp(calendar[0], t.car, _LAPS_PER_RACE)
            n_stops = len(strat.pit_laps)
            compounds = " -> ".join(c.name for c in strat.compound_sequence)
            pit_data.append(
                {
                    "Team": t.name,
                    "Stops": n_stops,
                    "Strategy": compounds,
                    "Pit Laps": ", ".join(str(lp) for lp in strat.pit_laps),
                }
            )

        for row in pit_data:
            st.write(
                f"**{row['Team']}**: {row['Stops']}-stop "
                f"({row['Strategy']}) -- pits on lap {row['Pit Laps'] or 'N/A'}"
            )

    # ── Footer ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "F1 2026 Season Intelligence Engine -- Phase 14 Dashboard. "
        "Core engine is not modified by this dashboard."
    )


if __name__ == "__main__":
    main()
