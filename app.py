"""
Rooftop Solar Forecast Application

This Streamlit app uses Monte Carlo simulations to forecast rooftop solar capacity,
energy production, revenues, and installation costs with Calgary, Alberta as a baseline. 
It provides several tabs for controlling simulations, viewing forecasts, analyzing NPV, 
exploring change distributions, tracking time-series metrics, and performing sensitivity analyses.

Modules:
    - plotting: Contains functions for generating Plotly charts.
    - layout: Provides layout functions, scenario controls, and detailed explanations.
    - analysis: Contains the Monte Carlo simulation logic and sensitivity analysis functions.
"""

import os

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import plotting
import layout
import analysis

import yaml

# -----------------------------------------------------------------------------
# Page Configuration and Session State Initialization
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Rooftop Solar Forecast", layout="wide")

# Initialize session state variables if not already set.
if "years" not in st.session_state:
    st.session_state.years = 25

if "iterations" not in st.session_state:
    st.session_state.iterations = 1000

if "config" not in st.session_state:
    st.session_state["config"] = {}

# App Title
st.title("Rooftop Solar Growth & Energy Forecast")

# -----------------------------------------------------------------------------
# Create Tabs
# -----------------------------------------------------------------------------
tabs = st.tabs(
    [
        "Info",
        "Controls",
        "Forecasts",
        "NPV",
        "Change Distribution",
        "Timeseries Metrics",
        "Capacity Sensitivity",
        "Energy Sensitivity",
        "Revenue Sensitivity",
    ]
)

# -----------------------------------------------------------------------------
# Tab 1: Info
# -----------------------------------------------------------------------------
with tabs[0]:
    # Displays a detailed help modal with instructions and simulation details.
    layout.show_help_modal()

# -----------------------------------------------------------------------------
# Tab 2: Controls & Distributions
# -----------------------------------------------------------------------------
with tabs[1]:
    st.header("Controls & Distributions")

    # Scenario Management Container
    with st.container():
        st.header("Manage Forecast Scenarios")
        if "scenarios" not in st.session_state:
            st.session_state.scenarios = []

    try:
        template_files = [f for f in os.listdir("scenarios") if f.endswith(".yaml")]
        templates = [os.path.splitext(f)[0] for f in template_files]
    except Exception as e:
        st.error("Error reading scenarios folder: " + str(e))
        templates = []

    col1, col2 = st.columns(2)
    with col1:
        selected_template = st.selectbox("Select Scenario Template", templates)

    with col2:
        scenario_name = st.text_input("Scenario Name", value=selected_template)

    if st.button("Create scenario"):
        if scenario_name not in st.session_state.scenarios:
            config_path = os.path.join("scenarios", f"{selected_template}.yaml")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
            st.session_state.scenarios.append(scenario_name)
            st.session_state[scenario_name] = {}
            st.session_state["config"][scenario_name] = config

    # Display a tab for each scenario if one or more exist.
    if st.session_state.scenarios:
        scenario_tabs = st.tabs(st.session_state.scenarios)
        for idx, (tab, scenario) in enumerate(
            zip(scenario_tabs, st.session_state.scenarios)
        ):
            with tab:
                layout.create_standard_scenario(idx, scenario)

# -----------------------------------------------------------------------------
# Tab 3: Forecasts
# -----------------------------------------------------------------------------
with tabs[2]:
    st.header("Forecasts")
    with st.expander("Detailed Forecast Calculation Information"):
        layout.monte_carlo_simulation_explanation()

    # Pre-create empty Plotly figures for each forecast type.
    for fig_name in ["capacity_fig", "energy_fig", "revenue_fig", "cost_fig"]:
        st.session_state[fig_name] = go.Figure()

    # Update forecast plots for each scenario.
    for scenario in st.session_state.scenarios:
        plotting.update_forecast_plots(scenario)

    # Create a combined figure with shared x-axis for better comparison.
    combined_forecast_fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            "Capacity Forecast (MW)",
            "Energy Forecast (kWh/year)",
            "PV Revenue Forecast ($/year)",
            "Installation Cost Forecast ($/year)",
        ),
    )

    capacity_fig = st.session_state["capacity_fig"]
    energy_fig = st.session_state["energy_fig"]
    revenue_fig = st.session_state["revenue_fig"]
    cost_fig = st.session_state["cost_fig"]

    # Add traces from individual forecast figures to the combined figure.
    for trace in capacity_fig.data:
        combined_forecast_fig.add_trace(trace, row=1, col=1)
    for trace in energy_fig.data:
        combined_forecast_fig.add_trace(trace, row=2, col=1)
    for trace in revenue_fig.data:
        combined_forecast_fig.add_trace(trace, row=3, col=1)
    for trace in cost_fig.data:
        combined_forecast_fig.add_trace(trace, row=4, col=1)

    # Update layout and axes labels.
    combined_forecast_fig.update_layout(
        height=1200, width=900, showlegend=True, hovermode="x unified"
    )
    combined_forecast_fig.update_xaxes(title_text="Year", row=4, col=1)
    combined_forecast_fig.update_yaxes(title_text="Capacity (MW)", row=1, col=1)
    combined_forecast_fig.update_yaxes(title_text="Energy (kWh/year)", row=2, col=1)
    combined_forecast_fig.update_yaxes(title_text="Revenue ($/year)", row=3, col=1)
    combined_forecast_fig.update_yaxes(title_text="Cost ($/year)", row=4, col=1)

    st.plotly_chart(combined_forecast_fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 4: NPV
# -----------------------------------------------------------------------------
with tabs[3]:
    st.header("NPV Forecast")
    with st.expander("Detailed NPV Information"):
        layout.npv_explanation()

    forecasts = ["Revenue", "Install Cost"]
    forecast_choice = st.selectbox("Pick", options=forecasts, key="npv_select")

    discount_rate = st.slider(
        "Discount Rate",
        min_value=0.0,
        max_value=0.2,
        value=0.08,
        step=0.01,
        help="Select the discount rate (as a decimal)",
        key="npv_of_forecast_df",
    )

    is_cumulative = st.checkbox("Cumulative NPV")
    plotting.create_npv_plot(forecast_choice, discount_rate, cumulative=is_cumulative)

# -----------------------------------------------------------------------------
# Tab 5: Change Distribution
# -----------------------------------------------------------------------------
with tabs[4]:
    st.header("Change Distributions")
    with st.expander("Detailed Change Distribution Information"):
        layout.change_distribution_explanation()

    forecasts = ["Capacity", "Energy", "Revenue", "Install Cost"]
    forecast_choice = st.selectbox("Pick", options=forecasts, key="change_dist")

    if st.session_state.scenarios:
        plotting.plot_all_scenarios_pdf_cdf_histograms(forecast_choice)

# -----------------------------------------------------------------------------
# Tab 6: Timeseries Metrics
# -----------------------------------------------------------------------------
with tabs[5]:
    st.header("Time Series Metrics")
    with st.expander("Detailed Time Series Metrics Information"):
        layout.time_series_metrics_explanation()

    metrics = [
        "New Panel Power Rating (W/m²)",
        "Panel Degradation Factor (%)",
        "Panel Power Gain Factor (%)",
        "Install Discount Factor (%)",
        "Median Installed Area (km²)",
        "Median Installed City Coverage (%)",
        "Median Incremental Capacity (MW)",
    ]
    metric_choice = st.selectbox(
        "Pick a metric", options=metrics, key="tracking_select"
    )
    if st.session_state.scenarios:
        plotting.plot_tracking_data(metric_choice)

# -----------------------------------------------------------------------------
# Tab 7: Capacity Sensitivity
# -----------------------------------------------------------------------------
with tabs[6]:
    st.header("Sensitivity Analysis: Total Capacity")
    if st.session_state.scenarios:
        scenario = st.selectbox(
            "Select the scenario",
            options=st.session_state.scenarios,
            key="select_capacity_sensitivity",
        )
        bounds = layout.get_sensitivity_bounds("Capacity")

        sensitivity_results = analysis.sensitivity_total(
            scenario, bounds, "Capacity", method="max"
        )
        plotting.create_tornado_figure(
            sensitivity_results,
            "Sensitivity of Max Capacity to Parameters",
            "Capacity (MW)",
        )

# -----------------------------------------------------------------------------
# Tab 8: Energy Sensitivity
# -----------------------------------------------------------------------------
with tabs[7]:
    st.header("Sensitivity Analysis: Total Energy Production")

    with st.expander("Detailed Energy SensitivityAnalysis Information"):
        layout.show_energy_sensitivity_info()

    if st.session_state.scenarios:
        scenario = st.selectbox(
            "Select the scenario",
            options=st.session_state.scenarios,
            key="select_energy_sensitivity",
        )
        bounds = layout.get_sensitivity_bounds("Total Energy")

        sensitivity_results = analysis.sensitivity_total(scenario, bounds, "Energy")
        sensitivity_results["Low"] /= 1e6
        sensitivity_results["High"] /= 1e6
        sensitivity_results["Baseline"] /= 1e6
        plotting.create_tornado_figure(
            sensitivity_results,
            "Sensitivity of Total Energy to Parameters",
            "Total Energy (GWh)",
        )


# -----------------------------------------------------------------------------
# Tab 9: Revenue Sensitivity
# -----------------------------------------------------------------------------
with tabs[8]:
    st.header("Sensitivity Analysis: Cumulative NPV for Revenue")

    with st.expander("Detailed NPV Sensitivity Analysis Information"):
        layout.show_revenue_sensitivity_info()

    discount_rate = st.slider(
        "Discount Rate",
        min_value=0.0,
        max_value=0.2,
        value=0.08,
        step=0.01,
        help="Select the discount rate (as a decimal)",
        key="sensitivity_df",
    )
    if st.session_state.scenarios:
        scenario = st.selectbox(
            "Select the scenario",
            options=st.session_state.scenarios,
            key="select_npv_sensitivity",
        )
        bounds = layout.get_sensitivity_bounds("Revenue NPV")
        sensitivity_results = analysis.sensitivity_cumulative_npv_revenue(
            scenario, bounds, discount_rate
        )
        plotting.create_tornado_figure(
            sensitivity_results,
            "Sensitivity of Cumulative NPV (Revenue) to Parameters",
            "Cumulative NPV ($)",
        )
