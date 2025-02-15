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

import streamlit as st
import pandas as pd

import plotting
import layout
import analysis

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


sections = pd.DataFrame(
    [
        {
            "title": "Info",
            "function": layout.show_help_modal,
        },
        {
            "title": "Controls",
            "function": layout.create_controls_tab,
        },
        {
            "title": "Forecasts",
            "function": layout.create_forecasts_tab,
        },
        {
            "title": "Change Distribution",
            "function": layout.create_change_dist_tab,
        },
        {
            "title": 'Timeseries Metrics"',
            "function": layout.create_ts_metrics_tab,
        },
        {
            "title": "Capacity Sensitivity",
            "function": layout.create_cap_sensitivity_tab,
        },
        {
            "title": 'Energy Sensitivity"',
            "function": layout.create_energy_sensitivity_tab,
        },
        {
            "title": 'Revenue Sensitivity"',
            "function": layout.create_energy_sensitivity_tab,
        },
        {"title": "NPV", "function": layout.create_npv_tab},
    ]
)

tabs = st.tabs(sections['title'].to_list())

for tab, func in zip(tabs, sections.function):
    func(tab)
