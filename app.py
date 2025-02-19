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

from layout import Layout

# -----------------------------------------------------------------------------
# Page Configuration and Session State Initialization
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Rooftop Solar Forecast", layout="wide")

# Initialize session state variables if not already set.
if "years" not in st.session_state:
    st.session_state.years = 25

if "iterations" not in st.session_state:
    st.session_state.iterations = 1000

if "model_run" not in st.session_state:
    st.session_state.model_run = False


if "config" not in st.session_state:
    st.session_state["config"] = {}

# App Title
st.title("Rooftop Solar Growth & Energy Forecast")

# -----------------------------------------------------------------------------
# Create Tabs
# -----------------------------------------------------------------------------


sections = [
    "Help",
    "Controls",
    "Forecasts",
    "Change Distribution",
    "Timeseries Metrics",
    # "Capacity Sensitivity",
    # "Energy Sensitivity",
    # "Revenue Sensitivity",
    # "NPV"
]



tabs = st.tabs(sections)

for tab_name, tab in zip(sections, tabs):
    Layout.dispatch(tab_name.lower().replace(" ", "_"), tab)