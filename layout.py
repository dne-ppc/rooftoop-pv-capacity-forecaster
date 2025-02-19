import os

import streamlit as st
import plotly.express as px
from metalog import metalog
import pandas as pd
import yaml

import plotting
from analysis import Analysis
from info import Info


class Layout:
    @staticmethod
    def controls(tab):
        with tab:
            st.header("Controls & Distributions")

            col1,col2 =st.columns(2)

            with col1:

                st.session_state.years = st.number_input(
                    "Years", min_value=1, max_value=50, value=25, step=1
                )

            with col2:

                st.session_state.iterations = st.number_input(
                    "Iterations", min_value=1, max_value=10, value=10, step=1
                )

            # Scenario Management Container
            with st.container():
                st.header("Manage Forecast Scenarios")
                if "scenarios" not in st.session_state:
                    st.session_state.scenarios = []

            try:
                template_files = [
                    f for f in os.listdir("scenarios") if f.endswith(".yaml")
                ]
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
                        create_standard_scenario(idx, scenario)

    @staticmethod
    def help(tab):
        with tab:
            Info.dispatch("help")

    @staticmethod
    def forecasts(tab):

        with tab:

            st.header("Forecasts")
            with st.expander("Detailed Forecast Calculation Information"):
                Info.dispatch("monte_carlo_simulation")

            forecasts = [
                "Annual Capacity",
                "Cumulative Capacity",
                "Annual Households Installed",
                "Cumulative Households Installed",
                # "Fraction Installed",
                # "Growth Factor",
                "Energy",
                # "Revenue",
                # "Install Cost",
            ]

            

            if st.session_state.scenarios:
                forecast_choice = st.selectbox(
                    "Select forecast to display",
                    options=forecasts,
                    key="select_forecast_metric",
                )

                if forecast_choice:

                    plotting.create_forecast_plot(forecast_choice)

    @staticmethod
    def change_distribution(tab):
        with tab:
            st.header("Change Distributions")
            with st.expander("Detailed Change Distribution Information"):
                Info.dispatch("change_distribution")

            forecasts = [
                "Annual Capacity",
                "Annual Households Installed",
                # "Fraction Installed",
                # "Growth Factor",
                # "Energy",
                # "Revenue",
                # "Install Cost",
            ]

            forecast_choice = st.selectbox("Pick", options=forecasts, key="change_dist")

            if st.session_state.scenarios:
                plotting.plot_all_scenarios_pdf_cdf_histograms(forecast_choice)

    @staticmethod
    def timeseries_metrics(tab):
        with tab:
            st.header("Time Series Metrics")
            with st.expander("Detailed Time Series Metrics Information"):
                Info.dispatch("time_series_metrics")

            metrics = [
                "New Panel Power Rating (W/m²)",
                # "Panel Degradation Factor (%)",
                "Panel Power Gain Factor (%)",
                # "Install Discount Factor (%)",
                # "Median Installed Area (km²)",
                # "Median Installed City Coverage (%)",
                # "Median Incremental Capacity (MW)",
            ]
            metric_choice = st.selectbox(
                "Pick a metric", options=metrics, key="tracking_select"
            )
            if st.session_state.scenarios:
                plotting.plot_tracking_data(metric_choice)

    @staticmethod
    def capacity_sensitivity(tab):
        with tab:
            st.header("Sensitivity Analysis: Total Capacity")
            if st.session_state.scenarios:
                scenario = st.selectbox(
                    "Select the scenario",
                    options=st.session_state.scenarios,
                    key="select_capacity_sensitivity",
                )
                bounds = get_sensitivity_bounds("Capacity")

                sensitivity_results = Analysis.sensitivity_total(
                    scenario, bounds, "Capacity", method="max"
                )
                plotting.create_tornado_figure(
                    sensitivity_results,
                    "Sensitivity of Max Capacity to Parameters",
                    "Capacity (MW)",
                )

    @staticmethod
    def energy_sensitivity(tab):
        with tab:
            st.header("Sensitivity Analysis: Total Energy Production")

            with st.expander("Detailed Energy Sensitivity Analysis Information"):
                Info.dispatch("sensitivity_analysis")

            if st.session_state.scenarios:
                scenario = st.selectbox(
                    "Select the scenario",
                    options=st.session_state.scenarios,
                    key="select_energy_sensitivity",
                )
                bounds = get_sensitivity_bounds("Total Energy")

                sensitivity_results = Analysis.sensitivity_total(
                    scenario, bounds, "Energy"
                )
                sensitivity_results["Low"] /= 1e6
                sensitivity_results["High"] /= 1e6
                sensitivity_results["Baseline"] /= 1e6
                plotting.create_tornado_figure(
                    sensitivity_results,
                    "Sensitivity of Total Energy to Parameters",
                    "Total Energy (GWh)",
                )

    @staticmethod
    def revenue_sensitivity(tab):
        with tab:
            st.header("Sensitivity Analysis: Cumulative NPV for Revenue")

            with st.expander("Detailed NPV Sensitivity Analysis Information"):
                Info.dispatch("revenue_sensitivity")

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
                bounds = get_sensitivity_bounds("Revenue NPV")
                sensitivity_results = Analysis.sensitivity_cumulative_npv_revenue(
                    scenario, bounds, discount_rate
                )
                plotting.create_tornado_figure(
                    sensitivity_results,
                    "Sensitivity of Cumulative NPV (Revenue) to Parameters",
                    "Cumulative NPV ($)",
                )

    @staticmethod
    def npv(tab):
        with tab:
            st.header("NPV Forecast")
            with st.expander("Detailed NPV Information"):
                Info.dispatch("npv")

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

            is_cumulative = st.checkbox("Cumulative NPV", value=True)
            plotting.create_npv_plot(
                forecast_choice, discount_rate, cumulative=is_cumulative
            )

    @staticmethod
    def dispatch(info_type: str, tab):

        func = getattr(Layout, info_type, None)

        if func is not None:
            func(tab)
        else:

            st.error("Invalid tab name")


def create_dist(
    scenario, label, min_value, max_value, p10, p50, p90, step=0.01, probs=None
):

    if probs is None:
        probs = [0.1, 0.5, 0.9]
    

    if scenario not in st.session_state:
        st.session_state[scenario] = {}

    if label not in st.session_state:
        st.session_state[scenario][label] = {}

    st.subheader(label)
    left, right = st.columns(2)
    with left:
        st.container(height=70, border=False)

        p10 = st.slider(
            "P10",
            min_value=min_value,
            max_value=max_value,
            value=p10,
            step=step,
            key=f"{scenario}_{label}_p10",
        )
        p50 = st.slider(
            "P50",
            min_value=min_value,
            max_value=max_value,
            value=p50,
            step=step,
            key=f"{scenario}_{label}_p50",
        )
        p90 = st.slider(
            "P90",
            min_value=min_value,
            max_value=max_value,
            value=p90,
            step=step,
            key=f"{scenario}_{label}_p90",
        )

    with right:
        try:
            dist = metalog.fit(
                x=[p10, p50, p90],
                boundedness="b",
                bounds=[min_value - step, max_value + step],
                term_limit=3,
                probs=probs,
            )
            st.plotly_chart(
                plotting.create_dist_plot(dist),
                use_container_width=True,
                key=f"{scenario}_{label}",
            )
        except Exception as e:
            st.error(f"Error fitting installation cost distribution: {e}")

    st.session_state[scenario][label] = dist

    st.divider()


def create_standard_scenario(idx, scenario_name):

    years = st.session_state.years
    iterations = st.session_state.iterations
    if 'run' not in st.session_state[scenario_name]:
        st.session_state[scenario_name]['run'] = False

    with st.container(border=True):

        run = st.button("Run Model", key=f"{scenario_name}_run_btn")

        # Create distribution sliders using values from config
        config = st.session_state["config"][scenario_name]
        create_dist(scenario_name, **config["annual_growth_rate"])

        st.subheader("Solar Panel Assumptions")
        panel_power = create_input(scenario_name, "panel_power")
        panel_gain = create_input(scenario_name, "panel_gain")

        #TODO Update simulation code to use these distributions
        # create_dist(scenario_name, **config["energy_price"])
        # create_dist(scenario_name, **config["installation_price"])

        # create_input(scenario_name, "install_discount")

        st.session_state[scenario_name]["color"] = px.colors.qualitative.Plotly[idx]

        if run:
            Analysis.monte_carlo_forecast(
                scenario_name, years, iterations, panel_power, panel_gain
            )
            st.session_state[scenario_name]['run'] = True

        if st.button("Delete scenario", key=f"{scenario_name}_delete_btn"):
            st.session_state.scenarios.pop(idx)
            del st.session_state[scenario_name]
            st.rerun()


def create_input(scenario_name: str, input_name: str):

    config = st.session_state["config"][scenario_name][input_name]
    key = f"{scenario_name}_{input_name}"
    input_type = config.pop("input_type")
    input_cls = getattr(st, input_type)
    elem = input_cls(key=key, **config)
    config["input_type"] = input_type
    st.session_state[scenario_name][input_name] = elem
    return elem


def get_sensitivity_bounds(key):
    """
    Displays two data editors:
      1. Slider Parameters Bounds: Editable table with each slider-controlled parameter and its lower/upper bounds.
      2. Distribution Parameters: Editable table with the P10, P50, and P90 values for annual growth rate, capacity factor, and energy price.

    Returns:
      slider_bounds_df, distribution_bounds_df : two DataFrames containing the edited values.
    """
    # Create a DataFrame for slider parameters bounds.

    records = [
        {
            "Impact": "positive",
            "Parameter": "pv_area_percent",
            "Lower Bound": 0.001,
            "Upper Bound": 0.20,
        },
        {
            "Impact": "positive",
            "Parameter": "panel_gain",
            "Lower Bound": 0.0,
            "Upper Bound": 0.1,
        },
        {
            "Impact": "negative",
            "Parameter": "panel_degradation_factor",
            "Lower Bound": 0.0,
            "Upper Bound": 0.1,
        },
        {
            "Impact": "positive",
            "Parameter": "Capacity Factor",
            "Lower Bound": 0.10,
            "Upper Bound": 0.30,
        },
        {
            "Impact": "positive",
            "Parameter": "Energy Price ($/kWh)",
            "Lower Bound": 0.01,
            "Upper Bound": 0.30,
        },
        {
            "Impact": "positive",
            "Parameter": "Annual Growth Factor",
            "Lower Bound": 0.01,
            "Upper Bound": 0.50,
        },
    ]

    bounds_df = pd.DataFrame(records)

    # Swap bounds for rows where the Impact is negative.
    for index, row in bounds_df.iterrows():
        if row["Impact"] == "negative":
            # Swap the values: Lower Bound becomes Upper Bound and vice versa.
            lower = row["Lower Bound"]
            upper = row["Upper Bound"]
            bounds_df.at[index, "Lower Bound"] = upper
            bounds_df.at[index, "Upper Bound"] = lower

    bounds_df = st.data_editor(
        bounds_df,
        key=f"{key}_bounds",
        hide_index=True,
        column_config={
            "Parameter": st.column_config.TextColumn("Parameter", disabled=True),
        },
        use_container_width=True,
    )

    return bounds_df
