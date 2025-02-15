import os

import streamlit as st
import plotly.express as px
from metalog import metalog
import pandas as pd
import yaml

import plotting
import analysis


def create_dist(
    scenario, label, min_value, max_value, p10, p50, p90, step=0.01, probs=None
):

    if probs is None:
        probs = [0.1, 0.5, 0.9]
    st.divider()

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


def create_standard_scenario(idx, scenario_name):

    years = st.session_state.years

    with st.container(border=True):
        st.subheader("City Capacity Bounds")
        col1, col2, col3 = st.columns(3)
        with col1:
            city_area = create_input(scenario_name, "city_area")
            pv_area_percent = create_input(scenario_name, "pv_area_percent")

        with col2:
            panel_power = create_input(scenario_name, "panel_power")
            panel_gain = create_input(scenario_name, "panel_gain")
            create_input(scenario_name, "panel_degradation_factor")

        with col3:
            create_input(scenario_name, "initial_city_capacity")
            max_gain = (1 + panel_gain) ** years
            max_capacity_val = city_area * pv_area_percent * max_gain * panel_power
            st.metric(
                label="City Maximum PV Capacity", value=f"{int(max_capacity_val)} MW"
            )

        # Create distribution sliders using values from config
        config = st.session_state["config"][scenario_name]
        create_dist(scenario_name, **config["annual_growth_rate"])
        create_dist(scenario_name, **config["capacity_factor"])
        create_dist(scenario_name, **config["energy_price"])
        create_dist(scenario_name, **config["installation_price"])

        create_input(scenario_name, "install_discount")

        st.session_state[scenario_name]["color"] = px.colors.qualitative.Plotly[idx]

        analysis.monte_carlo_forecast(scenario_name)

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


def show_help_modal(
        tab
):
    """
    Displays a modal with detailed instructions on how to use the Calgary Rooftop Solar Forecast application.

    The help content covers:
      - How to configure distributions (using the p10, p50, and p90 sliders) and what these parameters imply.
      - How key city and panel parameters (like city area and maximum PV area fraction) affect the maximum
        capacity, energy production, installation costs, and revenues.
      - What you can do and learn from each tab in the application.
    """
    # When the "Help" button is clicked, open a modal dialog.
    with tab:
        st.markdown(
        r"""
        ##  Help & Usage Guide

        **Overview:**  
        This application uses Monte Carlo simulations to forecast rooftop solar capacity, energy production, revenues, and installation costs in Calgary. The model incorporates a logistic (S-curve) growth framework with uncertainty captured through user-configured probability distributions.

        ## Controls & Distributions (Tab 1)
        - **Simulation Settings:**  
            Use the sliders to set the forecasting duration (in years) and the number of iterations (simulations per year).
        - **Scenario Creation:**  
            Enter a unique scenario name and click **Create scenario**. Each scenario can have different parameters.
        - **City and Panel Configuration:**  
            - **City Area (km²):** Defines the total area of the city. A larger city area increases the maximum potential area available for panel installations.
            - **Maximum PV Area Fraction:** Sets the fraction of the city area that can be covered with panels. A higher value increases the potential installed capacity.
            - **Initial Panel Power, Efficiency Gain, and Degradation:**  
            - **Initial Panel Power (W/m²):** The baseline performance of panels.
            - **Panel Efficiency Gain (%/year):** The rate at which new panels become more efficient, improving capacity.
            - **Panel Degradation (%/year):** The annual loss in performance of installed panels.
        - **Configuring Distributions:**  
            For each key parameter (e.g., **Annual Growth Factor**, **Capacity Factor**, **Energy Price (\$/kWh)**, and **Installation Price (\$/kWWh)**):
            - You’ll see three sliders for **P10**, **P50**, and **P90**. These represent the 10th percentile (optimistic), median, and 90th percentile (pessimistic) estimates.
            - Adjusting these sliders shapes the underlying probability distributions (using a metalog fit) which in turn affects the forecasts.
            - **Implications:** A change in these distributions influences the simulation outcomes. For example, a higher median energy price increases forecasted revenues, while a wider spread between P10 and P90 indicates higher uncertainty.
        
        ## Forecasts (Tab 2)
        - View the simulated forecasts for **Capacity (MW)**, **Energy (kWh/year)**, **PV Revenue**, and **Installation Cost**.
        - The plots display uncertainty bands (P10, P50, and P90) so you can visualize the variability in the predictions.
        
        ## NPV (Tab 3)
        - Calculate the Net Present Value (NPV) of future cash flows.
        - Choose between revenue and installation cost forecasts, adjust the discount rate, and toggle cumulative versus annual values.
        - This helps in assessing the investment’s financial viability.
        
        ## Change Distribution (Tab 4)
        - Examine histograms (PDF and CDF) for year-to-year percent changes in the forecasts.
        - This view is useful to understand the variability and likelihood of different changes from one year to the next.
        

        ## Timeseries Metrics (Tab 5)
        - Track detailed metrics such as:
            - **New Panel Power Rating:** The improved performance for new installations.
            - **Panel Degradation Factor:** How much the performance of existing panels decreases each year.
            - **Panel Power Gain Factor:** The efficiency gains for new panels.
            - **Installation Discount Factor:** How discounting affects cost calculations.
            - **Median Installed Area and Incremental Capacity:** How the cumulative installations evolve over time.
        - These metrics offer granular insights into how each component drives overall system performance.

        ## Energy Sensitivity (Tab 6)
           Analyzes the sensitivity of total energy production to key model parameters. For each parameter, the simulation is re-run with the parameter set to its lower and upper bounds. The impact on total energy production is visualized using a tornado diagram, highlighting which parameters most influence energy output.

           
        ## Revenue Sensitivity (Tab 7)
           Examines the sensitivity of the cumulative NPV for revenue to changes in key parameters. The analysis involves varying each parameter to its extreme bounds while applying an adjustable discount rate. The resulting changes in NPV are displayed in a tornado chart, making it clear which parameters have the greatest effect on the financial forecast.

           
        **Key Takeaways:**
        - **City Area and Coverage:**  
            A larger city area or higher PV area fraction increases the maximum possible installed capacity, boosting energy production and potential revenues.
        - **Distributions:**  
            Adjusting the distributions (via the three percentile sliders) directly affects the uncertainty in the forecasts. This enables you to simulate different market conditions and risk profiles.
        - **Tab Navigation:**  
            Each tab focuses on a different aspect of the forecast—from overall trends and financial metrics to detailed distributional changes and time-series tracking. Use these insights to understand the sensitivity of the system to various factors.

        Use this guide as a reference while exploring different scenarios and adjusting parameters to see how they influence the forecast outcomes.
        """
    )


def monte_carlo_simulation_explanation():
    """
    Displays a detailed explanation of the Monte Carlo simulation logic, including all formulas used.
    This explanation covers:
      - The simulation inputs and assumptions.
      - How panel performance improves over time.
      - The degradation of existing capacity.
      - The logistic growth model and its capacity limit factor.
      - Energy production, revenue, and installation cost calculations.
      - How probability distributions (configured via P10, P50, P90) are used in the simulation.
    """
    explanation_md = r"""
# Monte Carlo Simulation Detailed Explanation

**Overview:**  
This simulation models the growth of rooftop solar installations using a Monte Carlo approach. By sampling from probability distributions, the simulation captures uncertainties in growth, performance, prices, and costs over a user-specified number of years.

---

## Simulation Inputs

- **Forecast Duration (Years):** Total number of simulation years.
- **Iterations:** Number of simulation runs per year.
- **City Area (A):** Total area of the city (km²).
- **Maximum PV Area Fraction (f):** Fraction of the city area available for solar panels.
- **Initial City Capacity (C₀):** Starting installed capacity in MW.
- **Base Panel Power (P₀):** Initial performance of solar panels (W/m²).

---

## Panel Power Improvement

Newer panel models are usually more efficient \([e.g.](https://www.oxfordpv.com/news/oxford-pv-debuts-residential-solar-module-record-setting-269-efficiency)\). The panel power for year *t* is calculated as:

$$
P_t = P_0 \times (1 + \text{panel\_gain})^{t}
$$

---

## Discount Factor

For cost calculations, a discount factor is applied to future installation costs:

$$
DF_t = \frac{1}{(1 + r)^t}
$$

where \( r \) is the discount rate and \( t \) is the year number.

---

## Yearly Simulation Process

For each simulation year, the following steps are performed:

1. **Degradation of Existing Capacity:**  
   Installed capacity deteriorates over time. The effective capacity from the previous year is:

   $$
   C_{prev, effective} = C_{prev} \times (1 - \text{panel\_degradation})
   $$

2. **Potential Maximum Capacity:**  
   The maximum capacity possible is limited by the available area for panel installations.
   - **Total Possible Area:**

     $$
     \text{Total Area}_{possible} = A \times f
     $$

   - **Maximum Achievable Capacity:**

     $$
     C_{max} = C_{prev, effective} + \left(\text{Total Area}_{possible} - \text{Area Installed}\right) \times P_t
     $$

3. **Logistic Growth Model:**  
   To model saturation effects, a logistic growth approach is applied:
   - **Capacity Limit Factor:**

     $$
     \text{Limit Factor} = 1 - \frac{C_{prev, effective}}{C_{max}}
     $$

   - **Growth Rate Sampling:**  
     A growth rate \( g \) is randomly sampled from a probability distribution (capped at a maximum value \( g_{max} \)).
   - **New Effective Capacity:**

     $$
     C_{new} = C_{prev, effective} \times \left(1 + \min(g, g_{max}) \times \text{Limit Factor}\right)
     $$

   - **Incremental Capacity and Installed Area:**  
     The additional capacity added is:

     $$
     \Delta C = C_{new} - C_{prev, effective}
     $$

     The area required for this new capacity is:

     $$
     \Delta A = \frac{\Delta C}{P_t}
     $$

     The cumulative installed area is updated as:

     $$
     \text{Area Installed}_{new} = \text{Area Installed}_{prev} + \Delta A
     $$

4. **Energy Production:**  
   Energy produced in the year is computed using the new capacity:

   $$
   E_t = C_{new} \times CF_t \times 1000 \times 8760
   $$

   where:
   - \( CF_t \) is the capacity factor for year *t* (sampled from a probability distribution).
   - The factor \( 1000 \times 8760 \) converts MW to kWh per year.

5. **Revenue Calculation:**  
   The revenue for the year is determined by:

   $$
   R_t = E_t \times \text{Price}_t
   $$

   where \( \text{Price}_t \) is the energy price sampled for that year.

6. **Installation Cost Calculation:**  
   The cost for new installations, adjusted by the discount factor, is:

   $$
   \text{Cost}_t = \Delta C \times 10^6 \times \text{InstallPrice}_t \times DF_t
   $$

   Here, \( 10^6 \) is used to convert MW to the appropriate unit scale for cost calculation.

---

## Sampling Distributions

For each year and simulation iteration, the following key parameters are sampled from metalog distributions (configured via P10, P50, and P90 sliders):

- **Annual Growth Factor (g)**
- **Capacity Factor (CF)**
- **Energy Price**
- **Installation Price**

These distributions introduce uncertainty into the simulation, enabling a range of outcomes based on different possible future scenarios.

---

## Summary

The Monte Carlo simulation iteratively computes:
- **Capacity:** Updated using degradation and logistic growth.
- **Energy Production:** Based on effective capacity and capacity factor.
- **Revenue:** Derived from the energy produced and energy price.
- **Installation Costs:** Calculated using the incremental capacity, installation price, and discounted to present value.

By adjusting both the deterministic inputs (like city area and base panel power) and the stochastic inputs (via probability distributions), the simulation provides a robust framework for exploring various scenarios in rooftop solar installation growth.

---

Use this guide to understand how each element of the simulation contributes to forecasting solar capacity, energy, revenue, and costs.
    """
    return st.markdown(explanation_md)


def npv_explanation():
    """
    Returns a markdown explanation for the NPV tab.

    This explanation covers:
      - The NPV formula and its components.
      - The difference between annual and cumulative NPV.
      - How the discount rate is applied.
      - The purpose of the NPV analysis in evaluating the financial viability.
    """
    explanation_md = r"""
# Net Present Value (NPV) Explanation

**Overview:**  
The NPV tab computes the Net Present Value of forecasted cash flows over the simulation period, using either revenue or installation cost forecasts. This analysis helps assess the investment's financial viability by accounting for the time value of money.

---

## NPV Formula

NPV is calculated as:

$$
\text{NPV} = \sum_{t=1}^{T} \frac{C_t}{(1 + r)^t}
$$

Where:  
- \( C_t \) is the cash flow in year \( t \).  
  - For revenue forecasts, \( C_t \) represents the revenue generated.  
  - For cost forecasts, \( C_t \) represents the installation cost.  
- \( r \) is the discount rate, representing the time value of money.
- \( T \) is the total number of forecast years.

The discount factor for each year is:

$$
DF_t = \frac{1}{(1 + r)^t}
$$

Thus, the discounted cash flow for a given year is \( C_t \times DF_t \).

---

## Annual vs. Cumulative NPV

- **Annual NPV:**  
  Each year's discounted cash flow is shown individually.

- **Cumulative NPV:**  
  The discounted cash flows are summed cumulatively, providing an aggregate value over the simulation period.

---

## How to Use This Tab

- **Forecast Selection:** Choose whether to analyze revenue or installation cost forecasts.
- **Adjust Discount Rate:** Use the slider to set the discount rate \( r \) reflecting the opportunity cost of capital.
- **Toggle Cumulative Option:** Switch between viewing annual and cumulative NPV values.

This analysis assists in comparing different scenarios and understanding how future cash flows add up when discounted back to their present value.
    """
    return st.markdown(explanation_md)


def change_distribution_explanation():
    """
    Returns a markdown explanation for the Change Distribution tab.

    This explanation highlights that the purpose of this tab is to serve as a sanity check,
    ensuring that the year-over-year changes in forecasted values follow a reasonable distribution.
    """
    explanation_md = r"""
# Change Distribution Explanation

**Overview:**  
The Change Distribution tab is designed as a sanity check to verify that the year-over-year changes in the forecasted values are reasonable. This is achieved by analyzing the percentage changes in key forecasts such as capacity, energy, revenue, and costs.

---

## What is Being Analyzed?

- **Year-over-Year Percent Changes:**  
  For each forecast, the percent change between consecutive years is computed as:

  $$
  \text{Percent Change}_t = \frac{Value_t}{Value_{t-1}} - 1
  $$

- **Distribution Visualization:**  
  The computed changes are visualized using:
  - **PDF (Probability Density Function) Histogram:** Displays the density of the changes.
  - **CDF (Cumulative Distribution Function) Histogram:** Shows the cumulative probability distribution.

---

## Purpose of This Analysis

- **Sanity Check:**  
  The distribution of changes helps confirm that the simulation is not producing unrealistic jumps or drops from one year to the next.

- **Risk and Variability Assessment:**  
  - A **narrow distribution** suggests low variability and less uncertainty, or lower assumed rates of change.
  - A **wider distribution** indicates higher uncertainty or higher assumed rates of change. Low assumed rates of change and high observed changed would indicate model error.

By ensuring that the forecasted changes adhere to a reasonable distribution, you can validate that the underlying model and its assumptions are consistent and reliable.
    """
    return st.markdown(explanation_md)


def time_series_metrics_explanation():
    """
    Returns a markdown explanation for the Time Series Metrics tab.

    This explanation covers:
      - The key metrics tracked over time.
      - How these metrics help assess the growth and degradation factors.
      - The purpose of monitoring these metrics to validate the model's assumptions.
    """
    explanation_md = r"""
# Time Series Metrics Explanation

**Overview:**  
The Time Series Metrics tab tracks several key metrics over the simulation period. This provides insights into the dynamics of the model, particularly verifying whether the growth and degradation factors applied are reasonable.

---

## Key Metrics Tracked

- **New Panel Power Rating (W/m²):**  
  Shows the improvement in performance for newly installed panels over time.

- **Panel Degradation Factor (%):**  
  Indicates the annual loss in efficiency of already installed panels.

- **Panel Power Gain Factor (%):**  
  Represents the yearly efficiency improvement for new installations.

- **Installation Discount Factor (%):**  
  Reflects the discounting applied to future installation costs.

- **Median Installed Area (km²):**  
  Tracks the cumulative area covered by panels each year.

- **Median Incremental Capacity (MW):**  
  Monitors the additional effective capacity added annually.

---

## Purpose of Tracking These Metrics

- **Validation of Model Assumptions:**  
  By examining these time series, you can verify that the growth improvements and degradation effects are within expected ranges.  
  For example:
  - **Growth Factors:** Ensure that the improvements in panel power are consistent with technological advancements.
  - **Degradation Factors:** Check that the decline in performance of existing panels aligns with real-world expectations.

- **Operational Insights:**  
  These metrics help you understand how the various factors interact over time and affect the overall forecast. This can inform adjustments to the model if discrepancies are observed.

Overall, this tab acts as a diagnostic tool to confirm the internal consistency of the simulation and to ensure that the assumptions driving the forecasts are producing realistic outcomes.
    """
    return st.markdown(explanation_md)


def show_energy_sensitivity_info():
    """
    Displays an explanation for the Total Energy Production Sensitivity Analysis tab.

    This explanation covers:
      - How a baseline simulation is run using the median (P50) energy forecast.
      - How each slider-controlled parameter is varied (set to its lower and upper bounds) one at a time.
      - How re-running the simulation for each variation shows the impact on total energy production.
      - How the results are visualized in a tornado diagram.
    """

    st.markdown(
        r"""
        ## Total Energy Production Sensitivity Analysis

        In this analysis, the impact of key model parameters on total energy production is assessed as follows:

        - **Baseline Simulation:**  
          The model first runs with the current (median) settings to compute a baseline total energy production.

        - **Parameter Variation:**  
          Each slider-controlled parameter (such as *PV Area Fraction*, *Panel Efficiency Gain*, 
          *Panel Degradation*, *Capacity Factor*, and *Annual Growth Factor*) is individually set to its 
          lower and upper bounds.

        - **Re-simulation:**  
          For each bound, the simulation re-runs and computes the new total energy production.

        - **Visualization:**  
          A tornado diagram displays the baseline energy along with the values obtained at the low 
          and high parameter bounds, clearly identifying which parameters most affect energy output.
        """
    )


def show_revenue_sensitivity_info():
    """
    Displays an explanation for the Cumulative NPV for Revenue Sensitivity Analysis tab.

    This explanation covers:
      - How a baseline cumulative NPV is computed using the median revenue forecast.
      - How each parameter is varied (set to its lower and upper bounds) while applying an adjustable discount rate.
      - How re-running the simulation for each variation shows the impact on cumulative NPV.
      - How the results are visualized in a tornado chart.
    """

    st.markdown(
        r"""
        ## Cumulative NPV for Revenue Sensitivity Analysis

        This analysis explores how variations in key parameters influence the cumulative Net Present Value (NPV) of revenue:

        - **Baseline NPV:**  
          A baseline cumulative NPV is calculated using the median revenue forecast.

        - **Parameter Variation:**  
          Each slider-controlled parameter is varied one at a time to its specified lower and upper bounds.
          A discount rate is applied (which the user can adjust) to account for the time value of money.

        - **Re-simulation:**  
          The simulation re-runs for each bound and the new NPV is computed.

        - **Visualization:**  
          A tornado diagram then displays the baseline NPV along with the NPVs corresponding to the lower 
          and upper bounds for each parameter, highlighting their impact on the financial forecast.

        **Methodology Overview:**
        - A baseline simulation is executed with the default parameter values.
        - Each parameter is individually varied while keeping others constant.
        - The differences between the baseline and the varied simulations are computed.
        - The results are visualized using tornado diagrams, making it clear which parameters drive the most significant changes.
        """
    )


def create_controls_tab(tab):
    with tab:
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
                    create_standard_scenario(idx, scenario)


def create_forecasts_tab(tab):

    with tab:
        st.header("Forecasts")
        with st.expander("Detailed Forecast Calculation Information"):
            monte_carlo_simulation_explanation()

        forecasts = ["Capacity", "Energy", "Revenue", "Install Cost"]

        if st.session_state.scenarios:
            forecast_choice = st.selectbox(
                "Seelct forecast to display",
                options=forecasts,
                key="select_forecast_metric",
            )

            if forecast_choice:
                plotting.create_forecast_plot(forecast_choice)


def create_change_dist_tab(tab):
    with tab:
        st.header("Change Distributions")
        with st.expander("Detailed Change Distribution Information"):
            change_distribution_explanation()

        forecasts = ["Capacity", "Energy", "Revenue", "Install Cost"]
        forecast_choice = st.selectbox("Pick", options=forecasts, key="change_dist")

        if st.session_state.scenarios:
            plotting.plot_all_scenarios_pdf_cdf_histograms(forecast_choice)


def create_ts_metrics_tab(tab):

    with tab:
        st.header("Time Series Metrics")
        with st.expander("Detailed Time Series Metrics Information"):
            time_series_metrics_explanation()

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


def create_cap_sensitivity_tab(tab):
    with tab:
        st.header("Sensitivity Analysis: Total Capacity")
        if st.session_state.scenarios:
            scenario = st.selectbox(
                "Select the scenario",
                options=st.session_state.scenarios,
                key="select_capacity_sensitivity",
            )
            bounds = get_sensitivity_bounds("Capacity")

            sensitivity_results = analysis.sensitivity_total(
                scenario, bounds, "Capacity", method="max"
            )
            plotting.create_tornado_figure(
                sensitivity_results,
                "Sensitivity of Max Capacity to Parameters",
                "Capacity (MW)",
            )


def create_energy_sensitivity_tab(tab):

    with tab:
        st.header("Sensitivity Analysis: Total Energy Production")

        with st.expander("Detailed Energy SensitivityAnalysis Information"):
            show_energy_sensitivity_info()

        if st.session_state.scenarios:
            scenario = st.selectbox(
                "Select the scenario",
                options=st.session_state.scenarios,
                key="select_energy_sensitivity",
            )
            bounds = get_sensitivity_bounds("Total Energy")

            sensitivity_results = analysis.sensitivity_total(scenario, bounds, "Energy")
            sensitivity_results["Low"] /= 1e6
            sensitivity_results["High"] /= 1e6
            sensitivity_results["Baseline"] /= 1e6
            plotting.create_tornado_figure(
                sensitivity_results,
                "Sensitivity of Total Energy to Parameters",
                "Total Energy (GWh)",
            )


def create_rev_sensitivity_tab(tab):

    with tab:
        st.header("Sensitivity Analysis: Cumulative NPV for Revenue")

        with st.expander("Detailed NPV Sensitivity Analysis Information"):
            show_revenue_sensitivity_info()

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
            sensitivity_results = analysis.sensitivity_cumulative_npv_revenue(
                scenario, bounds, discount_rate
            )
            plotting.create_tornado_figure(
                sensitivity_results,
                "Sensitivity of Cumulative NPV (Revenue) to Parameters",
                "Cumulative NPV ($)",
            )


def create_npv_tab(tab):

    with tab:
        st.header("NPV Forecast")
        with st.expander("Detailed NPV Information"):
            npv_explanation()

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
