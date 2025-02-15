import streamlit as st
import numpy as np
import pandas as pd
from metalog import metalog


def process_panel_capacities(panel_data, forecast_years):
    """
    Process panel capacities (step‐function) from a list of dicts.
    """
    try:
        sorted_data = sorted(panel_data, key=lambda r: float(r.get("Year", 0)))
    except Exception:
        sorted_data = []
    panel_capacities = []
    current_capacity = 200
    current_index = 0
    n = len(sorted_data)
    for y in range(1, forecast_years + 1):
        while (
            current_index < n and float(sorted_data[current_index].get("Year", 0)) <= y
        ):
            current_capacity = float(
                sorted_data[current_index].get("Panel Capacity (W/m²)", 200)
            )
            current_index += 1
        panel_capacities.append(current_capacity)
    return panel_capacities


def monte_carlo_forecast(scenario_name, max_growth=1, sensitivity=None):
    """
    Monte Carlo Forecast that computes capacity, energy, revenue, and discounted installation costs.

    The model now:
      1. Improves panel power for new installations on the remaining (yet to be covered) city area.
      2. Degrades the effective capacity of previously installed panels by a specified percentage each year.
      3. Tracks time series for:
           - the new panel power rating,
           - the annual degradation factor,
           - the yearly gain factor,
           - the yearly discount factor,
           - the median installed panel area, and
           - the median incremental capacity added.
      4. Persists these tracking time series along with the updated panel power and city coverage.

    """
    # Retrieve simulation settings.
    years = st.session_state.years
    iterations = st.session_state.iterations
    city_area = st.session_state[scenario_name]["city_area"]
    pv_area_percent = st.session_state[scenario_name][
        "pv_area_percent"
    ]  # maximum fraction of city area allowed for panels
    initial_city_capacity = st.session_state[scenario_name]["initial_city_capacity"]
    base_panel_power = st.session_state[scenario_name][
        "panel_power"
    ]  # baseline panel power (MW per area unit)

    discount_rate = st.session_state[scenario_name]["install_discount"]
    panel_gain = st.session_state[scenario_name]["panel_gain"]
    # Retrieve the annual degradation factor (e.g., 0.02 for 2% per year)
    panel_degradation = st.session_state[scenario_name]["panel_degradation_factor"]

    # Pre-calculate panel power improvements for new installations.
    gain_factors = np.power((1 + panel_gain), np.arange(1, years + 1))
    panel_power_array = base_panel_power * gain_factors  # (MW per area unit)

    # Conversion constant: kWh per MW-year.
    conversion_constant = 1000 * 8760

    # Retrieve distributions.
    cf_dist = st.session_state[scenario_name]["Capacity Factor"]
    price_dist = st.session_state[scenario_name]["Energy Price ($/kWh)"]
    install_dist = st.session_state[scenario_name]["Installation Price($/kWWh)"]
    growth_dist = st.session_state[scenario_name]["Annual Growth Factor"]

    if sensitivity == "Annual Growth Factor":
        growth_rates = np.full([iterations, years], growth_dist)
    else:
        growth_rates = metalog.r(m=growth_dist, n=iterations * years).reshape(
            iterations, years
        )

    if sensitivity == "Capacity Factor":
        cf_samples = np.full([iterations, years], cf_dist)
    else:
        cf_samples = metalog.r(m=cf_dist, n=iterations * years).reshape(
            iterations, years
        )

    if sensitivity == "Energy Price ($/kWh)":
        price_samples = np.full([iterations, years], price_dist)
    else:
        price_samples = metalog.r(m=price_dist, n=iterations * years).reshape(
            iterations, years
        )

    install_samples = metalog.r(m=install_dist, n=iterations * years).reshape(
        iterations, years
    )

    # Pre-calculate discount factors for each year.
    discount_factors = 1 / np.power((1 + discount_rate), np.arange(1, years + 1))

    # Pre-allocate arrays for capacity and other outputs.
    capacities = np.empty((iterations, years + 1))
    capacities[:, 0] = initial_city_capacity

    energy = np.empty((iterations, years))
    revenue = np.empty((iterations, years))
    costs = np.empty((iterations, years))

    # --- Track time-series variables with descriptive names ---
    new_panel_power_per_year = []
    annual_degradation_factor = []
    median_installed_area_per_year = []
    median_incremental_capacity_per_year = []

    # --- Track installed area over time ---
    total_possible_area = city_area * pv_area_percent
    # Assume initial panels were installed with the base panel power.
    area_installed = np.full(iterations, initial_city_capacity / base_panel_power)

    for year in range(years):
        # Record tracking variables for this year.
        current_panel_power = panel_power_array[
            year
        ]  # improved panel power for new installations this year
        new_panel_power_per_year.append(current_panel_power)

        degradation_factor = 1 - panel_degradation
        annual_degradation_factor.append(degradation_factor)

        # --- Apply degradation to previously installed capacity ---
        effective_prev = capacities[:, year] * degradation_factor

        # --- Determine potential maximum effective capacity ---
        # New installations use the current panel power.
        potential_max_capacity = (
            effective_prev
            + (total_possible_area - area_installed) * current_panel_power
        )

        # --- Logistic growth model ---
        capacity_limit_factor = 1 - effective_prev / potential_max_capacity
        g_rate = growth_rates[:, year]
        growth_factor = 1 + np.minimum(g_rate, max_growth) * capacity_limit_factor

        new_effective_capacity = effective_prev * growth_factor

        # --- Calculate new installations and update installed area ---
        incremental_capacity = new_effective_capacity - effective_prev
        incremental_area = (
            incremental_capacity / current_panel_power
        )  # area needed for new panels
        area_installed += incremental_area  # update cumulative installed area

        # Record median incremental capacity and installed area across iterations.
        median_incremental_capacity_per_year.append(np.median(incremental_capacity))
        median_installed_area_per_year.append(np.median(area_installed))

        # Store the new effective capacity.
        capacities[:, year + 1] = new_effective_capacity

        # --- Energy, Revenue, and Cost Calculations ---
        energy[:, year] = (
            new_effective_capacity * conversion_constant * cf_samples[:, year]
        )
        revenue[:, year] = energy[:, year] * price_samples[:, year]
        costs[:, year] = (
            incremental_capacity
            * 1e6
            * install_samples[:, year]
            * discount_factors[year]
        )

    # Build DataFrames for outputs.
    cap_df = pd.DataFrame(capacities.T)
    cap_df["Year"] = np.arange(0, years + 1)
    energy_df = pd.DataFrame(energy.T)
    energy_df["Year"] = np.arange(1, years + 1)
    revenue_df = pd.DataFrame(revenue.T)
    revenue_df["Year"] = np.arange(1, years + 1)
    cost_df = pd.DataFrame(costs.T)
    cost_df["Year"] = np.arange(1, years + 1)

    # Persist DataFrames.
    st.session_state[scenario_name]["Capacity DF"] = cap_df
    st.session_state[scenario_name]["Energy DF"] = energy_df
    st.session_state[scenario_name]["Revenue DF"] = revenue_df
    st.session_state[scenario_name]["Install Cost DF"] = cost_df

    # --- Persist Tracking Time Series with Descriptive Names ---
    tracking_data = {
        "New Panel Power Rating (W/m²)": new_panel_power_per_year,  # New panel power rating for installations each year (MW per area unit)
        "Panel Degradation Factor (%)": np.cumprod(
            annual_degradation_factor
        ),  # Degradation factor applied to existing capacity each year (1 - degradation rate)
        "Panel Power Gain Factor (%)": gain_factors,  # Yearly gain factor for new panel power improvements
        "Install Discount Factor (%)": discount_factors,  # Discount factor applied to installation costs each year
        "Median Installed Area (km²)": median_installed_area_per_year,  # Median cumulative installed panel area (same units as city_area) per year
        "Median Installed City Coverage (%)": np.divide(
            median_installed_area_per_year, city_area
        ),  # Median cumulative installed panel area (same units as city_area) per year
        "Median Incremental Capacity (MW)": median_incremental_capacity_per_year,  # Median incremental effective capacity added each year (MW)
    }
    st.session_state[scenario_name]["tracking_data"] = tracking_data


def monte_carlo_forecast(scenario_name, max_growth=1, sensitivity=None):
    """
    Monte Carlo Forecast that computes capacity, energy, revenue, and discounted installation costs.

    This updated version applies the logistic growth logic based on available area rather than capacity.
    The available area fraction (1 - area_installed/total_possible_area) modulates the growth rate,
    so that as the installed area nears the maximum possible area, the growth slows down.
    """
    import streamlit as st
    import numpy as np
    import pandas as pd
    from metalog import metalog

    # Retrieve simulation settings and scenario parameters.
    years = st.session_state.years
    iterations = st.session_state.iterations
    city_area = st.session_state[scenario_name]["city_area"]
    pv_area_percent = st.session_state[scenario_name][
        "pv_area_percent"
    ]  # fraction of city area available for panels
    initial_city_capacity = st.session_state[scenario_name]["initial_city_capacity"]
    base_panel_power = st.session_state[scenario_name][
        "panel_power"
    ]  # baseline panel power (W/m²)

    discount_rate = st.session_state[scenario_name]["install_discount"]
    panel_gain = st.session_state[scenario_name]["panel_gain"]
    panel_degradation = st.session_state[scenario_name]["panel_degradation_factor"]

    # Degradation of previously installed capacity.
    degradation_factor = 1 - panel_degradation

    # Pre-calculate panel power improvements for new installations.
    gain_factors = np.power((1 + panel_gain), np.arange(1, years + 1))
    panel_power_array = (
        base_panel_power * gain_factors
    )  # Improved panel power for each year.

    # Conversion constant: kWh per MW-year.
    conversion_constant = 1000 * 8760

    # Retrieve distributions.
    cf_dist = st.session_state[scenario_name]["Capacity Factor"]
    price_dist = st.session_state[scenario_name]["Energy Price ($/kWh)"]
    install_dist = st.session_state[scenario_name]["Installation Price($/kWWh)"]
    growth_dist = st.session_state[scenario_name]["Annual Growth Factor"]

    if sensitivity == "Annual Growth Factor":
        growth_rates = np.full([iterations, years], growth_dist)
    else:
        growth_rates = metalog.r(m=growth_dist, n=iterations * years).reshape(
            iterations, years
        )

    if sensitivity == "Capacity Factor":
        cf_samples = np.full([iterations, years], cf_dist)
    else:
        cf_samples = metalog.r(m=cf_dist, n=iterations * years).reshape(
            iterations, years
        )

    if sensitivity == "Energy Price ($/kWh)":
        price_samples = np.full([iterations, years], price_dist)
    else:
        price_samples = metalog.r(m=price_dist, n=iterations * years).reshape(
            iterations, years
        )

    install_samples = metalog.r(m=install_dist, n=iterations * years).reshape(
        iterations, years
    )

    # Pre-calculate discount factors for each year.
    discount_factors = 1 / np.power((1 + discount_rate), np.arange(1, years + 1))

    # Pre-allocate arrays for capacity and outputs.
    capacities = np.empty((iterations, years + 1))
    capacities[:, 0] = initial_city_capacity

    energy = np.empty((iterations, years))
    revenue = np.empty((iterations, years))
    costs = np.empty((iterations, years))

    # --- Tracking Variables for Time Series ---
    median_installed_area_per_year = []
    median_incremental_capacity_per_year = []

    # Calculate total possible area available for panel installations.
    total_possible_area = city_area * pv_area_percent
    # Determine initial area installed based on initial capacity.
    area_installed = np.full(iterations, initial_city_capacity / base_panel_power)
    # area_installed =  initial_city_capacity / base_panel_power
    # available_area = initial_city_capacity - area_installed

    # Simulation loop over forecast years.
    for year in range(years):

        # --- Logistic Growth Based on Area ---
        # Compute the available area fraction (1 means full availability, 0 means fully saturated).
        available_area_fraction = 1 - area_installed / total_possible_area

        # Sample growth rate for the current year.
        g_rate = growth_rates[:, year]
        # The growth factor is modulated by the available area fraction.
        growth_factor = 1 + np.minimum(g_rate, max_growth) * available_area_fraction

        # New capacity is last year's capacity scaled by the growth factor.
        new_capacity = capacities[:, year] * growth_factor

        # Calculate the additional capacity added this year.
        incremental_capacity = new_capacity - capacities[:, year]
        # Determine the area required for the incremental capacity based on current panel power.
        incremental_area = incremental_capacity / panel_power_array[year]
        # Update the cumulative installed area.
        area_installed += incremental_area

        median_incremental_capacity_per_year.append(np.median(incremental_capacity))
        median_installed_area_per_year.append(np.median(area_installed))

        # Apply degradation to previous capacity.
        effective_prev = capacities[:, year] * degradation_factor

        # Store the new capacity.
        capacities[:, year + 1] = incremental_capacity + effective_prev

        # Energy, Revenue, and Cost Calculations.
        energy[:, year] = (
            capacities[:, year + 1] * conversion_constant * cf_samples[:, year]
        )
        revenue[:, year] = energy[:, year] * price_samples[:, year]
        costs[:, year] = (
            incremental_capacity
            * 1e6
            * install_samples[:, year]
            * discount_factors[year]
        )

    # Build DataFrames for outputs.
    cap_df = pd.DataFrame(capacities.T)
    cap_df["Year"] = np.arange(0, years + 1)
    energy_df = pd.DataFrame(energy.T)
    energy_df["Year"] = np.arange(1, years + 1)
    revenue_df = pd.DataFrame(revenue.T)
    revenue_df["Year"] = np.arange(1, years + 1)
    cost_df = pd.DataFrame(costs.T)
    cost_df["Year"] = np.arange(1, years + 1)

    # Persist results in session state.
    st.session_state[scenario_name]["Capacity DF"] = cap_df
    st.session_state[scenario_name]["Energy DF"] = energy_df
    st.session_state[scenario_name]["Revenue DF"] = revenue_df
    st.session_state[scenario_name]["Install Cost DF"] = cost_df

    # Persist tracking time series data.
    tracking_data = {
        "New Panel Power Rating (W/m²)": panel_power_array,
        "Panel Degradation Factor (%)": np.cumprod([degradation_factor]),
        "Panel Power Gain Factor (%)": gain_factors,
        "Install Discount Factor (%)": discount_factors,
        "Median Installed Area (km²)": median_installed_area_per_year,
        "Median Installed City Coverage (%)": np.divide(
            median_installed_area_per_year, city_area
        ),
        "Median Incremental Capacity (MW)": median_incremental_capacity_per_year,
    }
    st.session_state[scenario_name]["tracking_data"] = tracking_data


def compute_cumulative_npv_revenue(scenario_name, discount_rate):
    """
    Computes the cumulative Net Present Value (NPV) for revenue using the median (P50)
    revenue forecast from st.session_state[scenario_name]["Revenue DF"].

    This function mimics the logic in create_npv_plot:
      - Retrieves the revenue forecast DataFrame,
      - Drops the "Year" column (if present),
      - Computes the median (P50) revenue for each simulation year,
      - Discounts each year's median revenue using the discount rate,
      - Computes the cumulative sum over the simulation period.

    Returns:
      The cumulative NPV (a single numeric value).
    """
    # Get discount rate and number of years from session state.
    years = st.session_state.get("years", 25)
    years_arr = np.arange(1, years + 1)

    # Retrieve the revenue forecast DataFrame.
    revenue_df = st.session_state[scenario_name].get("Revenue DF")
    if revenue_df is None or revenue_df.empty:
        st.error(
            "Revenue forecast data not available. Please run the simulation first."
        )
        return None

    # Remove the "Year" column if present.
    if "Year" in revenue_df.columns:
        revenue_df = revenue_df.drop("Year", axis=1)

    # Compute the median (P50) revenue for each year.
    p50_revenue = revenue_df.quantile(0.50, axis=1).values  # shape should be (years,)

    # Calculate discount factors for each year.
    discount_factors = np.array([(1 + discount_rate) ** t for t in years_arr])

    # Discount the median revenues.
    discounted_revenue = p50_revenue / discount_factors

    # Compute cumulative NPV (sum of discounted revenues over all years).
    cumulative_npv = np.sum(discounted_revenue)
    return cumulative_npv


def sensitivity_cumulative_npv_revenue(scenario_name, bounds_df, discount_rate):
    """
    For each slider-controlled parameter listed in bounds_df, this function:
      - Saves the baseline value,
      - Runs a baseline simulation to compute the cumulative NPV for revenue,
      - Temporarily sets the parameter to its lower bound, re-runs the simulation, and computes NPV,
      - Sets the parameter to its upper bound, re-runs the simulation, and computes NPV,
      - Restores the baseline value.

    Instead of creating a separate dictionary, the computed NPVs are added as new columns
    ("Baseline NPV", "NPV Low", and "NPV High") to the provided bounds_df.

    Returns:
      The updated bounds_df DataFrame.
    """
    # Save baseline parameter values for each parameter.
    baseline = {}
    for param in bounds_df["Parameter"]:
        baseline[param] = st.session_state[scenario_name].get(param)

    # Run baseline simulation.
    with st.spinner("Running baseline simulation..."):
        monte_carlo_forecast(scenario_name)
    baseline_npv = compute_cumulative_npv_revenue(scenario_name, discount_rate)
    if baseline_npv is None:
        st.error("Baseline simulation did not produce revenue data.")
        return bounds_df

    # Create lists to store NPV values.
    npv_low_list = []
    npv_high_list = []

    # Loop over each parameter row in the bounds DataFrame.
    for _, row in bounds_df.iterrows():
        param = row["Parameter"]

        low_val = row["Lower Bound"]
        high_val = row["Upper Bound"]

        # if row['Impact'] == 'negative':
        #     low_val,high_val = high_val,low_val

        # Test lower bound.
        st.session_state[scenario_name][param] = low_val
        with st.spinner(f"Running simulation with {param} = {low_val}..."):
            monte_carlo_forecast(scenario_name, sensitivity=param)
        npv_low = compute_cumulative_npv_revenue(scenario_name, discount_rate)

        # Test upper bound.
        st.session_state[scenario_name][param] = high_val
        with st.spinner(f"Running simulation with {param} = {high_val}..."):
            monte_carlo_forecast(scenario_name, sensitivity=param)
        npv_high = compute_cumulative_npv_revenue(scenario_name, discount_rate)

        npv_low_list.append(npv_low)
        npv_high_list.append(npv_high)

        # Restore baseline value for this parameter.
        st.session_state[scenario_name][param] = baseline[param]
        with st.spinner(f"Restoring baseline for {param}..."):
            monte_carlo_forecast(scenario_name)

    # Add computed values to bounds_df.
    bounds_df["Baseline"] = baseline_npv  # Same for all parameters.
    bounds_df["Low"] = npv_low_list
    bounds_df["High"] = npv_high_list

    bounds_df["Range"] = bounds_df["High"] - bounds_df["Low"]
    bounds_df.sort_values("Range", ascending=True, inplace=True)

    return bounds_df


def compute_metric_summary(scenario_name, metric, method="sum"):
    """
    Computes the total energy production using the median (P50) energy forecast
    from st.session_state[scenario_name]["Energy DF"].

    It drops the "Year" column if present, computes the median energy production
    for each year, and returns the sum over all simulation years.

    Returns:
      The total energy production (a single numeric value).
    """
    df = st.session_state[scenario_name].get(f"{metric} DF")
    if df is None or df.empty:
        st.error(
            f"{metric} forecast data not available. Please run the simulation first."
        )
        return None
    # Remove the "Year" column if present.
    if "Year" in df.columns:
        df = df.drop("Year", axis=1)

    # Compute the median (P50) energy production for each year.
    p50 = df.quantile(0.50, axis=1).values
    # Sum the median energy production across all years.
    if method == "sum":
        return np.sum(p50)
    if method == "max":
        return np.max(p50)


def sensitivity_total(scenario_name, bounds_df, metric, method="sum"):
    """
    For each slider-controlled parameter listed in bounds_df, this function:
      - Saves the baseline value from st.session_state,
      - Runs a baseline simulation to compute the total energy production,
      - Temporarily sets the parameter to its lower bound, re-runs the simulation, and computes energy,
      - Sets the parameter to its upper bound, re-runs the simulation, and computes energy,
      - Restores the baseline value.

    The computed total energy production values are added as new columns to bounds_df:
      - "Baseline Energy": The baseline total energy production (same for all parameters),
      - "Energy Low": Total energy production when the parameter is set to its lower bound,
      - "Energy High": Total energy production when the parameter is set to its upper bound.

    Returns:
      The updated bounds_df DataFrame.
    """
    # Save baseline parameter values.
    baseline = {}
    for param in bounds_df["Parameter"]:
        baseline[param] = st.session_state[scenario_name].get(param)

    # Run baseline simulation.
    with st.spinner("Running baseline simulation..."):
        monte_carlo_forecast(scenario_name)
    baseline_energy = compute_metric_summary(scenario_name, metric, method)
    if baseline_energy is None:
        st.error("Baseline simulation did not produce energy data.")
        return bounds_df

    energy_low_list = []
    energy_high_list = []

    # Loop over each parameter in the bounds DataFrame.
    for idx, row in bounds_df.iterrows():
        param = row["Parameter"]
        low_val = row["Lower Bound"]
        high_val = row["Upper Bound"]

        # Test the lower bound.
        st.session_state[scenario_name][param] = low_val
        with st.spinner(f"Running simulation with {param} = {low_val}..."):
            monte_carlo_forecast(scenario_name, sensitivity=param)
        energy_low = compute_metric_summary(scenario_name, metric, method)

        # Test the upper bound.
        st.session_state[scenario_name][param] = high_val
        with st.spinner(f"Running simulation with {param} = {high_val}..."):
            monte_carlo_forecast(scenario_name, sensitivity=param)
        energy_high = compute_metric_summary(scenario_name, metric, method)

        energy_low_list.append(energy_low)
        energy_high_list.append(energy_high)

        # Restore the baseline value.
        st.session_state[scenario_name][param] = baseline[param]
        with st.spinner(f"Restoring baseline for {param}..."):
            monte_carlo_forecast(scenario_name)

    # Add computed energy production values to bounds_df.
    bounds_df["Baseline"] = baseline_energy  # Same for all parameters.
    bounds_df["Low"] = energy_low_list
    bounds_df["High"] = energy_high_list

    bounds_df["Range"] = bounds_df["High"] - bounds_df["Low"]
    bounds_df.sort_values("Range", ascending=True, inplace=True)

    return bounds_df
