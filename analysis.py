import streamlit as st
import numpy as np
import pandas as pd
from metalog import metalog
from numba import njit
from joblib import Parallel, delayed

calgary_capacity_2023 = 55


class Analysis:
    @staticmethod
    def initialize_city(panel_power, city_capacity):
        installs = pd.DataFrame(
            {
                "installed_kw": [3896, 7282, 6165, 10074, 20124],
                "year": [2019, 2020, 2021, 2022, 2023],
            }
        )

        initial_city_capacity_kw = city_capacity * 1e3

        installs_with_data = installs.installed_kw.sum()
        n_year = installs.shape[0]

        missing_capacity = initial_city_capacity_kw - installs_with_data

        installs.loc[n_year] = [missing_capacity, None]

        original_household_df = pd.read_csv(
            "data/Residential_Solar_Photovoltaic__PV__System_Potential_20250217.csv"
        )
        original_household_df["ac_annualy_total"] = (
            original_household_df.number_of_panels
            * original_household_df.ac_annually_per_panel
        )

        original_household_df = (
            original_household_df.groupby("address")
            .sum()
            .drop(columns=["ac_annually_per_panel"])
        ).reset_index()

        median_panels_installed = original_household_df.number_of_panels.median()

        calc_installs = lambda capacity: int(
            np.floor(capacity * 1000 / (median_panels_installed * panel_power))
        )

        installs["households_installed"] = installs.installed_kw.apply(calc_installs)

        installed_households = installs.households_installed.sum()

        drop_indices = np.random.choice(
            original_household_df.index, installed_households, replace=False
        )
        original_household_df.drop(drop_indices, inplace=True)

        latest_installs = installs.loc[
            installs.year == installs.year.max(), "households_installed"
        ].squeeze()

        return original_household_df, latest_installs, installed_households

    @staticmethod
    # @st.cache_data
    def monte_carlo_forecast(
        scenario_name,
        years,
        iterations,
        panel_power,
        panel_gain,
        sensitivity=None,
        city_capacity=calgary_capacity_2023,
    ):

        original_household_df, latest_installs, total_installs = (
            Analysis.initialize_city(panel_power, city_capacity)
        )

        # price_dist = st.session_state[scenario_name]["Energy Price ($/kWh)"]
        # install_dist = st.session_state[scenario_name]["Installation Price($/kWWh)"]
        growth_dist = st.session_state[scenario_name]["Annual Growth Factor"]
        end_year = years + 1

        if sensitivity == "Annual Growth Factor":
            growth_rates = np.full([iterations, years], growth_dist)
        else:
            growth_rates = metalog.r(m=growth_dist, n=iterations * end_year).reshape(
                iterations, end_year
            )

        # if sensitivity == "Energy Price ($/kWh)":
        #     price_samples = np.full([iterations, years], price_dist)
        # else:
        #     price_samples = metalog.r(m=price_dist, n=iterations * end_year).reshape(
        #         iterations, end_year
        #     )

        # if sensitivity == "Installation Price($/kWWh)":
        #     install_samples = np.full([iterations, years], install_dist)
        # else:
        #     install_samples = metalog.r(
        #         m=install_dist, n=iterations * end_year
        #     ).reshape(iterations, end_year)

        growth_df = pd.DataFrame(growth_rates)

        capacity_df = pd.DataFrame(np.zeros((iterations, years + 1)))
        install_df = pd.DataFrame(np.zeros((iterations, years + 1)))
        energy_df = pd.DataFrame(np.zeros((iterations, years + 1)))
        # costs_df = pd.DataFrame(np.zeros((iterations, years + 1)))
        # revenue_df = pd.DataFrame(np.zeros((iterations, years + 1)))

        panel_gain_factors = np.power((1 + panel_gain), np.arange(0, end_year))
        panel_powers = panel_power * panel_gain_factors

        total_household = original_household_df.shape[0]

        install_df.loc[:, 0] = total_installs
        capacity_df.loc[:, 0] = city_capacity

        for year in range(1, years + 1):
            growth_rate = growth_df[year]
            current_installs = install_df.loc[:, : year - 1].sum(axis=1)

            new_installs = (
                (
                    growth_rate
                    * current_installs
                    * (1 - current_installs / total_household)
                )
                .clip(lower=0, upper=total_household)
                .astype(int)
            )
            install_df.loc[:, year] = new_installs

        # -------------------------------------------------
        # Data Preparation
        # -------------------------------------------------

        # Convert the 'number_of_panels' column from the original household DataFrame to a NumPy array.
        panels_array = original_household_df["number_of_panels"].to_numpy()

        # Convert the 'ac_annualy_total' column from the original household DataFrame to a NumPy array.
        energy_array = original_household_df["ac_annualy_total"].to_numpy()

        # Convert the install DataFrame to a NumPy array.
        # It is assumed to have a shape of (iterations, years+1), where each row is an iteration and each column is a year.
        install_array = install_df.to_numpy()

        # Create a NumPy array for panel powers.
        # The first element is 0 (often a dummy value for year 0), and then we include panel powers for years 1 through 'years'.
        panel_powers_array = np.array(
            [panel_power] + [panel_powers[y] for y in range(1, years + 1)]
        )

        # -------------------------------------------------
        # Define the JIT-compiled Function with Numba
        # -------------------------------------------------

        @njit
        def calc_capacity_numba(
            iteration,
            install_array,
            panels_array,
            energy_array,
            panel_powers_array,
            city_capacity,
        ):
            # Get the total number of households from the panels_array.
            n_households = panels_array.shape[0]

            # Preallocate arrays to store the capacity and energy for each year (including a dummy index 0).
            capacity = np.empty(years + 1)
            energy = np.empty(years + 1)

            # Generate a random permutation of household indices for this iteration.
            # This simulates random selection of households.
            permuted = np.random.permutation(n_households)

            # 'pointer' keeps track of the current position in the permuted array.
            pointer = 0

            # Set the initial values for year 0 (assumed to be unused) to 0.
            capacity[0] = city_capacity
            energy[0] = 0.0

            # Loop over each year starting from year 1.
            for year in range(1, years + 1):
                # Get the number of installations for the current iteration and year.
                n_installs = int(install_array[iteration, year])

                # Calculate the next pointer position after selecting 'n_installs' households.
                pointer_next = pointer + n_installs

                # Initialize counters for the number of panels and energy installed for the current year.
                installed_panels = 0
                installed_energy = 0.0

                # Loop over the selected households for the current year.
                for i in range(pointer, pointer_next):
                    # Retrieve the actual household index from the random permutation.
                    idx = permuted[i]
                    # Add the number of panels from the selected household.
                    installed_panels += panels_array[idx]
                    # Add the energy value from the selected household.
                    installed_energy += energy_array[idx]

                # Calculate the capacity for the current year.
                # Multiply the total number of installed panels by the panel power (convert to MW by dividing by 1e6).
                capacity[year] = installed_panels * panel_powers_array[year] / 1e6

                # Calculate the energy for the current year (convert units by dividing by 1e3).
                energy[year] = installed_energy / 1e3

                # Update the pointer to the next position for the following year.
                pointer = pointer_next

            # Return the capacity and energy arrays for the current iteration.
            return capacity, energy

        # -------------------------------------------------
        # Running the Simulation Over All Iterations
        # -------------------------------------------------

        # Preallocate 2D arrays to store the results for all iterations.
        # Each row corresponds to an iteration, and each column corresponds to a year.
        results = Parallel(n_jobs=-1)(
            delayed(calc_capacity_numba)(
                i,
                install_array,
                panels_array,
                energy_array,
                panel_powers_array,
                city_capacity,
            )
            for i in range(iterations)
        )
        results = np.array(results)
        capacity_df = pd.DataFrame(results[:, 0])
        energy_df = pd.DataFrame(results[:, 1])

        # -------------------------------------------------
        # Converting Results Back to DataFrames
        # -------------------------------------------------

        capacity_df.drop(columns=0, inplace=True)
        install_df.drop(columns=0, inplace=True)
        energy_df.drop(columns=0, inplace=True)

        store = st.session_state[scenario_name]

        store["Annual Capacity DF"] = capacity_df
        store["Cumulative Capacity DF"] = capacity_df.cumsum(axis=1)
        store["Annual Households Installed DF"] = install_df
        store["Cumulative Households Installed DF"] = install_df.cumsum(axis=1)
        store["Energy DF"] = energy_df.cumsum(axis=1) / 1e3

        tracking_data = {
            "New Panel Power Rating (W/m²)": panel_powers,
            # "Panel Power Gain Factor (%)": panel_gain_factors,
        }
        store["tracking_data"] = tracking_data

        st.session_state["model_run"] = True

    @staticmethod
    def compute_cumulative_npv_revenue(scenario_name, discount_rate):
        years = st.session_state.get("years", 25)
        years_arr = np.arange(1, years + 1)

        revenue_df = st.session_state[scenario_name].get("Revenue DF")
        if revenue_df is None or revenue_df.empty:
            st.error(
                "Revenue forecast data not available. Please run the simulation first."
            )
            return None

        if "Year" in revenue_df.columns:
            revenue_df = revenue_df.drop("Year", axis=1)

        p50_revenue = revenue_df.quantile(0.50, axis=1).values

        discount_factors = np.array([(1 + discount_rate) ** t for t in years_arr])

        discounted_revenue = p50_revenue / discount_factors

        cumulative_npv = np.sum(discounted_revenue)
        return cumulative_npv

    @staticmethod
    def sensitivity_cumulative_npv_revenue(scenario_name, bounds_df, discount_rate):
        baseline = {}
        for param in bounds_df["Parameter"]:
            baseline[param] = st.session_state[scenario_name].get(param)

        with st.spinner("Running baseline simulation..."):
            Analysis.monte_carlo_forecast(scenario_name)
        baseline_npv = Analysis.compute_cumulative_npv_revenue(
            scenario_name, discount_rate
        )
        if baseline_npv is None:
            st.error("Baseline simulation did not produce revenue data.")
            return bounds_df

        npv_low_list = []
        npv_high_list = []

        for _, row in bounds_df.iterrows():
            param = row["Parameter"]

            low_val = row["Lower Bound"]
            high_val = row["Upper Bound"]

            st.session_state[scenario_name][param] = low_val
            with st.spinner(f"Running simulation with {param} = {low_val}..."):
                Analysis.monte_carlo_forecast(scenario_name, sensitivity=param)
            npv_low = Analysis.compute_cumulative_npv_revenue(
                scenario_name, discount_rate
            )

            st.session_state[scenario_name][param] = high_val
            with st.spinner(f"Running simulation with {param} = {high_val}..."):
                Analysis.monte_carlo_forecast(scenario_name, sensitivity=param)
            npv_high = Analysis.compute_cumulative_npv_revenue(
                scenario_name, discount_rate
            )

            npv_low_list.append(npv_low)
            npv_high_list.append(npv_high)

            st.session_state[scenario_name][param] = baseline[param]
            with st.spinner(f"Restoring baseline for {param}..."):
                Analysis.monte_carlo_forecast(scenario_name)

        bounds_df["Baseline"] = baseline_npv
        bounds_df["Low"] = npv_low_list
        bounds_df["High"] = npv_high_list

        bounds_df["Range"] = bounds_df["High"] - bounds_df["Low"]
        bounds_df.sort_values("Range", ascending=True, inplace=True)

        return bounds_df

    @staticmethod
    def compute_metric_summary(scenario_name, metric, method="sum"):
        df = st.session_state[scenario_name].get(f"{metric} DF")
        if df is None or df.empty:
            st.error(
                f"{metric} forecast data not available. Please run the simulation first."
            )
            return None

        if "Year" in df.columns:
            df = df.drop("Year", axis=1)

        p50 = df.quantile(0.50, axis=1).values

        if method == "sum":
            return np.sum(p50)
        if method == "max":
            return np.max(p50)

    @staticmethod
    def sensitivity_total(scenario_name, bounds_df, metric, method="sum"):
        baseline = {}
        for param in bounds_df["Parameter"]:
            baseline[param] = st.session_state[scenario_name].get(param)

        with st.spinner("Running baseline simulation..."):
            Analysis.monte_carlo_forecast(scenario_name)
        baseline_energy = Analysis.compute_metric_summary(scenario_name, metric, method)
        if baseline_energy is None:
            st.error("Baseline simulation did not produce energy data.")
            return bounds_df

        energy_low_list = []
        energy_high_list = []

        for idx, row in bounds_df.iterrows():
            param = row["Parameter"]
            low_val = row["Lower Bound"]
            high_val = row["Upper Bound"]

            st.session_state[scenario_name][param] = low_val
            with st.spinner(f"Running simulation with {param} = {low_val}..."):
                Analysis.monte_carlo_forecast(scenario_name, sensitivity=param)
            energy_low = Analysis.compute_metric_summary(scenario_name, metric, method)

            st.session_state[scenario_name][param] = high_val
            with st.spinner(f"Running simulation with {param} = {high_val}..."):
                Analysis.monte_carlo_forecast(scenario_name, sensitivity=param)
            energy_high = Analysis.compute_metric_summary(scenario_name, metric, method)

            energy_low_list.append(energy_low)
            energy_high_list.append(energy_high)

            st.session_state[scenario_name][param] = baseline[param]
            with st.spinner(f"Restoring baseline for {param}..."):
                Analysis.monte_carlo_forecast(scenario_name)

        bounds_df["Baseline"] = baseline_energy
        bounds_df["Low"] = energy_low_list
        bounds_df["High"] = energy_high_list

        bounds_df["Range"] = bounds_df["High"] - bounds_df["Low"]
        bounds_df.sort_values("Range", ascending=True, inplace=True)

        return bounds_df
