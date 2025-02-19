import streamlit as st

class Info:
    @staticmethod
    def energy_sensitivity():
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

    @staticmethod
    def revenue_sensitivity():
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

    @staticmethod
    def help():
        st.markdown(
            r"""
            ## Help & Usage Guide

            **Overview:**  
            This application uses Monte Carlo simulations to forecast rooftop solar capacity, energy production, revenues, and installation costs in Calgary. The model incorporates a logistic (S-curve) growth framework with uncertainty captured through user-configured probability distributions.

            ## Controls & Distributions
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
            
            ## Forecasts
            - View the simulated forecasts for **Capacity (MW)**, **Energy (kWh/year)**, **PV Revenue**, and **Installation Cost**.
            - The plots display uncertainty bands (P10, P50, and P90) so you can visualize the variability in the predictions.
            
            ## NPV
            - Calculate the Net Present Value (NPV) of future cash flows.
            - Choose between revenue and installation cost forecasts, adjust the discount rate, and toggle cumulative versus annual values.
            - This helps in assessing the investment’s financial viability.
            
            ## Change Distribution
            - Examine histograms (PDF and CDF) for year-to-year percent changes in the forecasts.
            - This view is useful to understand the variability and likelihood of different changes from one year to the next.
            

            ## Timeseries Metrics
            - Track detailed metrics such as:
                - **New Panel Power Rating:** The improved performance for new installations.
                - **Panel Degradation Factor:** How much the performance of existing panels decreases each year.
                - **Panel Power Gain Factor:** The efficiency gains for new panels.
                - **Installation Discount Factor:** How discounting affects cost calculations.
                - **Median Installed Area and Incremental Capacity:** How the cumulative installations evolve over time.
            - These metrics offer granular insights into how each component drives overall system performance.

            ## Energy Sensitivity 
               Analyzes the sensitivity of total energy production to key model parameters. For each parameter, the simulation is re-run with the parameter set to its lower and upper bounds. The impact on total energy production is visualized using a tornado diagram, highlighting which parameters most influence energy output.

               
            ## Revenue Sensitivity
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

    @staticmethod
    def monte_carlo_simulation():
        st.markdown(
            r"""
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
        )

    @staticmethod
    def npv():
        st.markdown(
            r"""
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
        )

    @staticmethod
    def change_distribution():
        st.markdown(
            r"""
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
        )

    @staticmethod
    def time_series_metrics():
        st.markdown(
            r"""
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
        )

    @staticmethod
    def dispatch(info_type: str):

        func = getattr(Info, info_type, None)

        if func is not None:
            func()
        else:
            st.error(f"Invalid info type: {info_type}")
