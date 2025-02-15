import streamlit as st
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


np.float_ = np.float64


def create_dist_plot(m: str):
    """
    Create a distribution plot showing the PDF and CDF.
    """
    quantiles = m["M"].iloc[:, 1]
    pdf_values = m["M"].iloc[:, 0]
    cdf_values = m["M"]["y"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=quantiles,
            y=pdf_values / sum(pdf_values),
            mode="lines",
            name="PDF",
            line=dict(color="blue"),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=quantiles,
            y=cdf_values,
            mode="lines",
            name="CDF",
            line=dict(color="red", dash="dash"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        xaxis=dict(title="Value"),
        yaxis=dict(title="PDF", title_font_color="blue", tickfont_color="blue"),
        yaxis2=dict(
            title="CDF",
            title_font_color="red",
            tickfont_color="red",
            overlaying="y",
            side="right",
        ),
        legend=dict(x=0, y=1.1, orientation="h"),
        template="plotly",
        hovermode="x unified",
    )
    return fig


def add_forecast_traces(
    fig: go.Figure,
    years: int,
    sim_data,
    scenario_label,
    color,
    line_dash="solid",
    width=2,
):
    """
    Helper function to add forecast traces (p10, p50, p90) to a figure.
    Now includes p10 and p90 in the legend.
    """
    x_vals = np.arange(1, years + 1)
    p10_vals = np.percentile(sim_data, 10, axis=1)
    p50_vals = np.median(sim_data, axis=1)
    p90_vals = np.percentile(sim_data, 90, axis=1)

    # p90 trace
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=p90_vals,
            mode="lines",
            name=f"{scenario_label} - P90",
            line=dict(color=color, dash="dot", width=width),
            legendgroup=scenario_label,
        )
    )

    # p50 trace (Median)
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=p50_vals,
            mode="lines",
            name=f"{scenario_label} - P50",
            line=dict(color=color, dash=line_dash, width=width),
            legendgroup=scenario_label,
        )
    )

    # p10 trace
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=p10_vals,
            mode="lines",
            name=f"{scenario_label} - P10",
            line=dict(color=color, dash="dot", width=width),
            legendgroup=scenario_label,
        )
    )


def update_forecast_plots(
    scenario_name,
):

    capacity_fig = st.session_state["capacity_fig"]
    energy_fig = st.session_state["energy_fig"]
    revenue_fig = st.session_state["revenue_fig"]
    cost_fig = st.session_state["cost_fig"]

    cap_df = st.session_state[scenario_name]["Capacity DF"]
    energy_df = st.session_state[scenario_name]["Energy DF"]
    revenue_df = st.session_state[scenario_name]["Revenue DF"]
    cost_df = st.session_state[scenario_name]["Install Cost DF"]

    years = st.session_state.years

    color = st.session_state[scenario_name]["color"]

    add_forecast_traces(
        capacity_fig, years, cap_df, scenario_name + " - Capacity", color
    )
    add_forecast_traces(
        energy_fig, years, energy_df, scenario_name + " - Energy", color
    )
    add_forecast_traces(
        revenue_fig, years, revenue_df, scenario_name + " - Revenue", color
    )
    add_forecast_traces(cost_fig, years, cost_df, scenario_name + " - Cost", color)
    # add_forecast_traces(net_fig, years, net_df, scenario_name + " - Net", color)

    # Create capacity forecast figure

    capacity_fig.update_layout(
        title="Capacity Forecast (MW)",
        xaxis_title="Year",
        yaxis_title="Capacity (MW)",
    )
    energy_fig.update_layout(
        title="Energy Forecast (kWh/year)",
        xaxis_title="Year",
        yaxis_title="Energy Production (kWh/year)",
    )
    revenue_fig.update_layout(
        title="PV Revenue Forecast ($/year)",
        xaxis_title="Year",
        yaxis_title="Revenue ($/year)",
    )
    cost_fig.update_layout(
        title="Total Installation Cost Forecast ($/year)",
        xaxis_title="Year",
        yaxis_title="Installation Costs ($/year)",
    )


def create_npv_plot(forecast_name, discount_rate, cumulative=False):

    years = st.session_state.years

    # Determine the title based on cumulative or annual values.
    if cumulative:
        title = f"Cumulative NPV Forecast of P10, P50, and P90 {forecast_name}"
    else:
        title = f"NPV Forecast of P10, P50, and P90 {forecast_name}"
    years_arr = np.arange(1, years + 1)

    npv_fig = go.Figure()
    forecast_key = f"{forecast_name} DF"

    # Loop through each scenario.
    for scenario in st.session_state.scenarios:
        try:
            # Retrieve the DataFrame and drop the "Year" column.
            df_scenario = st.session_state[scenario][forecast_key].drop("Year", axis=1)

            # Compute the 10th, 50th, and 90th percentiles across simulation iterations for each forecast year.
            p10_net = df_scenario.quantile(0.10, axis=1).values
            p50_net = df_scenario.quantile(0.50, axis=1).values
            p90_net = df_scenario.quantile(0.90, axis=1).values

            # Calculate discount factors.
            discount_factors = np.array([(1 + discount_rate) ** t for t in years_arr])

            # Discount the cash flows.
            discounted_p10 = p10_net / discount_factors
            discounted_p50 = p50_net / discount_factors
            discounted_p90 = p90_net / discount_factors

            # If cumulative, compute the cumulative sum.
            if cumulative:
                discounted_p10 = np.cumsum(discounted_p10)
                discounted_p50 = np.cumsum(discounted_p50)
                discounted_p90 = np.cumsum(discounted_p90)

            # Choose the plotting method based on the cumulative flag.
            if cumulative:
                # Use line traces for cumulative values.
                npv_fig.add_trace(
                    go.Scatter(
                        x=years_arr,
                        y=discounted_p10,
                        mode="lines+markers",
                        name=f"{scenario} - P10",
                        line=dict(
                            color=st.session_state[scenario]["color"], dash="dot"
                        ),
                    )
                )
                npv_fig.add_trace(
                    go.Scatter(
                        x=years_arr,
                        y=discounted_p50,
                        mode="lines+markers",
                        name=f"{scenario} - P50",
                        line=dict(
                            color=st.session_state[scenario]["color"], dash="solid"
                        ),
                    )
                )
                npv_fig.add_trace(
                    go.Scatter(
                        x=years_arr,
                        y=discounted_p90,
                        mode="lines+markers",
                        name=f"{scenario} - P90",
                        line=dict(
                            color=st.session_state[scenario]["color"], dash="dash"
                        ),
                    )
                )
            else:
                # Use bar traces for annual (non-cumulative) values.
                npv_fig.add_trace(
                    go.Bar(
                        x=years_arr,
                        y=discounted_p10,
                        name=f"{scenario} - P10",
                        marker_color=st.session_state[scenario]["color"],
                    )
                )
                npv_fig.add_trace(
                    go.Bar(
                        x=years_arr,
                        y=discounted_p50,
                        name=f"{scenario} - P50",
                        marker_color=st.session_state[scenario]["color"],
                    )
                )
                npv_fig.add_trace(
                    go.Bar(
                        x=years_arr,
                        y=discounted_p90,
                        name=f"{scenario} - P90",
                        marker_color=st.session_state[scenario]["color"],
                    )
                )
        except Exception as e:
            st.error(f"Error computing NPV for {scenario}: {e}")

    # Update layout.
    if cumulative:
        npv_fig.update_layout(
            title=title,
            xaxis=dict(title="Year"),
            yaxis=dict(title="NPV ($)"),
            height=800,
            hovermode="x unified",
        )
    else:
        npv_fig.update_layout(
            title=title,
            xaxis=dict(title="Year"),
            yaxis=dict(title="NPV ($)"),
            barmode="group",  # grouped bar chart for multiple traces per year
            height=800,
            hovermode="x unified",
        )
    st.plotly_chart(npv_fig, use_container_width=True)


def plot_all_scenarios_pdf_cdf_histograms(forecast_name):
    """
    Generates a Plotly figure with two subplots:
      - Left: PDF histogram (probability density) of the year-to-year percent changes.
      - Right: CDF histogram (cumulative density) of the year-to-year percent changes.

    One trace per scenario is added to both subplots.

    Parameters:
      - scenario_names (list of str): A list of scenario keys in st.session_state.
      - forecast_name (str): The forecast to analyze (e.g., "Capacity", "Energy",
                             "Revenue").

    Returns:
      - A Plotly figure object containing the two subplots with the histograms.

    Assumptions:
      - For each scenario in scenario_names, the forecast DataFrame is stored in
        st.session_state[scenario] under the key "<forecast_name> DF".
      - Each DataFrame contains a "Year" column and one or more iteration columns.
    """

    # Create a subplot figure with 1 row and 2 columns.
    fig = make_subplots(
        rows=2, cols=1, subplot_titles=["PDF Histogram", "CDF Histogram"]
    )

    # Iterate over each scenario to add traces.
    for scenario in st.session_state.scenarios:
        # Retrieve the forecast DataFrame.
        forecast_key = f"{forecast_name} DF"
        if forecast_key not in st.session_state[scenario]:
            raise ValueError(
                f"{forecast_key} not found in st.session_state[{scenario}]."
            )
        df = st.session_state[scenario][forecast_key]

        # Identify iteration columns (all columns except "Year").
        iter_cols = [col for col in df.columns if col != "Year"]

        # Compute the year-to-year percent changes; drop the first row (NaN).
        pct_change_df = df[iter_cols].pct_change().iloc[1:]

        # Flatten all percent change values into a single array and remove any NaNs.
        all_pct_changes = pct_change_df.values.flatten()
        all_pct_changes = all_pct_changes[~np.isnan(all_pct_changes)]

        # Create a PDF histogram trace.
        pdf_trace = go.Histogram(
            x=all_pct_changes,
            name=scenario + "PDF",
            opacity=0.75,
            nbinsx=30,
            histnorm="probability density",
            marker_color=st.session_state[scenario]["color"],
            legendgroup=scenario,
        )

        # Create a CDF histogram trace (using cumulative distribution).
        cdf_trace = go.Histogram(
            x=all_pct_changes,
            name=scenario + "CDF",
            opacity=0.75,
            nbinsx=30,
            histnorm="probability density",
            cumulative=dict(enabled=True),
            marker_color=st.session_state[scenario]["color"],
            legendgroup=scenario,
        )

        # Add the traces to the appropriate subplots.
        fig.add_trace(pdf_trace, row=1, col=1)
        fig.add_trace(cdf_trace, row=2, col=1)

    # Update the layout: overlay the histograms and set unified hover mode.
    fig.update_layout(
        barmode="overlay",
        hovermode="x unified",
        title=f"Year-to-Year Percent Change Distribution for {forecast_name}",
        height=1000,
    )

    # Set axis titles for both subplots.
    fig.update_xaxes(title_text="Year-to-Year Percent Change", row=1, col=1)
    fig.update_yaxes(title_text="Probability Density", row=1, col=1)

    fig.update_xaxes(title_text="Year-to-Year Percent Change", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Density", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def plot_forecast_violin_overlay(forecast_name):
    """
    Creates an overlaid violin plot for a specific forecast across all years for multiple scenarios.

    For each scenario in scenario_names, the function:
      - Retrieves the forecast DataFrame from st.session_state[scenario] using the key "<forecast_name> DF".
      - Reshapes the DataFrame into long format (with columns "Year", "Value", and "Scenario").

    All scenarios are combined into one plot so that the distribution for each year is
    visualized with separate violin traces for each scenario.

    The figure uses hovermode="x unified" for an improved interactive experience.

    Parameters:
      - scenario_names (list of str): A list of scenario keys in st.session_state.
      - forecast_name (str): The forecast to analyze (e.g., "Capacity", "Energy",
                             "Revenue", "Cost", or "Net").

    Returns:
      - A Plotly figure object containing the overlaid violin plot.
    """

    all_data = []

    forecast_key = f"{forecast_name} DF"

    # Loop through each scenario, retrieve and reshape the data.
    for scenario in st.session_state.scenarios:

        df = st.session_state[scenario][forecast_key]
        # Identify iteration columns (all columns except "Year").
        iter_cols = [col for col in df.columns if col != "Year"]

        # Convert the wide DataFrame to long format.
        df_long = pd.melt(
            df,
            id_vars="Year",
            value_vars=iter_cols,
            var_name="Iteration",
            value_name="Value",
        )
        df_long["Scenario"] = scenario
        all_data.append(df_long)

    # Combine all scenario data into one DataFrame.
    combined_df = pd.concat(all_data, ignore_index=True)

    # Create the figure.
    fig = go.Figure()

    # For each scenario, add a violin trace.
    for scenario in combined_df["Scenario"].unique():
        scenario_df = combined_df[combined_df["Scenario"] == scenario]
        fig.add_trace(
            go.Violin(
                x=scenario_df["Year"],
                y=scenario_df["Value"],
                name=scenario,
                box_visible=True,
                meanline_visible=True,
                opacity=0.7,
                side="both",  # Center the violins on each x value.
                spanmode="hard",
            )
        )

    fig.update_layout(
        title=f"Violin Plot of {forecast_name} by Year for Multiple Scenarios",
        hovermode="x unified",
        xaxis_title="Year",
        yaxis_title=forecast_name,
        height=1000,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_forecast_violin_overlay_faceted(
    forecast_list=["Capacity", "Energy", "Revenue", "Cost", "Net", "Cost Per MW"]
):
    """
    Creates a faceted violin plot for multiple forecast types across multiple scenarios.

    For each forecast type in forecast_list and for each scenario in scenario_names, the function:
      - Retrieves the forecast DataFrame from st.session_state[scenario] using the key "<forecast> DF".
      - Reshapes the DataFrame into long format with columns "Year", "Value", and adds columns "Scenario" and "Forecast".

    The combined data is used to create a faceted violin plot (with one row per forecast type) that shows
    the distribution of simulation iterations for each year.

    Parameters:
      - scenario_names (list of str): A list of scenario keys in st.session_state.
      - forecast_list (list of str): A list of forecast types to compile. Defaults to
                                     ["Capacity", "Energy", "Revenue", "Cost", "Net", "Cost Per MW"].

    Returns:
      - A Plotly figure object containing the faceted violin plots.
    """

    all_data = []

    # Loop over each forecast type and scenario to compile data.
    for forecast in forecast_list:
        forecast_key = f"{forecast} DF"
        for scenario in st.session_state.scenarios:

            df = st.session_state[scenario][forecast_key]
            # Identify iteration columns (all columns except "Year").
            iter_cols = [col for col in df.columns if col != "Year"]

            # Melt the DataFrame into long format.
            df_long = pd.melt(
                df,
                id_vars="Year",
                value_vars=iter_cols,
                var_name="Iteration",
                value_name="Value",
            )
            df_long["Scenario"] = scenario
            df_long["Forecast"] = forecast
            all_data.append(df_long)

    # Combine all data into a single DataFrame.
    if not all_data:
        raise ValueError(
            "No data found for the specified scenarios and forecast types."
        )
    combined_df = pd.concat(all_data, ignore_index=True)

    # Create the faceted violin plot.
    fig = px.violin(
        combined_df,
        x="Year",
        y="Value",
        color="Scenario",
        facet_row="Forecast",
        box=True,  # Display box plots inside the violins.
        points="all",  # Show all individual data points.
        title="Violin Plots of Forecasts by Year (Faceted by Forecast Type)",
        labels={"Value": "Forecast Value"},
    )

    # Update layout for unified hover and improve spacing.
    fig.update_layout(hovermode="x unified", margin=dict(l=50, r=50, t=50, b=50))

    # Optionally, adjust the facet row spacing for better readability.
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    st.plotly_chart(fig, use_container_width=True)


def plot_tracking_data(metric):
    """
    Plots the tracking data time series for each scenario overlaid using Plotly.

    For each metric in the tracking data (e.g., "New Panel Power Per Year",
    "Annual Degradation Factor", etc.), this function creates an interactive plot
    where the x-axis represents the simulation year and each scenario's time series is
    overlaid as a line.

    Parameters:
        scenario_names (list of str): List of scenario names whose tracking data will be plotted.
    """

    years = st.session_state.years
    years_arr = list(range(1, years + 1))

    fig = go.Figure()

    for scenario in st.session_state.scenarios:
        y_data = st.session_state[scenario].get("tracking_data", {})[metric]
        color = st.session_state[scenario]["color"]

        fig.add_trace(
            go.Scatter(
                x=years_arr,
                y=y_data,
                mode="lines+markers",
                name=scenario,
                line_color=color,
            )
        )

    fig.update_layout(
        title=metric,
        xaxis_title="Year",
        yaxis_title=metric,
        template="plotly_white",
        height=1000,
    )
    st.plotly_chart(fig, use_container_width=True)


def create_tornado_figure(bounds_df: pd.DataFrame, title, x_axis_title):
    """
    Creates a tornado diagram using the bounds DataFrame that includes:
      - "Parameter": The parameter name.
      - "Baseline NPV": The baseline cumulative NPV (same for all parameters).
      - "NPV Low": The cumulative NPV when the parameter is set to its lower bound.
      - "NPV High": The cumulative NPV when the parameter is set to its upper bound.

    The rows are ordered by the maximum absolute deviation from the baseline (in descending order).
    For each parameter:
      - A red horizontal segment is drawn from NPV Low to Baseline NPV (if Baseline > NPV Low).
      - A green horizontal segment is drawn from Baseline NPV to NPV High (if NPV High > Baseline).
      - A black marker indicates the Baseline NPV.

    Returns:
      A Plotly figure object.
    """
    # Create a copy to avoid modifying the original DataFrame.
    df = bounds_df.copy()

    # Calculate the maximum absolute deviation from baseline for each parameter.

    fig = go.Figure()

    # Loop through each parameter row and add segments and marker.
    for _, row in df.iterrows():
        param = row["Parameter"]
        baseline = row["Baseline"]
        low = row["Low"]
        high = row["High"]

        # if row["Impact"] == "negative":
        #     low, high = high, low

        # Red segment: from NPV Low to Baseline (if baseline > low).
        if baseline > low:
            fig.add_trace(
                go.Scatter(
                    x=[low, baseline],
                    y=[param, param],
                    mode="lines",
                    line=dict(color="red", width=10),
                    showlegend=False,
                )
            )

        # Green segment: from Baseline to NPV High (if high > baseline).
        if high > baseline:
            fig.add_trace(
                go.Scatter(
                    x=[baseline, high],
                    y=[param, param],
                    mode="lines",
                    line=dict(color="green", width=10),
                    showlegend=False,
                )
            )

        # Black marker for the baseline.
        fig.add_trace(
            go.Scatter(
                x=[baseline],
                y=[param],
                mode="markers",
                marker=dict(color="black", size=12),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title,
        yaxis=dict(title="Parameter", automargin=True),
        margin=dict(l=150, r=50, t=80, b=50),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)
