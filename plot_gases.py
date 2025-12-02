# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 11:34:41 2025

@author: barrio_o
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import plotly.graph_objects as go


def trace_gas_visualizer_temp_second(
    merged_temp_gases,
    gases=['CO wet', "SO2 wet"],
    temp_cols=["Target Temp (C)", "Measured Temp (C)"],
    start_time=None,
    end_time=None,
    figsize=(12, 7),
    normalize=False
):
    """
    Visualizes MIRO trace gas data with temperature on a secondary y-axis.
    
    Parameters
    ----------
    merged_temp_gases : pd.DataFrame
        DataFrame containing timestamp ('t-stamp'), gas data, and temperature columns.
    gases : list of str
        Gas columns to plot (left y-axis).
    temp_cols : list of str
        Temperature columns to plot (right y-axis).
    start_time, end_time : str or datetime.time, optional
        Filter time range (e.g. '13:00' or datetime.time(13,0)).
    figsize : tuple
        Figure size in inches.
    normalize : bool
        If True, normalize gas concentrations to 0-1 range.
    
    Returns
    -------
    fig, (ax1, ax2) : matplotlib Figure and Axes objects
    """

    # Copy and preprocess
    df = merged_temp_gases.copy()
    df['t-stamp'] = pd.to_datetime(df['t-stamp'], errors='coerce')
    df = df.dropna(subset=['t-stamp'])

# --- Time filtering ---
    if start_time and end_time:
       if isinstance(start_time, str):
        start_time = pd.to_datetime(start_time).time()
    if isinstance(end_time, str):
        end_time = pd.to_datetime(end_time).time()
    df = df[(df['t-stamp'].dt.time >= start_time) & (df['t-stamp'].dt.time <= end_time)]

# --- Compute relative time in seconds (starting at 0, after filtering) --- ##check this step out
    if not df.empty:
       start_time_abs = df['t-stamp'].iloc[0]
       df['time_s'] = (df['t-stamp'] - start_time_abs).dt.total_seconds()
    else:
       df['time_s'] = np.nan


    # Convert gas concentrations to ppb
    df[gases] = df[gases] * 1e9

    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()  # Secondary y-axis for temperature

    # --- Plot gases ---
    gas_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
    for i, gas in enumerate(gases):
        color = gas_colors[i % len(gas_colors)]
        y_data = df[gas].values

        # Normalize if needed
        if normalize:
            y_min, y_max = y_data.min(), y_data.max()
            if y_max > y_min:
                y_data = (y_data - y_min) / (y_max - y_min)

        # --- NEW: use relative time (seconds) for x-axis ---
        ax1.plot(df['time_s'], y_data, color=color, linewidth=2, label=gas, alpha=0.8)

    # --- Plot temperatures (secondary axis) ---
    temp_colors = ['#FF0000', '#FFA500']  # Red and orange for contrast
    for i, tcol in enumerate(temp_cols):
        if tcol in df.columns:
            color = temp_colors[i % len(temp_colors)]
            # --- NEW: also plot temp vs relative time ---
            ax2.plot(df['time_s'], df[tcol], color=color, linewidth=6,
                     linestyle='--', label=tcol, alpha=0.9)

    # --- NEW: x-axis label for relative time ---
    #ax1.set_xlabel('Relative Time [s]', fontsize=20, fontweight='bold')

    # Y-axis labels
    ax1.set_ylabel('Mixing ratio [ppb]' if normalize else 'Concentration [ppb]',
                   fontsize=30, fontweight='bold', color='black')
    ax2.set_ylabel('Temperature [°C]', fontsize=30, fontweight='bold', color='black')

    # Style & grid
    ax1.grid(False, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.tick_params(axis='both', which='major', labelsize=30)
    ax2.tick_params(axis='y', labelsize=30, colors='black')

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=18, framealpha=0.9, ncol=2)

    # Borders
    for spine in ax1.spines.values():
        spine.set_linewidth(1.2)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.2)

    # --- REMOVED absolute time formatting (not needed for relative seconds) ---
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    # ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    # fig.autofmt_xdate(rotation=45)

    ax1.grid(False)
    ax2.grid(False)

    plt.tight_layout()
    return fig, (ax1, ax2)

def trace_gas_visualizer_temp_second_soft(
    merged_temp_gases,
    gases=['CO wet', "SO2 wet"],
    temp_cols=["Target Temp (C)", "Measured Temp (C)"],
    start_time=None,
    end_time=None,
    figsize=(12, 7),
    normalize=False,
    gas_smoothing="raw",          # "raw", "median", "rolling_median"
    rolling_window=30             # number of samples for rolling median
):
    """
    Visualizes MIRO trace gas data with temperature on a secondary y-axis.

    Parameters
    ----------
    merged_temp_gases : pd.DataFrame
        Must contain 't-stamp', gas columns, and temperature columns.
    gases : list of str
        Gas columns to plot (left y-axis).
    temp_cols : list of str
        Temperature columns to plot (right y-axis).
    start_time, end_time : str or datetime.time
        Optional time filtering (HH:MM format or datetime.time).
    figsize : tuple
        Figure size in inches.
    normalize : bool
        If True, normalize gas concentrations to 0–1.
    gas_smoothing : str
        "raw"            – plot raw gas data
        "median"         – plot one horizontal median line per gas
        "rolling_median" – apply rolling median smoothing
    rolling_window : int
        Window size for rolling median in number of samples.
    """

    # Copy and preprocess
    df = merged_temp_gases.copy()
    df['t-stamp'] = pd.to_datetime(df['t-stamp'], errors='coerce')
    df = df.dropna(subset=['t-stamp'])

    # -------------------------------------------------
    # TIME FILTERING
    # -------------------------------------------------
    if start_time and end_time:
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time).time()

        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time).time()

        df = df[
            (df['t-stamp'].dt.time >= start_time) &
            (df['t-stamp'].dt.time <= end_time)
        ]

    # -------------------------------------------------
    # RELATIVE TIME IN SECONDS
    # -------------------------------------------------
    if not df.empty:
        t0 = df['t-stamp'].iloc[0]
        df['time_s'] = (df['t-stamp'] - t0).dt.total_seconds()
    else:
        df['time_s'] = np.nan

    # -------------------------------------------------
    # CONVERT GASES TO PPB SAFELY
    # -------------------------------------------------
    for g in gases:
        if g in df.columns:
            df.loc[:, g] = df[g] * 1e9

    # -------------------------------------------------
    # FIGURE AND AXES
    # -------------------------------------------------
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    gas_colors  = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
    temp_colors = ['#FF0000', '#FFA500']

    # -------------------------------------------------
    # PLOT GAS DATA (with smoothing options)
    # -------------------------------------------------
    for i, gas in enumerate(gases):

        if gas not in df.columns:
            continue

        color = gas_colors[i % len(gas_colors)]

        # ----------- MEDIAN MODE (single line) -----------
        if gas_smoothing == "median":
            med = df[gas].median()
            ax1.hlines(
                med,
                xmin=df['time_s'].min(),
                xmax=df['time_s'].max(),
                color=color,
                linewidth=2.5,
                label=f"{gas} (median)"
            )
            continue

        # ----------- ROLLING MEDIAN MODE -----------
        if gas_smoothing == "rolling_median":
            smoothed = df[gas].rolling(window=rolling_window, center=True).median()

            if normalize:
                y_min, y_max = smoothed.min(), smoothed.max()
                if y_max > y_min:
                    smoothed = (smoothed - y_min) / (y_max - y_min)

            ax1.plot(
                df['time_s'], smoothed,
                color=color, linewidth=2.5,
                label=f"{gas}"
            )
            continue

        # ----------- RAW MODE -----------
        y_data = df[gas].values
        if normalize:
            y_min, y_max = y_data.min(), y_data.max()
            if y_max > y_min:
                y_data = (y_data - y_min) / (y_max - y_min)

        ax1.plot(
            df['time_s'], y_data,
            color=color, linewidth=2,
            label=gas, alpha=0.8
        )

    # -------------------------------------------------
    # TEMPERATURE (ALWAYS RAW)
    # -------------------------------------------------
    for i, tcol in enumerate(temp_cols):
        if tcol not in df.columns:
            continue

        color = temp_colors[i % len(temp_colors)]
        ax2.plot(
            df['time_s'], df[tcol],
            color=color,
            linewidth=5,
            linestyle='--',
            alpha=0.9,
            label=tcol
        )

    # -------------------------------------------------
    # LABELS, LEGEND, STYLE
    # -------------------------------------------------
    ax1.set_ylabel('Mixing ratio [ppb]' if normalize else 'Concentration [ppb]',
                   fontsize=28, fontweight='bold')
    ax2.set_ylabel('Temperature [°C]',
                   fontsize=28, fontweight='bold')

    ax1.tick_params(axis='both', labelsize=26)
    ax2.tick_params(axis='y', labelsize=26)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        fontsize=12,
        framealpha=0.9,
        ncol=1
    )

    ax1.grid(False)
    ax2.grid(False)

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_temperature_profile(
    merged_temp_gases,
    temp_cols=["Target Temp (C)", "Measured Temp (C)"],
    start_time=None,
    end_time=None,
    figsize=(12, 7)
):
    """
    Plot ONLY the temperature profile vs. relative time (seconds).

    Parameters
    ----------
    merged_temp_gases : pd.DataFrame
        Must contain 't-stamp' + temperature columns.
    temp_cols : list of str
        Temperature columns to plot.
    start_time, end_time : str or datetime.time, optional
        Time-of-day filter.
    figsize : tuple
        Size of the figure.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """

    df = merged_temp_gases.copy()
    df['t-stamp'] = pd.to_datetime(df['t-stamp'], errors='coerce')
    df = df.dropna(subset=['t-stamp'])

    # -------- Time filtering --------
    if start_time and end_time:
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time).time()
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time).time()

        df = df[(df['t-stamp'].dt.time >= start_time) &
                (df['t-stamp'].dt.time <= end_time)]

    # -------- Compute relative time (s) --------
    if not df.empty:
        start_time_abs = df['t-stamp'].iloc[0]
        df['time_s'] = (df['t-stamp'] - start_time_abs).dt.total_seconds()
    else:
        df['time_s'] = np.nan

    # -------- Plot --------
    fig, ax = plt.subplots(figsize=figsize)

    temp_colors = ['#FF0000', '#FFA500']

    temp_colors = ['#FF0000', '#FFA500']
    linestyles = ['-', '--']   # first temp = solid, second = dashed

    for i, tcol in enumerate(temp_cols):
        if tcol in df.columns:
            ax.plot(
            df['time_s'],
            df[tcol],
            color=temp_colors[i % len(temp_colors)],
            linewidth=4,
            linestyle=linestyles[i % len(linestyles)],
            label=tcol
        )


    # Labels & styling
    ax.set_xlabel("Relative Time [s]", fontsize=28, fontweight="bold")
    ax.set_ylabel("Temperature [°C]", fontsize=28, fontweight="bold")
    ax.tick_params(axis='both', labelsize=24)

    ax.legend(fontsize=24)
    ax.grid(False)

    plt.tight_layout()
    return fig, ax

