# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 10:52:28 2025

@author: barrio_o
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates


import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_heatmap(cleaned_df, mz_min=100, mz_max=185, vmin=None, vmax=None,
    figsize=(16, 12), ytick_step=10, dpi=300, cmap="viridis", highlight_color="black",
    highlight_linewidth=4, use_log=True): 
    """
    Plot UMR values from post processing data 2024 campaign with MS region labels.
    
    Parameters
    ----------
    cleaned_df : pd.DataFrame
        Dataframe with 'time' or 'Absolute time', 'Measured Temp (C)', Target Temp (C), and UMR columns.
    mz_min, mz_max : float
        Mass range to display (y-axis).
    vmin, vmax : float
        Color intensity range for the heatmap. If None, computed as mean ± 3σ.
    figsize : tuple
        Figure size in inches.
    ytick_step : int
        Step between m/z tick marks.
    dpi : int
        Resolution for the figure.
    cmap : str
        Colormap to use ('inferno', 'viridis', 'plasma', 'magma', 'turbo', 'Reds').
    highlight_color : str
        Color for MS label boxes.
    highlight_linewidth : float
        Not used in this version (kept for compatibility).
    use_log : bool
        Whether to log-scale intensity values (default True).
    
    Returns
    -------
    fig, ax : Matplotlib Figure and Axes objects
    """

    df = cleaned_df.copy()
    
    # Select time column
    if "time" in df.columns:
        time_col = "time"
    elif "Absolute time" in df.columns:
        time_col = "Absolute time"
    else:
        raise ValueError("No time column found (expected 'time' or 'Absolute time').")
    
    # Identify m/Q columns
    mq_cols = [col for col in df.columns if col.startswith("m/Q")]
    if not mq_cols:
        raise ValueError("No 'm/Q' columns found in dataframe.")
    
    # Extract m/z values from column names and sort 
    mz_data = []
    for col in mq_cols:
        try:
            mz = float(col.split()[1])
            mz_data.append((mz, col))
        except (IndexError, ValueError):
            continue
    
    # Sort by m/z value
    mz_data.sort(key=lambda x: x[0])
    mz_values = [x[0] for x in mz_data]
    sorted_cols = [x[1] for x in mz_data]
    
    # Create intensity matrix 
    heatmap_data = df.set_index(time_col)[sorted_cols].T
    heatmap_data.index = mz_values

    # --- Data transformation ---
    if use_log:
        heatmap_data = np.log10(heatmap_data.clip(lower=1e-12))
    else:
        # Remove negative values for linear scale
        heatmap_data = heatmap_data.clip(lower=0)

    # Auto color scaling , three times standart desviations
    finite_vals = np.ravel(heatmap_data[np.isfinite(heatmap_data)])
    mean_val = np.mean(finite_vals)
    std_val = np.std(finite_vals)

    if use_log:
        if vmin is None:
            vmin = mean_val - 3 * std_val
        if vmax is None:
            vmax = mean_val + 3 * std_val
    else:
        if vmin is None:
            vmin = 0  # force start at zero intensity
        if vmax is None:
            vmax = mean_val + 3 * std_val

    #  Plot setup, aesthetic
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    time_vals = df[time_col].values
    time_min, time_max = time_vals.min(), time_vals.max()
    mz_all = np.array(mz_values)
    mz_extent_min = mz_all.min()
    mz_extent_max = mz_all.max()
    
    im = ax.imshow(
        heatmap_data.values,
        aspect='auto',
        cmap=cmap,
        origin='lower', 
        extent=[time_min, time_max, mz_extent_min, mz_extent_max],
        vmin=vmin,
        vmax=vmax,
        interpolation=None,
        rasterized=True
    )
    
    #  Axes settings 
    ax.set_ylim(mz_min, mz_max)
    y_ticks = np.arange(int(np.ceil(mz_min/ytick_step)*ytick_step), 
                        int(mz_max) + 1, ytick_step)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks, fontsize=30)
    
    ax.set_xlim(time_min, time_max)
    ax.tick_params(axis='x', labelsize=30)
    
    #  Label MS regions
    square_height = (mz_max - mz_min) * 0.030
    square_width = (time_max - time_min) * 0.100
    label_y = mz_max - square_height * 1.2
    label_positions = [25, 620, 780, 985]
    ms_labels = ["MS1", "MS2", "MS3", "MS4"]

    for t_label, label in zip(label_positions, ms_labels):
        rect = patches.Rectangle(
            (t_label - square_width / 2, label_y - square_height / 2),
            square_width, square_height,
            facecolor=highlight_color, edgecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            t_label, label_y,
            label, color="white",
            ha="center", va="center",
            fontsize=25, fontweight="bold"
        )
    
    #  Colorbar 
    im.set_clim(vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, location="right", pad=0.02, fraction=0.046)
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label("Intensity (log₁₀)" if use_log else "Intensity",
                   fontsize=30, fontweight="bold")
    
    # Labels and layout 
    ax.set_xlabel("Time [s]", fontsize=30, fontweight="bold")
    ax.set_ylabel("UMR (m/Q)", fontsize=30, fontweight="bold")
    plt.tight_layout()
    
    return fig, ax



########################################### if I want to include temperature ######################



def plot_heatmap_temp(
    cleaned_df,
    mz_min=42, mz_max=399,
    vmin=None, vmax=None,
    figsize=(12, 16),
    ytick_step=10,
    dpi=200,
    cmap="viridis",
    highlight_color="black",
    highlight_linewidth=3,
    use_log=True,
    x_tick_step=100,
):
    """
    Plot UMR heatmap with temperature.

    This function plots a combined figure showing the UMR (Unit Mass Resolution) heatmap 
    as a function of time and m/Q, together with the oven temperature evolution 
    (measured and target). It supports highlighting specific time intervals 
    corresponding to measurement stages (MS1, MS2, etc.), and automatically 
    scales the color range using mean ± 3σ. Optionally applies log10 scaling.

    Parameters
    ----------
    cleaned_df : pandas.DataFrame
        DataFrame containing VOCUS data, including:
        - 'time' or 'Absolute time'
        - 'Measured Temp (C)' and 'Target Temp (C)'
        - UMR signal columns starting with 'm/Q'
    mz_min, mz_max : float
        Minimum and maximum m/z to show.
    vmin, vmax : float
        Color scale limits. If None, computed as mean ± 3σ.
    figsize : tuple
        Figure size (in inches).
    ytick_step : int
        Spacing between m/z tick labels.
    dpi : int
        Plot resolution.
    cmap : str
        Colormap for heatmap.
    highlight_color : str
        Color for highlight markers.
    highlight_linewidth : float
        Border width for highlighted areas.
    use_log : bool
        Whether to log-scale intensity values.
    x_tick_step : int
        Time tick spacing in seconds.

    Returns
    -------
    fig, ax_temp, ax_heatmap : matplotlib objects
        Figure and axes for further customization.
    """

    df = cleaned_df.copy()

    # --- Detect time column ---
    if "time" in df.columns:
        time_col = "time"
    elif "Absolute time" in df.columns:
        time_col = "Absolute time"
    else:
        raise ValueError("No time column found (expected 'time' or 'Absolute time').")

    if "Target Temp (C)" not in df.columns:
        raise ValueError("Missing 'Target Temp (C)' column to define MS regions.")

    # --- Identify m/Q columns ---
    mq_cols = [c for c in df.columns if c.startswith("m/Q")]
    if not mq_cols:
        raise ValueError("No 'm/Q' columns found in DataFrame.")

    # --- Extract m/z values ---
    mz_data = []
    for col in mq_cols:
        try:
            mz = float(col.split()[1])
            mz_data.append((mz, col))
        except Exception:
            continue
    mz_data.sort(key=lambda x: x[0])
    mz_values = [x[0] for x in mz_data]
    sorted_cols = [x[1] for x in mz_data]

    # --- Prepare heatmap data ---
    heatmap_data = df.set_index(time_col)[sorted_cols].T
    heatmap_data.index = mz_values

    if use_log:
        heatmap_data = np.log10(heatmap_data.clip(lower=1e-12))

    # --- Auto color scaling (mean ± 3σ) ---
    if vmin is None or vmax is None:
        finite_vals = np.ravel(heatmap_data[np.isfinite(heatmap_data)])
        mean_val = np.mean(finite_vals)
        std_val = np.std(finite_vals)
        vmin = mean_val - 3 * std_val
        vmax = mean_val + 3 * std_val

    # --- Time range ---
    time_vals = df[time_col].values
    time_min, time_max = time_vals.min(), time_vals.max()

    # --- Create figure ---
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.02)
    ax_temp = fig.add_subplot(gs[0])
    ax_heatmap = fig.add_subplot(gs[1], sharex=ax_temp)

    # --- TEMPERATURE PLOT ---
    ax_temp.plot(df[time_col], df["Target Temp (C)"], color="steelblue", lw=4, label="Target Temp")
    if "Measured Temp (C)" in df.columns:
        ax_temp.plot(df[time_col], df["Measured Temp (C)"], color="darkorange", lw=4, label="Measured Temp")

    ax_temp.set_ylabel("Temp [°C]", fontsize=11, fontweight="bold")
    ax_temp.legend(fontsize=9, loc="upper left")
    ax_temp.grid(False)
    ax_temp.tick_params(axis="x", labelbottom=False)
    ax_temp.set_xlim(time_min, time_max)

    # --- HEATMAP ---
    im = ax_heatmap.imshow(
        heatmap_data.values,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        extent=[time_min, time_max, min(mz_values), max(mz_values)],
        vmin=vmin, vmax=vmax,
    )

    ax_heatmap.set_ylim(mz_min, mz_max)
    ax_heatmap.set_yticks(np.arange(np.ceil(mz_min / ytick_step) * ytick_step, mz_max + 1, ytick_step))
    ax_heatmap.set_ylabel("UMR (m/Q)", fontsize=12, fontweight="bold")
    ax_heatmap.set_xlabel("Time [s]", fontsize=12, fontweight="bold")
    
    # --- LABEL MS REGIONS WITH SMALL SQUARES (on heatmap only) ---
    square_height = (mz_max - mz_min) * 0.020
    square_width = (time_max - time_min) * 0.100
    label_y = mz_max - square_height 

    # Fixed positions for MS labels
    label_positions = [400, 750, 950, 1150]
    ms_labels = ["MS1", "MS2", "MS3", "MS4"]

    for t_label, label in zip(label_positions, ms_labels):
        # Draw square
        rect = patches.Rectangle(
            (t_label - square_width / 2, label_y - square_height / 2),
            square_width, square_height,
            facecolor=highlight_color, edgecolor="none"
        )
        ax_heatmap.add_patch(rect)

        # Label inside
        ax_heatmap.text(
            t_label, label_y,
            label, color="white",
            ha="center", va="center",
            fontsize=20, fontweight="bold"
        )

    # --- Colorbar ---
    cbar = fig.colorbar(im, ax=ax_heatmap, location="right", pad=0.02, fraction=0.04)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Intensity (log₁₀)" if use_log else "Intensity", fontsize=10)

    ax_temp.set_title("VOCUS UMR Heatmap", fontsize=14, fontweight="bold", pad=10)

    plt.show()
    return fig, ax_temp, ax_heatmap

