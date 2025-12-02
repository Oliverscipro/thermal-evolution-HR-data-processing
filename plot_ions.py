# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 11:31:27 2025

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

def plot_peaks_timeseries(df, formulas=None, time_col="time", normalize=False, figsize=(12, 7)):
    """
    Plot time series of selected peaks with dual x-axis (time and temperature).
  
    Bottom x-axis: Relative Time [s]
    Top x-axis: Target Temp (C)
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing m/Q columns, time columns, and Target Temp column.
    formulas : list of str
        List of chemical formulas to plot
    time_col : str
        Name of the time column
    normalize : bool
        Whether to normalize each trace to [0, 1]
    figsize : tuple
        Figure size in inches.
    
    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each peak
    for formula in formulas:
        if formula in df.columns:
            # Get only positive values
            y_values = df[formula].where(df[formula] > 0)
            
            # Normalize if needed (BEFORE plotting)
            if normalize:
                y_min, y_max = y_values.min(), y_values.max()
                if y_max > y_min:  # Avoid division by zero
                    y_values = (y_values - y_min) / (y_max - y_min)
            
            # Now plot the processed values
            ax.plot(df[time_col], y_values, label=formula, linewidth=2.2)
        else:
            print(f"Warning: '{formula}' not found in DataFrame columns.")
    
    # Aesthetics
    ylabel = "Normalized intensity" if normalize else "Key ion intensity"
    ax.set_ylabel(ylabel, fontsize=30, color="black", fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.legend(loc='upper left', fontsize=25, framealpha=0.9, ncol=2)
    ax.grid(False)
    
    return fig, ax

    
    return fig, ax

def plot_peaks_timeseries_temp(df, formulas=None, time_col="time", temp_col="Target Temp (C)", 
                          measured_temp_col="Measured Temp (C)", figsize=(14, 7), dpi=150):
    """
    Plot time series of selected peaks with dual y-axis (intensity and temperature).
  
    Left y-axis: Peak intensities (normalized)
    Right y-axis: Temperature (°C)
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing m/Q columns, time columns, and temperature columns.
    formulas : list of str
        List of column names (formulas) to plot (e.g., ['C5H8O', 'C4H6O2'])
    time_col : str
        Name of the time column to use for x-axis.
    temp_col : str
        Name of the target temperature column.
    measured_temp_col : str
        Name of the measured temperature column.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Resolution of the figure.
    
    Returns
    -------
    fig, ax, ax2 : matplotlib Figure and Axes objects
    """
   

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Define better colors for peaks (avoiding orange which is used for temperature)
    peak_colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
                   '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Normalize peak intensities to 0-1 range for better visualization
    if formulas:
        for idx, formula in enumerate(formulas):
            if formula in df.columns:
                # Normalize each peak individually
                values = df[formula].values
                if values.max() > 0:  # Avoid division by zero
                    normalized = (values - values.min()) / (values.max() - values.min())
                    ax.plot(df[time_col], normalized, 
                           label=f"{formula}", 
                           linewidth=1.5, 
                           alpha=0.85,
                           color=peak_colors[idx % len(peak_colors)])
                else:
                    ax.plot(df[time_col], values, 
                           label=formula, 
                           linewidth=1.5, 
                           alpha=0.85,
                           color=peak_colors[idx % len(peak_colors)])
            else:
                print(f"Warning: '{formula}' not found in DataFrame columns.")
    
    # Styling for left y-axis (peaks)
    ax.set_xlabel("Time [s]", fontsize=13, fontweight='bold')
    ax.set_ylabel("Normalized Intensity", fontsize=13, fontweight='bold', color='black')
    #ax.set_title("Time Series of Peaks with Temperature Profile", fontsize=15, fontweight='bold', pad=15)
    ax.tick_params(axis='y', labelcolor='black', labelsize=11)
    ax.tick_params(axis='x', labelsize=11)
    ax.set_ylim(-0.02, 1.05)  # Set limits for normalized data
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    
    # Legend for peaks - better positioning
    if formulas:
        legend1 = ax.legend(loc='upper left', fontsize=10, framealpha=0.9, 
                           edgecolor='gray', fancybox=True)
    
    # Add secondary y-axis for temperature
    ax2 = ax.twinx()
    
    # Plot both temperatures on the secondary axis with improved styling
    line1 = ax2.plot(df[time_col], df[temp_col], 
                     label='Target Temp', 
                     color='#1E88E5',  # Nice blue
                     linewidth=6, 
                     linestyle='--', 
                     alpha=0.8,
                     zorder=100)  # Bring to front
    
    line2 = ax2.plot(df[time_col], df[measured_temp_col], 
                     label='Measured Temp', 
                     color='#FF6F00',  # Nice orange
                     linewidth=6, 
                     linestyle='-', 
                     alpha=0.8,
                     zorder=101)  # Bring to front
    
    # Styling for right y-axis (temperature)
    ax2.set_ylabel("Temperature [°C]", fontsize=13, fontweight='bold', color='#D84315')
    ax2.tick_params(axis='y', labelcolor='#D84315', labelsize=11)
    
    # Set temperature y-axis limits based on data with better margins
    temp_min = min(df[temp_col].min(), df[measured_temp_col].min())
    temp_max = max(df[temp_col].max(), df[measured_temp_col].max())
    margin = (temp_max - temp_min) * 0.08  # 8% margin
    ax2.set_ylim(temp_min - margin, temp_max + margin)
    
    # Legend for temperatures - better positioning and styling
    legend2 = ax2.legend(loc='upper right', fontsize=10, framealpha=0.9,
                        edgecolor='gray', fancybox=True)
    
    # Add subtle background color distinction
    ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    return fig, ax, ax2