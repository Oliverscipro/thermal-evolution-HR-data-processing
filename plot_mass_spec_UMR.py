# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 19:07:44 2025

@author: barrio_o
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_mass_spectrum_formulas_VF(df, time=None, mz_min=40, mz_max=250, figsize=(10, 6), window_seconds=None,
                                formulas=None, top_n=3, times=None, formulas_list=None, subplot_layout=(4,1)):
    """
    Plot mass spectrum(s) using 'time' as the time axis.
    Can create either a single plot or multiple subplots at different time points/intervals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'time' and m/Q columns.
    time : float, optional
        Target time (in seconds) for single plot. If None, the first time is used.
        Ignored if 'times' is provided.
    window_seconds : float, optional
        Time window (±seconds) to average spectra around the selected time.
        If None, only one spectrum is plotted. Applied to all subplots.
        Ignored if 'times' contains intervals.
    mz_min, mz_max : float
        Mass range to display on x-axis.
    figsize : tuple
        Figure size in inches.
    formulas : list of str, optional
        Formulas for single plot. Must have exactly top_n elements.
        Ignored if 'times' is provided.
    top_n : int
        Number of peaks to annotate.
    times : list of float or list of tuples, optional
        List of time points OR time intervals for creating subplots.
        - If float: single time point (uses window_seconds if provided)
        - If tuple: (start_time, end_time) interval
        Example: [100, (200, 400), 500] or [(0, 100), (200, 400), (500, 600)]
    formulas_list : list of lists, optional
        List of formula lists, one for each subplot. Must match length of 'times'.
        Each inner list must have exactly top_n elements.
    subplot_layout : tuple of int, optional
        (rows, cols) for subplot layout. If None, automatically determined.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes (or array of Axes for subplots)
    
    Examples
    --------
    # Single plot:
    fig, ax = plot_mass_spectrum_formulas_VF(df, time=10.5, formulas=['H2O+', 'CO2+', 'N2+', 'O2+', 'Ar+'])
    
    # Multiple subplots with intervals:
    fig, axes = plot_mass_spectrum_formulas_VF(
        df, 
        times=[(0, 100), (200, 400), (500, 600)],
        formulas_list=[
            ['H2O+', 'CO2+', 'N2+', 'O2+', 'Ar+'],
            ['CH4+', 'NH3+', 'H2O+', 'CO+', 'N2+'],
            ['O2+', 'CO2+', 'H2O+', 'N2+', 'Ar+']
        ],
        subplot_layout=(1, 3)
    )
    
    # Mixed: some intervals, some single points
    fig, axes = plot_mass_spectrum_formulas_VF(
        df, 
        times=[(0, 100), 250, (400, 500)],
        window_seconds=10,  # Only applies to single time point (250)
        subplot_layout=(1, 3)
    )
    """

    df = df.copy()

    # --- Determine single plot vs subplots mode ---
    if times is not None:
        return _plot_multiple_spectra(df, times, mz_min, mz_max, figsize, window_seconds,
                                     formulas_list, top_n, subplot_layout)
    else:
        return _plot_single_spectrum(df, time, mz_min, mz_max, figsize, window_seconds,
                                    formulas, top_n)


def _plot_single_spectrum(df, time, mz_min, mz_max, figsize, window_seconds, formulas, top_n):
    """Helper function to plot a single mass spectrum."""
    
    # --- Validate formulas ---
    if formulas is not None:
        if len(formulas) != top_n:
            raise ValueError(f"formulas list must have exactly {top_n} elements (one per top peak)")
            
    # --- Validate time column ---
    if "time" not in df.columns:
        raise ValueError("Expected 'time' column not found in dataframe.")
    time_col = "time"

    # --- Identify m/Q columns ---
    mq_cols = [col for col in df.columns if col.startswith("m/Q")]
    if not mq_cols:
        raise ValueError("No 'm/Q' columns found in dataframe.")

    # --- Handle time selection ---
    if time is None:
        selected_time = df[time_col].iloc[0]
        df_sel = df.iloc[[0]]
    else:
        target_time = float(time)
        if window_seconds is None:
            idx = (df[time_col] - target_time).abs().idxmin()
            df_sel = df.loc[[idx]]
        else:
            mask = (df[time_col] >= target_time - window_seconds) & (df[time_col] <= target_time + window_seconds)
            df_sel = df.loc[mask]

        if df_sel.empty:
            raise ValueError("No data found in the specified time window.")
        selected_time = target_time

    # --- Compute spectrum and get top peaks ---
    masses, intensities, top_masses, top_intensities = _compute_spectrum(df_sel, mq_cols, mz_min, mz_max, top_n)
    
    # --- Print top N peaks ---
    print(f"\nTop {len(top_masses)} peaks at {selected_time:.2f} s:")
    print("-" * 50)
    for i, (mass, intensity) in enumerate(zip(top_masses, top_intensities), 1):
        formula_label = f" ({formulas[i-1]})" if formulas is not None else ""
        print(f"{i}. m/z: {mass:.2f}{formula_label} | Intensity: {intensity:.2e}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    _plot_spectrum_on_axis(ax, masses, intensities, top_masses, top_intensities, formulas, 
                          mz_min, mz_max, selected_time, window_seconds, time_range=None)
    
    plt.tight_layout()
    return fig, ax


def _plot_multiple_spectra(df, times, mz_min, mz_max, figsize, window_seconds, formulas_list, top_n, subplot_layout):
    """Helper function to plot multiple mass spectra as subplots."""
    
    n_plots = len(times)
    
    # --- Validate formulas_list ---
    if formulas_list is not None:
        if len(formulas_list) != n_plots:
            raise ValueError(f"formulas_list must have {n_plots} elements (one for each time point)")
        for i, formulas in enumerate(formulas_list):
            if len(formulas) != top_n:
                raise ValueError(f"formulas_list[{i}] must have exactly {top_n} elements")
    
    # --- Validate time column ---
    if "time" not in df.columns:
        raise ValueError("Expected 'time' column not found in dataframe.")
    time_col = "time"

    # --- Identify m/Q columns ---
    mq_cols = [col for col in df.columns if col.startswith("m/Q")]
    if not mq_cols:
        raise ValueError("No 'm/Q' columns found in dataframe.")
    
    # --- Determine subplot layout ---
    if subplot_layout is None:
        # Auto-determine layout
        if n_plots <= 3:
            nrows, ncols = 1, n_plots
        elif n_plots == 4:
            nrows, ncols = 2, 2
        else:
            ncols = 3
            nrows = (n_plots + ncols - 1) // ncols
    else:
        nrows, ncols = subplot_layout
        if nrows * ncols < n_plots:
            raise ValueError(f"subplot_layout {subplot_layout} is too small for {n_plots} plots")
    
    # --- Create subplots ---
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
   
    
    # --- Plot each spectrum ---
    for idx, time_spec in enumerate(times):
        ax = axes[idx]
        formulas = formulas_list[idx] if formulas_list is not None else None
        
        # --- Determine if time_spec is interval or single point ---
        if isinstance(time_spec, (tuple, list)) and len(time_spec) == 2:
            # Interval: (start_time, end_time)
            start_time, end_time = time_spec
            mask = (df[time_col] >= start_time) & (df[time_col] <= end_time)
            df_sel = df.loc[mask]
            time_range = (start_time, end_time)
            display_time = (start_time + end_time) / 2  # For printing
        else:
            # Single time point
            target_time = float(time_spec)
            if window_seconds is None:
                time_idx = (df[time_col] - target_time).abs().idxmin()
                df_sel = df.loc[[time_idx]]
            else:
                mask = (df[time_col] >= target_time - window_seconds) & (df[time_col] <= target_time + window_seconds)
                df_sel = df.loc[mask]
            time_range = None
            display_time = target_time
        
        if df_sel.empty:
            ax.text(0.5, 0.5, f'No data in range', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(mz_min, mz_max)
            continue
        
        # --- Compute spectrum and get top peaks ---
        masses, intensities, top_masses, top_intensities = _compute_spectrum(df_sel, mq_cols, mz_min, mz_max, top_n)
        
        # --- Print top N peaks ---
        if time_range:
            print(f"\nTop {len(top_masses)} peaks in interval [{time_range[0]:.2f}, {time_range[1]:.2f}] s:")
        else:
            print(f"\nTop {len(top_masses)} peaks at {display_time:.2f} s:")
        print("-" * 50)
        for i, (mass, intensity) in enumerate(zip(top_masses, top_intensities), 1):
            formula_label = f" ({formulas[i-1]})" if formulas is not None else ""
            print(f"{i}. m/z: {mass:.2f}{formula_label} | Intensity: {intensity:.2e}")
        
        # --- Plot on this axis ---
        _plot_spectrum_on_axis(ax, masses, intensities, top_masses, top_intensities, formulas,
                              mz_min, mz_max, display_time, window_seconds, time_range)
    
    # --- Hide unused subplots ---
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig, axes


def _compute_spectrum(df_sel, mq_cols, mz_min, mz_max, top_n):
    """Helper function to compute spectrum and identify top peaks."""
    
    # --- Compute mean spectrum ---
    spectrum = df_sel[mq_cols].mean(axis=0)
    
    # --- Extract m/z and intensities ---
    masses = []
    intensities = []
    for col in mq_cols:
        try:
            mass = float(col.split()[1])
            intensity = spectrum[col]
            masses.append(mass)
            intensities.append(intensity)
        except (IndexError, ValueError):
            continue
    
    masses = np.array(masses)
    intensities = np.array(intensities)
    
    # --- Filter by m/z range ---
    mask = (masses >= mz_min) & (masses <= mz_max)
    masses = masses[mask]
    intensities = intensities[mask]
    
    # --- Identify top N peaks ---
    if len(intensities) >= top_n:
        top_indices = np.argsort(intensities)[-top_n:][::-1]
        top_masses = masses[top_indices]
        top_intensities = intensities[top_indices]
    else:
        top_masses = masses
        top_intensities = intensities
    
    return masses, intensities, top_masses, top_intensities


def _plot_spectrum_on_axis(ax, masses, intensities, top_masses, top_intensities, formulas,
                          mz_min, mz_max, selected_time, window_seconds, time_range=None):
    """Helper function to plot spectrum on a given axis."""
    
    # --- Plot all peaks ---
    ax.vlines(masses, 0, intensities, color='black', linewidth=0.8, alpha=0.7)
    
    # --- Highlight and label top N peaks ---
    for i, (mass, intensity) in enumerate(zip(top_masses, top_intensities)):
        # Use formula if provided, otherwise use m/z value
        label = formulas[i] if formulas is not None else f'{mass:.1f}'
        
        ax.text(mass, intensity, label,
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow',
                          alpha=0.7, edgecolor='black', linewidth=0.5))
        ax.vlines(mass, 0, intensity, color='red', linewidth=1.5, alpha=0.8)
    
    # --- Set limits ---
    ax.set_xlim(mz_min, mz_max)
    if np.max(intensities) > 0:
        ax.set_ylim(0, np.max(intensities) * 1.15)
    
    # --- Titles and labels ---
    if time_range:
        title = f"Average Spectrum [{time_range[0]:.0f}-{time_range[1]:.0f}] s"
    elif window_seconds:
        title = f"Average Spectrum (±{window_seconds}s) at {selected_time:.2f} s"
    else:
        title = f"Spectrum at {selected_time:.2f} s"
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("m/z", fontsize=10, fontweight='bold')
    ax.set_ylabel("Intensity [a.u.]", fontsize=10, fontweight='bold')
    
    ax.grid(False)


