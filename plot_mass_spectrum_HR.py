# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 11:42:21 2025

@author: barrio_o
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def plot_multiple_spectra(df, times=None, mz_min=100, mz_max=150, figsize=(14,30),
                           window_seconds=None, formulas_list=None, top_n=5, subplot_layout=(4,1)):
    """
    Plot mass spectra for multiple time points or intervals.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a 'time' column and numeric columns representing m/z values.
    times : list
        List of times (floats) or intervals (tuples (start, end)).
    mz_min, mz_max : float
        m/z plotting range.
    figsize : tuple
        Figure size.
    window_seconds : float or None
        Window half-width for selecting rows when times contains single times.
    formulas_list : list of lists or None
        Optional labels for top peaks; must match len(times) and each inner list length top_n.
    top_n : int
        Number of top peaks to highlight.
    subplot_layout : tuple or None
        (nrows, ncols) or None to auto-arrange.
    """
    
    # Defensive copy of df
    df = df.copy()

    # Default time selection: split total time into 4 parts. Not necessary?
    if times is None:
        if "time" not in df.columns:
            raise ValueError("The dataframe must contain a 'time' column to automatically select times.")
        n_default = 4
        unique_times = np.sort(df["time"].unique())
        if len(unique_times) < n_default:
            times = unique_times.tolist()
        else:
            times = np.linspace(unique_times.min(), unique_times.max(), n_default)
        print(f"[INFO] 'times' not provided → automatically using {len(times)} evenly spaced points across time range.")

    # Ensure time is numeric
    if not np.issubdtype(df["time"].dtype, np.number):
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        if df["time"].isna().all():
            raise ValueError("'time' column could not be converted to numeric values.")

    # Select hight resolution values to plot 
    excluded_keywords = {"target temp", "measured temp", "absolute time", "m/q", "time"}
    
    spectrum_cols = []
    spectrum_mzs = []
    
    for col in df.columns:
        col_str = str(col).strip()
        if any(keyword in col_str.lower() for keyword in excluded_keywords):
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            continue
        try:
            mz_value = float(col_str)
            spectrum_cols.append(col)
            spectrum_mzs.append(mz_value)
        except ValueError:
            continue

    if not spectrum_cols:
        raise ValueError("No numeric m/z columns found. Check your dataframe column names.")

    # Sort by m/z value. not necessary?
    sorted_indices = np.argsort(spectrum_mzs)
    spectrum_cols = [spectrum_cols[i] for i in sorted_indices]
    spectrum_mzs = np.array([spectrum_mzs[i] for i in sorted_indices], dtype=float)

    #  Subplot layout 
    n_plots = len(times)
    if subplot_layout is None:
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

    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else np.array([axes], dtype=object)

    # --- Loop and plot ---
    for idx, time_spec in enumerate(times):
        ax = axes_flat[idx]
        formulas = formulas_list[idx] if formulas_list is not None else None

        if isinstance(time_spec, (tuple, list)) and len(time_spec) == 2:
            start_time, end_time = time_spec
            mask = (df["time"] >= float(start_time)) & (df["time"] <= float(end_time))
            df_sel = df.loc[mask]
            time_range = (float(start_time), float(end_time))
            display_time = (time_range[0] + time_range[1]) / 2.0
        else:
            target_time = float(time_spec)
            if window_seconds is None:
                idx_min = (df["time"] - target_time).abs().idxmin()
                df_sel = df.loc[[idx_min]]
            else:
                mask = (df["time"] >= target_time - window_seconds) & (df["time"] <= target_time + window_seconds)
                df_sel = df.loc[mask]
            time_range = None
            display_time = target_time

        if df_sel.empty:
            ax.text(0.5, 0.5, 'No data in range', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(mz_min, mz_max)
            #space for the formulas names
            ax.margins(y=0.25)
            ax.set_ylim(0, 1)
            continue

        intensities = df_sel[spectrum_cols].mean(axis=0).to_numpy(dtype=float)
        masses = spectrum_mzs.copy()
        mask_mzrange = (masses >= mz_min) & (masses <= mz_max)
        masses_plot = masses[mask_mzrange]
        intensities_plot = intensities[mask_mzrange]

        if masses_plot.size == 0:
            ax.text(0.5, 0.5, 'No m/z in range', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(mz_min, mz_max)
            #space for the formulas names
            ax.margins(y=0.25)
            continue

        if intensities_plot.size >= top_n:
            top_idx = np.argsort(intensities_plot)[-top_n:][::-1]
            top_masses = masses_plot[top_idx]
            top_intensities = intensities_plot[top_idx]
        else:
            top_masses = masses_plot
            top_intensities = intensities_plot

        if time_range:
            print(f"\nTop {len(top_masses)} peaks in interval [{time_range[0]:.2f}, {time_range[1]:.2f}] s:")
        else:
            print(f"\nTop {len(top_masses)} peaks at {display_time:.2f} s:")
        print("-" * 50)
        for i, (mval, ival) in enumerate(zip(top_masses, top_intensities), start=1):
            formula_label = f" ({formulas[i-1]})" if formulas is not None and i-1 < len(formulas) else ""
            print(f"{i}. m/z: {mval:.4f}{formula_label} | Intensity: {ival:.2e}")

        ax.vlines(masses_plot, 0, intensities_plot, linewidth=4, color='#382eaa', alpha=0.8)
        
        for i, (mval, ival) in enumerate(zip(top_masses, top_intensities)):
            label = formulas[i] if (formulas is not None and i < len(formulas)) else f"{mval:.1f}"
            ax.text(mval, ival * 1.1, label, ha='center', va='bottom', fontsize=35, 
                    rotation=90, fontweight='normal')
            ax.vlines(mval, 0, ival, color='#382eaa', linewidth=4, alpha=1.0)

        ax.set_xlim(mz_min, mz_max)
        #space for the formulas names
        ax.margins(y=0.25)
        if np.max(intensities_plot) > 0:
            ax.set_ylim(0, np.max(intensities_plot) * 1.15)
        else:
            ax.set_ylim(0, 1)
            
            
      # Define fixed times for each MS
        #ms_times = [450, 600, 750, 930]

       # Loop through each MS (assuming idx is your loop index)
        ms_label = f"MS{idx + 1}"

      # Add text in the top-right corner inside the plot
        ax.text(
       0.98, 1.0, ms_label,
    transform=ax.transAxes,
    fontsize=35,
    fontweight='bold',
    verticalalignment='top',
    horizontalalignment='right'
)

        # Only set y-axis label for all subplots
        ax.set_ylabel("Intensity [a.u.]", fontsize=35, fontweight='bold')
        ax.grid(False)

        # Increase tick label size for better readability
        ax.tick_params(axis='both', which='major', labelsize=35)
        # --- Remove top & right axis lines for cleaner look and extra label space ---
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused axes
    for j in range(n_plots, axes_flat.size):
        axes_flat[j].set_visible(False)

    # Set only **one** x-axis label at the bottom
    for ax in axes_flat[:-1]:
        ax.set_xlabel("")  # remove x-axis labels from all but last
    axes_flat[-1].set_xlabel("m/z", fontsize=35, fontweight='bold')
    
# Set y-axis label only on the second plot (MS2)
    for i, ax in enumerate(axes_flat[:n_plots]):
        if i == 1:  # Second plot (index 1 = MS2)
            ax.set_ylabel("Intensity", fontsize=35, fontweight='bold')
        else:
            ax.set_ylabel("")
    
    plt.tight_layout()
    return fig, axes

def plot_spectra_custom_ranges(df, times=None, mz_ranges=None, figsize=(10,15),
                                window_seconds=None, formulas_list=None, top_n=3, 
                                subplot_layout=(4,1)):
    """
    Plot mass spectra with custom m/z ranges for each subplot.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a 'time' column and numeric columns representing m/z values.
    times : list
        List of times (floats) or intervals (tuples (start, end)).
    mz_ranges : list of tuples
        List of (mz_min, mz_max) tuples, one for each subplot.
        Example: [(40, 250), (100, 200), (150, 300), (50, 150)]
    figsize : tuple
        Figure size.
    window_seconds : float or None
        Window half-width for selecting rows when times contains single times.
    formulas_list : list of lists or None
        Optional labels for top peaks.
    top_n : int
        Number of top peaks to highlight.
    subplot_layout : tuple or None
        (nrows, ncols) or None to auto-arrange.
    """
    
    # Defensive copy
    df = df.copy()
    
    # Default time selection
    if times is None:
        if "time" not in df.columns:
            raise ValueError("The dataframe must contain a 'time' column.")
        n_default = 4
        unique_times = np.sort(df["time"].unique())
        if len(unique_times) < n_default:
            times = unique_times.tolist()
        else:
            times = np.linspace(unique_times.min(), unique_times.max(), n_default)
        print(f"[INFO] 'times' not provided → using {len(times)} evenly spaced points.")
    
    # Default mz_ranges if not provided
    if mz_ranges is None:
        mz_ranges = [(40, 250)] * len(times)
        print(f"[INFO] 'mz_ranges' not provided → using default range (40, 250) for all subplots.")
    
    # Validate mz_ranges length
    if len(mz_ranges) != len(times):
        raise ValueError(f"mz_ranges must have {len(times)} elements (one per time point).")
    
    # Ensure time is numeric
    if not np.issubdtype(df["time"].dtype, np.number):
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        if df["time"].isna().all():
            raise ValueError("'time' column could not be converted to numeric values.")
    
    # Find spectral columns
    excluded_keywords = {"target temp", "measured temp", "absolute time", "m/q", "time"}
    spectrum_cols = []
    spectrum_mzs = []
    
    for col in df.columns:
        col_str = str(col).strip()
        if any(keyword in col_str.lower() for keyword in excluded_keywords):
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            continue
        try:
            mz_value = float(col_str)
            spectrum_cols.append(col)
            spectrum_mzs.append(mz_value)
        except ValueError:
            continue
    
    if not spectrum_cols:
        raise ValueError("No numeric m/z columns found.")
    
    # Sort by m/z
    sorted_indices = np.argsort(spectrum_mzs)
    spectrum_cols = [spectrum_cols[i] for i in sorted_indices]
    spectrum_mzs = np.array([spectrum_mzs[i] for i in sorted_indices], dtype=float)
    
    # Subplot layout
    n_plots = len(times)
    if subplot_layout is None:
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
    
    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else np.array([axes], dtype=object)

    
    # Loop and plot
    for idx, (time_spec, mz_range) in enumerate(zip(times, mz_ranges)):
        ax = axes_flat[idx]
        mz_min, mz_max = mz_range
        formulas = formulas_list[idx] if formulas_list is not None else None
        
        # Select rows based on time
        if isinstance(time_spec, (tuple, list)) and len(time_spec) == 2:
            start_time, end_time = time_spec
            mask = (df["time"] >= float(start_time)) & (df["time"] <= float(end_time))
            df_sel = df.loc[mask]
            time_range = (float(start_time), float(end_time))
            display_time = (time_range[0] + time_range[1]) / 2.0
        else:
            target_time = float(time_spec)
            if window_seconds is None:
                idx_min = (df["time"] - target_time).abs().idxmin()
                df_sel = df.loc[[idx_min]]
            else:
                mask = (df["time"] >= target_time - window_seconds) & (df["time"] <= target_time + window_seconds)
                df_sel = df.loc[mask]
            time_range = None
            display_time = target_time
        
        if df_sel.empty:
            ax.text(0.5, 0.5, 'No data in range', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(mz_min, mz_max)
            ax.set_ylim(0, 1)
            continue
        
        # Compute mean intensities
        intensities = df_sel[spectrum_cols].mean(axis=0).to_numpy(dtype=float)
        masses = spectrum_mzs.copy()
        
        # Filter by m/z range
        mask_mzrange = (masses >= mz_min) & (masses <= mz_max)
        masses_plot = masses[mask_mzrange]
        intensities_plot = intensities[mask_mzrange]
        
        if masses_plot.size == 0:
            ax.text(0.5, 0.5, 'No m/z in range', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(mz_min, mz_max)
            # --- Remove top & right axis lines for cleaner look and extra label space ---
            continue
        
        # Identify top peaks
        if intensities_plot.size >= top_n:
            top_idx = np.argsort(intensities_plot)[-top_n:][::-1]
            top_masses = masses_plot[top_idx]
            top_intensities = intensities_plot[top_idx]
        else:
            top_masses = masses_plot
            top_intensities = intensities_plot
        
        # Print top peaks
        print(f"\nMS{idx + 1} - Top {len(top_masses)} peaks:")
        print("-" * 50)
        for i, (mval, ival) in enumerate(zip(top_masses, top_intensities), start=1):
            formula_label = f" ({formulas[i-1]})" if formulas is not None and i-1 < len(formulas) else ""
            print(f"{i}. m/z: {mval:.4f}{formula_label} | Intensity: {ival:.2e}")
        
        # Plot all masses
        ax.vlines(masses_plot, 0, intensities_plot, linewidth=1.2, color='#2E5EAA', alpha=0.8)
        
        # Highlight top peaks
        for i, (mval, ival) in enumerate(zip(top_masses, top_intensities)):
            label = formulas[i] if (formulas is not None and i < len(formulas)) else f"{mval:.1f}"
            ax.text(mval, ival * 1.05, label, ha='center', va='bottom', fontsize=18, 
                    rotation=90, fontweight='normal')
            ax.vlines(mval, 0, ival, color='#2E5EAA', linewidth=1.5, alpha=1.0)
        
        # Set axis limits
        ax.set_xlim(mz_min, mz_max)
        if np.max(intensities_plot) > 0:
            ax.set_ylim(0, np.max(intensities_plot) * 1.15)
        else:
            ax.set_ylim(0, 1)
        
        # Title and labels
        ms_label = f"MS{idx + 1}"
        ax.set_title(ms_label, fontsize=18, fontweight='bold')
        ax.set_ylabel("Intensity [a.u.]", fontsize=18, fontweight='bold')
        
        # Make tick numbers bigger
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.grid(False)
    
    # Hide unused axes
    for j in range(n_plots, axes_flat.size):
        axes_flat[j].set_visible(False)
    
    # Only bottom subplot has "m/z" x-axis label
    for ax in axes_flat[:-1]:
        ax.set_xlabel("")
    axes_flat[-1].set_xlabel("m/z", fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    return fig, axes


