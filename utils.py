# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 13:32:46 2025

@author: barrio_o
"""


import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
from glob import glob


def format_formulas_with_subscripts(formulas_nested):
    """
    
    Convert the chemicals formulas into IUPAC format

    Example input:
        [["C10H11O6", "C8H10N4O2"], ["C6H12O6", "H2O"]]

    Example output:
        [["C₁₀H₁₁O₆", "C₈H₁₀N₄O₂"], ["C₆H₁₂O₆", "H₂O"]]
    """
    
    if not formulas_nested:
        return []

    subs = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

    # Handle both flat and nested lists
    formatted = []
    for group in formulas_nested:
        if isinstance(group, (list, tuple)):
            formatted.append([f.translate(subs) if isinstance(f, str) else f for f in group])
        elif isinstance(group, str):
            formatted.append(group.translate(subs))
        else:
            formatted.append(group)
    return formatted



def add_seconds_from_start(t_profile, time_col="Timestamp", start_time="10/12/2024 13:05:19"):
    """
    Adds second-level precision to a time column by starting from a given
    initial time and incrementing each subsequent row by +1 second.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a datetime column with minute precision (e.g. '2024-12-10 13:05').
    time_col : str, optional
        Name of the datetime column (default = 'Timestamp').
    start_time : str or datetime.time, optional
        Starting time (e.g. '10/12/2024 13:05:19'). The first row will use this exact second.
    Returns
    -------
    pd.DataFrame
        DataFrame with updated datetime column including seconds in format '%d/%m/%Y %H:%M:%S'.
    """
    
    if time_col not in t_profile.columns:
        raise KeyError(f"Column '{time_col}' not found in DataFrame!")
    df = t_profile.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).reset_index(drop=True)
    
    # *** KEY CHANGE 1: Parse the full datetime string with correct format ***
    if isinstance(start_time, str):
        start_datetime = datetime.strptime(start_time, "%d/%m/%Y %H:%M:%S")
    else:
        # If it's already a datetime object, use it directly
        start_datetime = start_time
    
    # Generate a sequence of datetimes, incrementing by 1 second per row
    new_datetimes = [start_datetime + timedelta(seconds=i) for i in range(len(df))]
    
    # *** KEY CHANGE 2: Convert back to the desired string format ***
    df[time_col] = [dt.strftime("%d/%m/%Y %H:%M:%S") for dt in new_datetimes]
    
    return df

def align_time_axes(figs_dict, time_range=None):
    """
    Align x-axes of multiple figures to the same time range.
    
    Parameters
    ----------
    figs_dict : dict
        Dictionary of figures to align (e.g., {'fig_miro': fig1, 'fig_ts': fig2})
    time_range : tuple or None
        (min_time, max_time) to set. If None, uses the widest range across all figures.
    """
    
    if time_range is None:
        # Find the widest time range across all figures
        all_xlims = []
        for fig in figs_dict.values():
            if isinstance(fig, tuple):
                fig = fig[0]
            for ax in fig.axes:
                all_xlims.append(ax.get_xlim())
        
        time_min = min(xlim[0] for xlim in all_xlims)
        time_max = max(xlim[1] for xlim in all_xlims)
        time_range = (time_min, time_max)
    
    # Apply the same xlim to all figures
    for fig in figs_dict.values():
        if isinstance(fig, tuple):
            fig = fig[0]
        for ax in fig.axes:
            ax.set_xlim(time_range)
    
    return time_range

def rename_hydronium_columns(df):
    """
    Rename hydronium cluster columns to their simplified molecular formulas.
    Prints confirmation of changes or warnings if columns are not found.
    
    Parameters:
    df : pandas DataFrame
        DataFrame containing columns to rename
    
    Returns:
    pandas DataFrame
        DataFrame with renamed columns
    """
    
    rename_dict = {
        '(H2O)2H3O+': 'H7O3+',
        '(H2O)3H3O+': 'H9O4+'
    }
    
    # Check which columns exist before renaming
    for old_name, new_name in rename_dict.items():
        if old_name in df.columns:
            print(f"✓ Column '{old_name}' found and renamed to '{new_name}'")
        else:
            print(f"✗ Column '{old_name}' not found in DataFrame")
    
    # Perform the rename
    df_renamed = df.rename(columns=rename_dict)
    
    return df_renamed


def get_top_peaks_list(df, times=None, mz_min=None, mz_max=None, top_n=None):
    """
    Get top N peaks for multiple time points or intervals.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a 'time' column and numeric columns representing m/z values.
    times : list
        List of times (floats) or intervals (tuples (start, end)).
    mz_min, mz_max : float
        m/z filtering range.
    top_n : int
        Number of top peaks to return for each time point.
    
    Returns
    -------
    tuple of lists
        Tuple containing lists of m/z values for top peaks at each time point.
        Format: ([mz1, mz2, mz3, mz4, mz5], [mz1, mz2, mz3, mz4, mz5], ...)
    """
    
    # Defensive copy of df
    df = df.copy()
    
    # Default time selection: split total time into 4 parts
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
    
    # --- Simplified column filtering ---
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
    
    # Sort by m/z value
    sorted_indices = np.argsort(spectrum_mzs)
    spectrum_cols = [spectrum_cols[i] for i in sorted_indices]
    spectrum_mzs = np.array([spectrum_mzs[i] for i in sorted_indices], dtype=float)
    
    # --- Loop through times and collect top peaks ---
    top_peaks_list = []
    
    for time_spec in times:
        if isinstance(time_spec, (tuple, list)) and len(time_spec) == 2:
            # Handle time interval
            start_time, end_time = time_spec
            mask = (df["time"] >= float(start_time)) & (df["time"] <= float(end_time))
            df_sel = df.loc[mask]
        else:
            # Handle single time point - find closest match
            target_time = float(time_spec)
            idx_min = (df["time"] - target_time).abs().idxmin()
            df_sel = df.loc[[idx_min]]
        
        if df_sel.empty:
            top_peaks_list.append([])
            continue
        
        intensities = df_sel[spectrum_cols].mean(axis=0).to_numpy(dtype=float)
        masses = spectrum_mzs.copy()
        mask_mzrange = (masses >= mz_min) & (masses <= mz_max)
        masses_plot = masses[mask_mzrange]
        intensities_plot = intensities[mask_mzrange]
        
        if masses_plot.size == 0:
            top_peaks_list.append([])
            continue
        
        if intensities_plot.size >= top_n:
            top_idx = np.argsort(intensities_plot)[-top_n:][::-1]
            top_masses = masses_plot[top_idx]
        else:
            top_masses = masses_plot
        
        # Create list of m/z values only for this time point
        time_peaks = [float(mz) for mz in top_masses]
        top_peaks_list.append(time_peaks)
    
    return tuple(top_peaks_list)

def match_peaks_to_formulas(top_peaks_list, peak_table, tol=1e-3):
    """
    Match m/z peaks to molecular formulas from a reference peak_table.

    Parameters
    ----------
    top_peaks_tuple : tuple of lists
        Output from get_top_peaks_list(), e.g. ([mz1, mz2, ...], [mz1, mz2, ...], ...).
    peak_table : pandas.DataFrame
        Must contain columns "mass" (float) and "label" (formula or numeric).
    tol : float, optional
        Maximum allowed difference between m/z values for matching (default = 1e-3).

    Returns
    -------
    tuple of lists
        Tuple of lists containing formulas ("label") corresponding to each m/z value.
        If no match or label is numeric, "Unk" is used.
    """

    if not {"mass", "label"}.issubset(peak_table.columns):
        raise ValueError("peak_table must contain 'mass' and 'label' columns.")

    # Defensive copy and ensure numeric mass
    df = peak_table.copy()
    df["mass"] = pd.to_numeric(df["mass"], errors="coerce")
    df = df.dropna(subset=["mass"])

    # Convert labels to strings for uniform handling
    df["label"] = df["label"].astype(str)

    result = []

    for time_peaks in top_peaks_list:
        matched_labels = []
        for mz in time_peaks:
            # Compute absolute difference and find closest mass
            diffs = np.abs(df["mass"] - mz)
            idx_min = diffs.idxmin()
            if diffs.loc[idx_min] <= tol:
                label = df.loc[idx_min, "label"]
                # If label looks numeric (float or int), replace with "Unk"
                try:
                    float(label)
                    label = "Unk"
                except ValueError:
                    pass
            else:
                label = "Unk"
            matched_labels.append(label)
        result.append(matched_labels)

    return tuple(result)

def copy_headers_from_reference(folder, reference_file="20241210_113626_blank_p.csv"):
    """
    Copies the header row (column names) from a reference CSV file
    to all other CSV files in the same folder.

    Parameters
    ----------
    folder : str
        Path to the folder containing the CSV files.
    reference_file : str
        Name of the reference CSV file (without the full path).
        Default: "20241210_113626_blank_p.csv"
    """
    # Build full path to the reference file
    reference_path = os.path.join(folder, reference_file)
    
    # Check that the reference file exists
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"❌ Reference file not found: {reference_path}")
    
    # Read only the headers from the reference file
    ref_df = pd.read_csv(reference_path, nrows=0)
    reference_headers = ref_df.columns.tolist()
    
    print(f"📋 Reference headers ({len(reference_headers)} columns):")
    print(f"   {', '.join(reference_headers[:5])}... (showing first 5)")
    print()
    
    # Get all CSV files in the folder
    csv_files = glob(os.path.join(folder, "*.csv"))
    
    processed_files = 0
    skipped_files = 0
    
    for file in csv_files:
        # Skip the reference file itself
        if os.path.basename(file) == reference_file:
            print(f"⏭  Skipping reference file: {os.path.basename(file)}")
            skipped_files += 1
            continue
        
        # Read the current file
        df = pd.read_csv(file)
        
        # Check that the number of columns matches
        if len(df.columns) != len(reference_headers):
            print(f"⚠  WARNING: {os.path.basename(file)} has {len(df.columns)} columns, "
                  f"but the reference has {len(reference_headers)}. Skipping file.")
            skipped_files += 1
            continue
        
        # Assign the new headers
        df.columns = reference_headers
        
        # Save the file with updated headers
        df.to_csv(file, index=False)
        
        print(f"✔  Headers updated in: {os.path.basename(file)}")
        processed_files += 1
    
    print()
    print(f"{'='*60}")
    print(f"✅ Process completed:")
    print(f"   - Files processed: {processed_files}")
    print(f"   - Files skipped: {skipped_files}")
    print(f"{'='*60}")


def create_formulas_dict(top_peaks, formulas_list):
    """
    Create a dictionary mapping m/z values to their corresponding formulas.
    
    Parameters
    ----------
    top_peaks : list of lists
        Each inner list contains the top m/z values for each MS spectrum.
        Example: [[59.049, 85.028, ...], [85.028, 59.049, ...], ...]
    
    formulas_list : list of lists
        Each inner list contains formulas corresponding to the m/z values in top_peaks.
        The order must match the order in top_peaks.
        Example: [['C₃H₇O+', 'C₄H₅O₂+', ...], ['C₄H₅O₂+', 'C₃H₇O+', ...], ...]
    
    Returns
    -------
    dict
        Dictionary mapping m/z values (float) to formulas (str).
        Example: {59.049: 'C₃H₇O+', 85.028: 'C₄H₅O₂+', ...}
    """
    formulas_dict = {}
    
    # Validate inputs
    if len(top_peaks) != len(formulas_list):
        raise ValueError(f"Length mismatch: top_peaks has {len(top_peaks)} elements, "
                        f"formulas_list has {len(formulas_list)} elements")
    
    # Iterate through each MS spectrum
    for ms_idx, (peaks, formulas) in enumerate(zip(top_peaks, formulas_list)):
        if len(peaks) != len(formulas):
            print(f"Warning: MS{ms_idx + 1} has {len(peaks)} peaks but {len(formulas)} formulas. "
                  f"Will only map the minimum of both.")
        
        # Map each m/z to its formula
        for mz, formula in zip(peaks, formulas):
            # Round m/z to 4 decimal places for consistency
            mz_rounded = round(mz, 4)
            
            # Check if this m/z already exists with a different formula
            if mz_rounded in formulas_dict:
                if formulas_dict[mz_rounded] != formula:
                    print(f"Warning: m/z {mz_rounded} already mapped to '{formulas_dict[mz_rounded]}', "
                          f"but MS{ms_idx + 1} suggests '{formula}'. Keeping first assignment.")
            else:
                formulas_dict[mz_rounded] = formula
    
    # Sort dictionary by m/z value for easier reading
    formulas_dict = dict(sorted(formulas_dict.items()))
    
    print(f"\nCreated formulas dictionary with {len(formulas_dict)} unique m/z values:")
    print("-" * 60)
    for mz, formula in formulas_dict.items():
        print(f"m/z {mz:>8.4f} → {formula}")
    
    return formulas_dict