# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 08:49:56 2025

@author: barrio_o
"""

import pandas as pd

def merge_hr_with_temps(hr, t_profile):
    """
    Merge hr dataframe (0.5s resolution) with temperature data from t_profile (1s resolution).
    Sets relative time 0 at first overlapping timestamp, with negative times before temp data.
    
    Parameters:
    -----------
    hr : pd.DataFrame
        High resolution dataframe with 0.5 second intervals
    t_profile : pd.DataFrame
        Temperature profile dataframe with 1 second intervals
    
    Returns:
    --------
    pd.DataFrame
        Merged dataframe with hr data and temperature columns next to time columns.
        Relative time is 0 at first overlapping point, negative before that.
    """
    import pandas as pd
    
    # Create copies to avoid modifying originals
    hr_copy = hr.copy()
    t_profile_copy = t_profile.copy()
    
    # Convert timestamps to datetime
    hr_copy['Absolute time'] = pd.to_datetime(hr_copy['Absolute time'])
    t_profile_copy['Timestamp'] = pd.to_datetime(t_profile_copy['Timestamp'])
    
    # Select only needed columns from t_profile
    t_profile_subset = t_profile_copy[['Timestamp', 'Target Temp (C)', 'Measured Temp (C)']].copy()
    
    # Sort both dataframes by time
    hr_copy = hr_copy.sort_values('Absolute time')
    t_profile_subset = t_profile_subset.sort_values('Timestamp')
    
    # Merge using merge_asof (matches to nearest timestamp)
    merged_df = pd.merge_asof(
        hr_copy,
        t_profile_subset,
        left_on='Absolute time',
        right_on='Timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('0.5s')  # Only match if within 0.5 seconds
    )
    
    # Drop the extra Timestamp column from t_profile
    if 'Timestamp' in merged_df.columns:
        merged_df = merged_df.drop('Timestamp', axis=1)
    
    # Find the first row where temperature data exists (first overlap)
    first_temp_idx = merged_df['Target Temp (C)'].first_valid_index()
    
    if first_temp_idx is not None:
        # Get the absolute time at first overlap
        reference_time = merged_df.loc[first_temp_idx, 'Absolute time']
        
        # Calculate relative time from this reference point
        # Before reference: negative values, at reference: 0, after reference: positive values
        merged_df['time'] = (merged_df['Absolute time'] - reference_time).dt.total_seconds()
    else:
        # If no overlap found, keep original time column if it exists
        if 'time' not in merged_df.columns:
            merged_df['time'] = 0
    
    # Reorder columns to place temperature columns after time columns
    cols = merged_df.columns.tolist()
    
    # Find the position of 'time' column
    time_col_idx = cols.index('time')
    
    # Remove temperature columns from their current position
    cols.remove('Target Temp (C)')
    cols.remove('Measured Temp (C)')
    
    # Insert temperature columns right after the time column
    cols.insert(time_col_idx + 1, 'Target Temp (C)')
    cols.insert(time_col_idx + 2, 'Measured Temp (C)')
    
    # Reorder the dataframe
    merged_df = merged_df[cols]
    
    return merged_df


def merge_temp_gases(t_profile, df_trace_gas):
    """
    Merge temperature profile with trace gas data using overlapping period.
    Handles mixed timestamp formats and ignores milliseconds in trace gas.

    Parameters
    ----------
    t_profile : pd.DataFrame
        Has 'Timestamp' in format "%d/%m/%Y %H:%M:%S" or similar.
    df_trace_gas : pd.DataFrame
        Has 't-stamp' in format "YYYY-MM-DD HH:MM:SS.sss".

    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing overlapping data.
    """

    t_profile = t_profile.copy()
    df_trace_gas = df_trace_gas.copy()

    # --- Parse timestamps with correct format ---
    # Try dd/mm/yyyy first (your profile file)
    t_profile["Timestamp"] = pd.to_datetime(
        t_profile["Timestamp"], format="%d-%m-%Y %H:%M:%S", errors="coerce"
    )
    if t_profile["Timestamp"].isna().all():
        # If all failed, try ISO (yyyy-mm-dd)
        t_profile["Timestamp"] = pd.to_datetime(
            t_profile["Timestamp"], errors="coerce"
        )

    # Parse ISO format for gas timestamps (e.g., 2024-12-10 00:00:00.800)
    df_trace_gas["t-stamp"] = pd.to_datetime(df_trace_gas["t-stamp"], errors="coerce")

    # --- Drop NaNs ---
    t_profile.dropna(subset=["Timestamp"], inplace=True)
    df_trace_gas.dropna(subset=["t-stamp"], inplace=True)

    # --- Ignore milliseconds ---
    df_trace_gas["t-stamp"] = df_trace_gas["t-stamp"].dt.floor("S")

    # --- Compute overlap range ---
    start = max(t_profile["Timestamp"].min(), df_trace_gas["t-stamp"].min())
    end = min(t_profile["Timestamp"].max(), df_trace_gas["t-stamp"].max())

    print(f"🕒 t_profile range: {t_profile['Timestamp'].min()} → {t_profile['Timestamp'].max()}")
    print(f"🕒 df_trace_gas range: {df_trace_gas['t-stamp'].min()} → {df_trace_gas['t-stamp'].max()}")
    print(f"🔍 Overlap: {start} → {end}")

    if start >= end:
        print("⚠️ No overlapping data found — check date formats or ranges.")
        return pd.DataFrame()

    # --- Keep only overlapping rows ---
    t_profile = t_profile[(t_profile["Timestamp"] >= start) & (t_profile["Timestamp"] <= end)]
    df_trace_gas = df_trace_gas[(df_trace_gas["t-stamp"] >= start) & (df_trace_gas["t-stamp"] <= end)]

    # --- Merge on exact seconds ---
    merged = pd.merge(
        df_trace_gas,
        t_profile,
        left_on="t-stamp",
        right_on="Timestamp",
        how="inner"
    ).sort_values("t-stamp").reset_index(drop=True)

    print(f"✅ Merged rows: {len(merged)}")
    return merged


