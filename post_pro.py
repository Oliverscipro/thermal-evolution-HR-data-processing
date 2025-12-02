# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 09:46:41 2025

@author: barrio_o
"""

import pandas as pd
from datetime import datetime, timedelta
import numpy as np


def hr_sub(merged_df):
    """
    Subtract background from HR mass spectrometry data.
    
    Parameters
    ----------
    merged_df : pandas.DataFrame
        Merged data frame with temperature profile and high resolution mass spectrometry.
        Must contain a 'time' column for background calculation.
    
    Returns
    -------
    sub_hr : pandas.DataFrame
        Data frame with background-subtracted values for all mass spectrometry columns.
    """
    # Create a copy to avoid modifying the original
    sub_hr = merged_df.copy()
    
    # Define columns to exclude from background subtraction
    exclude_cols = ['Index', 'Absolute time', 'time', 'Measured Temp (C)', 'Target Temp (C)']
    
    # Get all columns except the excluded ones
    cols_to_subtract = [col for col in sub_hr.columns if col not in exclude_cols]
    
    # Filter data for background period (0 <= time <= 30 seconds)
    background_df = merged_df[(merged_df["time"] >= 0) & (merged_df["time"] <= 30)]

    # Check if background data is available
    if background_df.empty:
        print("⚠️ Warning: No data found for background period (0–30 s)")
        return sub_hr

    # Calculate mean background for each column
    background_means = background_df[cols_to_subtract].mean()
    
    # Subtract background from each column
    for col in cols_to_subtract:
        sub_hr[col] = sub_hr[col] - background_means[col]
    
    print("✓ Background subtracted using time range 0–30 s")
    print(f"✓ Number of background rows used: {len(background_df)}")
    print(f"✓ Processed {len(cols_to_subtract)} mass spectrometry columns")
    
    return sub_hr


        
def remove_rows_without_temperature_watercluster(merged_df_hr_temp):
    """
    Remove rows from merged_df_hr_temp where temperature data is missing,
    and drop water cluster columns. Time column is preserved as-is.
    """
    
    df = merged_df_hr_temp.copy()
    
    # --- Identify temperature columns ---
    temp_cols = [col for col in ["Measured Temp (C)", "Target Temp (C)"] if col in df.columns]
    if not temp_cols:
        raise ValueError("No temperature columns found in the DataFrame.")
    
    # --- Remove rows missing both temperature columns ---
    cleaned_df = df.dropna(subset=temp_cols, how="all").reset_index(drop=True)
    
    # --- Remove water cluster columns ---
    water_cluster_cols = ['H3O+', 'H5O2+', 'H7O3+',
     'H9O4+', 'm/Q 37', 'm/Q 55', 'm/Q 73']
    existing_water_clusters = [col for col in water_cluster_cols if col in cleaned_df.columns]
    
    if existing_water_clusters:
        cleaned_df = cleaned_df.drop(columns=existing_water_clusters)
        print(f"💧 Removed {len(existing_water_clusters)} water cluster columns: {existing_water_clusters}")
    else:
        print("ℹ️ No water cluster columns found to remove.")
    
    # --- Report removed temperature rows ---
    removed_rows = len(df) - len(cleaned_df)
    print(f"✅ Removed {removed_rows} rows with missing temperature data. Remaining rows: {len(cleaned_df)}")
    
    return cleaned_df


def convert_columns_to_mz(cleaned_df: pd.DataFrame, peak_table: pd.DataFrame) -> pd.DataFrame:
    """
    Replace column names in cleaned_df with corresponding m/z values
    based on matches found in peak_table (label → mass mapping).
    """
    
    df = cleaned_df.copy()
    
    # Create mapping from formula (label) to mass
    mapping = pd.Series(peak_table['mass'].values, index=peak_table['label']).to_dict()
    
    # Replace column names: if found in mapping, use mass; otherwise, keep as is
    new_columns = [mapping.get(col, col) for col in cleaned_df.columns]
    
    # Assign new columns
    df.columns = new_columns
    
    return df

