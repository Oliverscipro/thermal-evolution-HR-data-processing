# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 15:05:52 2025

@author: barrio_o
"""

import os
import glob
import h5py
import numpy as np
import pandas as pd



def import_h5(folder, dataset_name=None):
    """
    Load multiple HDF5 (.h5) files from a folder.

    Parameters
    ----------
    folder : str
        Select your path to the folder containing .h5 files exported from IGOR
    dataset_name : str, optional
        Name of the dataset to extract from each file.
        If None, only lists the available datasets/groups.

    Returns
    -------
    dict
        {filename: data or list of datasets}
    """
    data = {}  #Create an empty dictionary to store the results 
    path = os.path.join(folder, "*.h5") # Build the path pattern to search for all .h5 files inside the given folder
    files = glob.glob(path) # Build the path pattern to search for all .h5 files inside the given folder

    for file in files:
        with h5py.File(file, "r") as f:
            name = os.path.basename(file)
            if dataset_name and dataset_name in f:
                data[name] = f[dataset_name][:]
            else:
                # If no dataset is given, just list the groups/datasets
                data[name] = list(f.keys())
    return data


# Example usage
#folder = "processed_2024_12_10_glucose_calibration"
#results = load_h5_from_folder(folder, dataset_name=None)

#for file, content in results.items():
 #   print(f"\n{file}: {content}")


def hr_matrix_from_h5(input_folder, output_folder):
    """
    Parameters
    ----------
    input_folder : Str
      Select the folder with the h5 files exported from IGOR.
      
    output_folder : Str
        Create a folder to save HR matrix in csv for each IGOR exported file
    Returns
    -------
    dict
      Pandas dataframe for each sample (HR matrix)
    """
    
    os.makedirs(output_folder, exist_ok=True)
    files = glob.glob(os.path.join(input_folder, "*.h5"))
    df_csv = {}
    
    for file in files:
        with h5py.File(file, "r") as f:
            
            # read peak list (m/z values)
            peak_table = f["PeakData/PeakTable"][:]
            raw_labels = peak_table["label"].astype(str)
        
            # Convert all labels to strings, formatting floats nicely
            labels = []
            for val in raw_labels:
                if isinstance(val, (float, np.floating)):
                   labels.append(f"{val:.4f}")  # format floats to 4 decimal places
                else:
                   labels.append(str(val))       # leave strings as-is
            
  
            # Get labels if available
           # if "label" in peak_table.dtype.names:
            #    labels = peak_table["label"]
             #   # Create column names: use label if exists, otherwise use mass
              #  column_names = []
               # for i, (label, mass) in enumerate(zip(labels, masses)):
                    # Decode if bytes, strip whitespace
                #    if isinstance(label, bytes):
                 #       label = label.decode('utf-8').strip()
                  #  else:
                   #     label = str(label).strip()
                    
                    # Use label if not empty, otherwise use mass
               #     if label and label != '' and label != 'nan':
                #        column_names.append(label)
                 #   else:
                  #      column_names.append(f"{mass:.4f}")
            # else:
                # No labels available, use masses only
              #  column_names = [f"{m:.4f}" for m in masses]
            
             # Peak intensities
            peak_data = f["PeakData/PeakData"][:, :, 0, :]  # (2082, 2, 1, 1611)
            peak_data = peak_data.reshape(-1, peak_data.shape[2])  # -> (4164, 1611)
            
            # read the time
            buf_times = np.array(f["TimingData/BufTimes"][:].flatten())
            if buf_times.ndim == 2 and buf_times.shape[1] == 2:
                buf_times = buf_times[:, 0]
                
            # Build the data frame     
            df = pd.DataFrame(peak_data, columns=labels)
            df.insert(0, "time", buf_times[:len(df)])
            
        # Save as CSV (correct indentation - outside 'with' block, inside 'for' loop)
        sample_name = os.path.splitext(os.path.basename(file))[0]
        output_path = os.path.join(output_folder, f"{sample_name}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {output_path}")
        
        # Keep in memory
        df_csv[sample_name] = df
    
    return df_csv
    # Process files into DataFrames + CSV
    
    
dfs =hr_matrix_from_h5(
    input_folder=r"Z:\Therm_Evo_Analysis_Oliver\Data analysis\Data analysis 2024 campaign\IGOR_exported\2025_10_compilation\Processed",
    output_folder=r"Z:\Therm_Evo_Analysis_Oliver\Data analysis\Data analysis 2024 campaign\IGOR_exported\2025_10_compilation\Processed\csv_processed")


            
    