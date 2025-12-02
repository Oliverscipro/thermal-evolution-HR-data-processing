# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 13:53:53 2025

@author: barrio_o
"""
import os
import glob
import h5py
import numpy as np
import pandas as pd

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
    print(f"Found {len(files)} files")
    df_csv = {}
    
    for file in files:
        print(f"\n{'='*50}")
        print(f"Processing: {file}")
        print('='*50)
        
        with h5py.File(file, "r") as f:
            # 1. Peak list (m/z values and labels)
            peak_table = f["PeakData/PeakTable"][:]
            
            # DEBUG: Print structure
            print("\n>>> Peak table dtype names:", peak_table.dtype.names)
            print(">>> Peak table shape:", peak_table.shape)
            print(">>> First entry full:", peak_table[0])
            print(">>> Entry at index 1:", peak_table[1])
            
            masses = peak_table["mass"]
            print(">>> First 5 masses:", masses[:5])
            
            # Try to get labels - check actual field name
            labels = None
            possible_names = ["label", "Label", "name", "Name", "ion", "Ion", "compound", "Compound"]
            
            for possible_name in possible_names:
                if possible_name in peak_table.dtype.names:
                    labels = peak_table[possible_name]
                    print(f"\n>>> FOUND labels in field: '{possible_name}'")
                    print(f">>> Labels dtype: {labels.dtype}")
                    print(f">>> First 15 labels: {labels[:15]}")
                    break
            
            if labels is None:
                print("\n>>> NO label field found! Available fields:", peak_table.dtype.names)
            
            # Create column names
            column_names = []
            if labels is not None:
                for i, (label, mass) in enumerate(zip(labels, masses)):
                    # Decode if bytes, strip whitespace
                    if isinstance(label, bytes):
                        label = label.decode('utf-8').strip()
                    elif isinstance(label, (np.bytes_, np.str_)):
                        label = str(label).strip()
                    else:
                        label = str(label).strip()
                    
                    # Use label if not empty, otherwise use mass
                    if label and label not in ['', 'nan', 'None', 'b""', "b''", '0', '0.0']:
                        column_names.append(label)
                    else:
                        column_names.append(f"{mass:.4f}")
            else:
                # No labels available, use masses only
                column_names = [f"{m:.4f}" for m in masses]
            
            print(f"\n>>> First 15 column names: {column_names[:15]}")
            
            # 2. Peak intensities
            peak_data = f["PeakData/PeakData"][:, :, 0, :]
            peak_data = peak_data.reshape(-1, peak_data.shape[2])
            print(f">>> Peak data shape: {peak_data.shape}")
            
            # 3. Times
            buf_times = np.array(f["TimingData/BufTimes"][:].flatten())
            print(f">>> BufTimes shape: {buf_times.shape}")
            if buf_times.ndim == 2 and buf_times.shape[1] == 2:
                buf_times = buf_times[:, 0]
            
            # 4. Build the dataframe
            df = pd.DataFrame(peak_data, columns=column_names)
            df.insert(0, "time", buf_times[:len(df)])
        
        # Save as CSV
        sample_name = os.path.splitext(os.path.basename(file))[0]
        output_path = os.path.join(output_folder, f"{sample_name}.csv")
        df.to_csv(output_path, index=False)
        print(f"\n✓ Saved {output_path}")
        print(f"✓ DataFrame shape: {df.shape}")
        
        # Keep in memory
        df_csv[sample_name] = df
    
    print(f"\n{'='*50}")
    print(f"Completed! Processed {len(df_csv)} files")
    print('='*50)
    
    return df_csv