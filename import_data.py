# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 21:54:56 2025

@author: barrio_o
"""

import pandas as pd
from datetime import datetime, timedelta
from glob import glob
import os
import numpy as np


def import_miro_data(file_path):
    """
    Simple function to import MIRO .txt data files.
    """
    # Read the file
    df = pd.read_csv(file_path, sep=';', encoding='ISO-8859-1')

    # Convert timestamp column
    df['t-stamp'] = pd.to_datetime(df['t-stamp'], format='%d.%m.%Y %H:%M:%S.%f')

    # Drop the FitWin columns (not needed)
    df = df[[c for c in df.columns if 'FitWin' not in c]]

    return df

def import_VOCUS_HR_absolute_time_2024VI_VF(filename):
    '''
    Imports a single VOCUS HR .csv file and formats it along an absolute time 
    dimension (without time offset correction).

    Parameters
    ----------
    filename : str
        Full path to the VOCUS HR .csv file.

    Returns
    -------
    hr : pandas.DataFrame
        The imported and formatted VOCUS HR data.
    '''
    
    print(f'Selected -VOCUS HR- file: {filename}')
    
    # Import VOCUS HR data from .csv file
    hr = pd.read_csv(filename, sep=',')
    
    # Drop index column if it exists
    if 'index' in hr.columns:
        hr = hr.drop(columns=['index'])
    
    # Extract start date and time from file name
    fname = filename.split('/')[-1]
    startdate = fname.split('_')[0]
    starttime = fname.split('_')[1]
    
    # Convert timestamp to datetime object
    startdatetime = datetime.strptime(startdate + starttime, '%Y%m%d%H%M%S')
    
    # Convert relative time [s from start] to absolute time (no offset)
    hr['Absolute time'] = startdatetime + pd.to_timedelta(hr['time'], unit='s')
    
    # Reformat absolute time column
    hr['Absolute time'] = hr['Absolute time'].dt.strftime('%d.%m.%Y %H:%M:%S.%f')
    
    # Move absolute time column to the front
    hr.insert(0, 'Absolute time', hr.pop('Absolute time'))
    
    return hr