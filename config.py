# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 15:26:54 2025

@author: barrio_o

This module contain specific lines to import the data and define the varaibles
that are specific for each sample
"""

#import general packages
import pandas as pd


#import specific modules
from import_data import *

#import peak list which contaions mass, formulas and integration range.
peak_table = pd.read_csv("data/Peak_Table.csv", sep=",")

# Import temperature data profile from oven
t_profile = pd.read_csv("data/oven_data/Measurement_11-12-2024_14-02-21_D-raffinose_0.15umol.csv"
                        , sep = ";", header=1)

# Import and adjust the relative time to absolute time from VOCUS data
hr = import_VOCUS_HR_absolute_time_2024VI_VF(
    "data/vocus_data/20241211_135846_D-raffinose_0.15umol_p.csv")

# Import MIRO data 
df_trace_gas = import_miro_data("data/miro_data/2024-12-11 MGA SN21.txt")

#define times for the differents mass  and to calculate the top peaks 
times = [(20,30), (616, 625), (770,790), (960, 1010)]

#define formula for key ions
formulas_ions = ["C8H5O3+"]

gases=[  "CO wet", "CH4 wet"]