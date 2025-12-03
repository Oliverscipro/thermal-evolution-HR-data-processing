# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 16:04:02 2025

@author: barrio_o
"""

import pandas as pd
from datetime import datetime, timedelta
from glob import glob
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import re
import matplotlib.patches as patches



# Import speciific modules


from post_pro import *
from plot_gases import *
from plot_heatmap import *
from plot_mass_spectrum_HR import *
from plot_ions import *
from plot_combined import *
from merged_data import *
from import_data import *
from utils import *
from plots_subplots import *



#import peak list which contaions mass, formulas and integration range
peak_table = pd.read_csv("data/Peak_Table.csv", sep=",")

# Import temperature data profile from oven
t_profile = pd.read_csv("data/oven_data/Measurement_11-12-2024_14-02-21_D-raffinose_0.15umol.csv"
                        , sep = ";", header=1)


# Import and adjust the relative time to absolute time from VOCUS data
hr = import_VOCUS_HR_absolute_time_2024VI_VF("data/vocus_data/20241211_135846_D-raffinose_0.15umol_p.csv")
                                          
# Merged VOCUS data with oven data in order to have temperature profile with intensities.
merged_df = merge_hr_with_temps(hr, t_profile)

# Substract the base line (first 30 second)
sub_df = hr_sub(merged_df)

#rename water cluster columns in order to remove them later
sub_df = rename_hydronium_columns(sub_df)

# Clean the data, drop nan and water clusters
cleaned_df = remove_rows_without_temperature_watercluster(sub_df)


# Clean the data, drop nan and water clusters
cleaned_df2 = remove_rows_without_temperature_watercluster(sub_df)

#for have the correct data frame to plot masspec for HR values
cleaned_df2 = convert_columns_to_mz(cleaned_df2, peak_table)


#define times for the differents mass  and to calculate the top peaks 
times = [(20,30), (616, 625), (770,790), (960, 1010)]

#to know the predominant peak (also possible to know ater usinf the fuction for ms)
top_peaks = get_top_peaks_list(cleaned_df2, times=times, mz_min=50, mz_max=250, top_n=5)

#assign formula to the predominant peaks
formulas_list = match_peaks_to_formulas(top_peaks, peak_table, tol=1e-5)

#convert in IUPAC annotation
formulas_list = format_formulas_with_subscripts(formulas_list)

#define formula for key ions
formulas_ions = ["C2H5O2+", "C5H5O3+"]
#formulas_ions_iu = format_formulas_with_subscripts(formulas_ions)

fig_no_gas = combined_plotly_ions_left(
    cleaned_df, formulas=formulas_ions,
    top_n=5,  mz_min=50, mz_max=200,
    use_log=False, normalize_ions=False, times=None,
    auto_show=True, title=None, save_path="raffinose.png", tmax=None,
)

#plot mass spec. Delete formula list if you want to see the mains peaks 
fig_ms = plot_multiple_spectra(cleaned_df2, times=times, mz_min=55, mz_max=155, formulas_list=formulas_list)