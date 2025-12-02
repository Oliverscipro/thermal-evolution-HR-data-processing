# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 09:18:01 2025

@author: barrio_o
"""

from plots_subplots import *

import pandas as pd
from datetime import datetime, timedelta
from glob import glob
import os
import numpy as np
import pandas as pd
import numpy as np
import re




# Import speciific modules


from post_pro import *
from merged_data import *
from import_data import *
from utils import *
from plots_subplots import *



#import peak list which contaions mass, formulas and integration range
peak_table = pd.read_csv("data/Peak_Table.csv", sep=",")

# Import temperature data profile from oven
t_profile = pd.read_csv("data/oven_data/Measurement_11-12-2024_14-02-21_D-raffinose_0.15umol.csv"
                        , sep = ";", header=1)

#this allow to add seconds in the time for the oven usufull to merged later, put first exact time
                         
#t_profile = add_seconds_from_start(t_profile, time_col="Timestamp", start_time="10/12/2024 15:38:46")

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

# Import MIRO data 
df_trace_gas = import_miro_data("data/therm_evo_github_codes/miro_data/2024-12-11 MGA SN21.txt")

# Clean the data, drop nan and water clusters
cleaned_df2 = remove_rows_without_temperature_watercluster(sub_df)

#for have the correct data frame to plot masspec for HR values
cleaned_df2 = convert_columns_to_mz(cleaned_df2, peak_table)

# data frame for gases and temperature merged , only produces the overlaping period
merged_temp_gases = merge_temp_gases(t_profile, df_trace_gas)


#define times for the differents mass  and to calculate the top peaks 
times = [(0,400), (400,600), (600,800), (800, 1200)]

#to know the predominant peak (also possible to know ater usinf the fuction for ms)
top_peaks = get_top_peaks_list(cleaned_df2, times=times, mz_min=50, mz_max=150, top_n=5)

#assign formula to the predominant peaks
formulas_list = match_peaks_to_formulas(top_peaks, peak_table, tol=1e-3)

#convert in IUPAC annotation
formulas_list = format_formulas_with_subscripts(formulas_list)

#Those are the formulas to show in the masspectrum
#formulas_list = [("C2H5O2+", "C5H5O2+","C6H7O+","C3H5O2+","C6H7O2+"),
 #                ("C2H5O2+","C5H5O2+","C4H5O2+","C3H5O2+","C4H5O+"),
  #               ("C5H5O2+","C2H5O2+","C4H5O2+","C4H5O+","C3H5O2+"  ), 
   #              ("C5H5O2+","C2H5O2+","C4H5O2+","C4H5O+", "C5H5O+" )]



#define formula for key ions
formulas = ["C5H5O2+", "C2H5O2+"]

fig_left = combined_plotly_left(cleaned_df, merged_temp_gases, formulas,
                              gases=['CO wet'], mz_min=70, mz_max=185,
                              use_log=False, normalize=False, auto_show=True)

plot_multiple_spectra_plotly(
    cleaned_df2, times=times, mz_min=100, mz_max=150,
    window_seconds=None, formulas_list=formulas_list,
    top_n=5, subplot_layout=(4,1)
)