# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 16:52:01 2025

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
from merged_data import *
from import_data import *
from utils import *
from plots_subplots import *
from plot_comparative_spec import *





#import peak list which contaions mass, formulas and integration range
peak_table = pd.read_csv("data/Peak_Table.csv", sep=",")


###########################first data frame##########################

# Import temperature data profile from oven
t_profile_ch = pd.read_csv("data/oven_data/Measurement_19-12-2024_12-09-10_BB_chamber_CT04_aged POA and SOA.csv"
                        , sep = ";", header=1)


# Import and adjust the relative time to absolute time from VOCUS data
hr_ch = import_VOCUS_HR_absolute_time_2024VI_VF("data/vocus_data/20241219_120111__BB_chamber_CT04_aged POA and SOA_p.csv")
                                          
# Merged VOCUS data with oven data in order to have temperature profile with intensities.
merged_df_ch = merge_hr_with_temps(hr_ch, t_profile_ch)

# Substract the base line (first 30 second)
sub_df_ch = hr_sub(merged_df_ch)

#rename water cluster columns in order to remove them later
sub_df_ch = rename_hydronium_columns(sub_df_ch)

# Clean the data, drop nan and water clusters
cleaned_df_ch = remove_rows_without_temperature_watercluster(sub_df_ch)


# Clean the data, drop nan and water clusters
cleaned_df2_ch = remove_rows_without_temperature_watercluster(sub_df_ch)

#for have the correct data frame to plot masspec for HR values
cleaned_df2_ch = convert_columns_to_mz(cleaned_df2_ch, peak_table)

####################################second dataframe############################

# Import temperature data profile from oven
t_profile_mag = pd.read_csv("data/oven_data/Measurement_21-01-2025_11-38-41_Swiss_MAG_2018_12_12.csv"
                        , sep = ";", header=1)


# Import and adjust the relative time to absolute time from VOCUS data
hr_mag = import_VOCUS_HR_absolute_time_2024VI_VF("data/vocus_data/20250121_113444_MAG_2018_12_12_p.csv")
                                          
# Merged VOCUS data with oven data in order to have temperature profile with intensities.
merged_df_mag = merge_hr_with_temps(hr_mag, t_profile_mag)

# Substract the base line (first 30 second)
sub_df_mag = hr_sub(merged_df_mag)

#rename water cluster columns in order to remove them later
sub_df_mag = rename_hydronium_columns(sub_df_mag)

# Clean the data, drop nan and water clusters
cleaned_df_mag = remove_rows_without_temperature_watercluster(sub_df_mag)


# Clean the data, drop nan and water clusters
cleaned_df2_mag = remove_rows_without_temperature_watercluster(sub_df_mag)

#for have the correct data frame to plot masspec for HR values
cleaned_df2_mag = convert_columns_to_mz(cleaned_df2_mag, peak_table)

# Import temperature data profile from oven
t_profile_ch = pd.read_csv("data/oven_data/Measurement_19-12-2024_12-09-10_BB_chamber_CT04_aged POA and SOA.csv"
                        , sep = ";", header=1)


        ####################### variables definitons #######

#define times for the differents mass  and to calculate the top peaks 
time =  [(960,1000)]

#to know the predominant peak (also possible to know ater usinf the fuction for ms)
top_peaks_ch = get_top_peaks_list(cleaned_df2_ch, times=time_point, mz_min=50, mz_max=250, top_n=5)

#assign formula to the predominant peaks
formulas_list_ch = match_peaks_to_formulas(top_peaks_ch, peak_table, tol=1e-5)

#convert in IUPAC annotation
formulas_list_ch = format_formulas_with_subscripts(formulas_list_ch)

#to know the predominant peak (also possible to know ater usinf the fuction for ms)
top_peaks_mag = get_top_peaks_list(cleaned_df2_mag, times=time_point, mz_min=50, mz_max=250, top_n=5)

#assign formula to the predominant peaks950,110
formulas_list_mag = match_peaks_to_formulas(top_peaks_mag, peak_table, tol=1e-5)

#convert in IUPAC annotation
formulas_list_mag = format_formulas_with_subscripts(formulas_list_mag)

# Plot with custom formulas
plot_mirror_spec(df_top=cleaned_df2_ch, 
                 df_bottom=cleaned_df2_mag, 
                 times=time, 
                 mz_min=55, 
                 mz_max=155,
                 figsize=(60,30),
                 top_n=5,
                 formulas_list_top=formulas_list_ch,
                 formulas_list_bottom=formulas_list_mag, save_path="mag_ch.png")