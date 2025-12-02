# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 14:05:04 2025

@author: barrio_o
"""

# Import general modules

import pandas as pd
from datetime import datetime, timedelta
from glob import glob
import os
import numpy as np
import pandas as pd
import numpy as np
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
times = [(20,30), (616, 625), (770,790), (960, 1010)]

#to know the predominant peak (also possible to know ater usinf the fuction for ms)
top_peaks = get_top_peaks_list(cleaned_df2, times=times, mz_min=50, mz_max=250, top_n=5)

#assign formula to the predominant peaks
formulas_list = match_peaks_to_formulas(top_peaks, peak_table, tol=1e-5)

#convert in IUPAC annotation
formulas_list = format_formulas_with_subscripts(formulas_list)

#define formula for key ions
formulas_ions = ["C8H5O3+"]

gases=[  "CO wet", "CH4 wet"]


#Plot heat map, optonial use log scale
fig_hm = plot_heatmap(cleaned_df,use_log=False, mz_min=55, mz_max=399, vmin=0)

fig_left = combined_plotly_ions_gases_left(
    cleaned_df, merged_temp_gases, formulas=formulas_ions,
    top_n=5, gases=gases, mz_min=50, mz_max=200,
    use_log=False, normalize=True, times=None,
    auto_show=True, title=None, save_path="alpha-pinene-SOA-chamber-dry.png", gas_smoothing="raw",
     rolling_window=None
)

fig_no_gas = combined_plotly_ions_left(
    cleaned_df, formulas=formulas_ions,
    top_n=5,  mz_min=50, mz_max=200,
    use_log=False, normalize_ions=False, times=None,
    auto_show=True, title=None, save_path="first.png", tmax=None,
)

#plot mass spec. Delete formula list if you want to see the mains peaks 
fig_ms = plot_multiple_spectra(cleaned_df2, times=times, mz_min=55, mz_max=155, formulas_list=formulas_list) 

#plot time series
#fig_ts = plot_peaks_timeseries(cleaned_df, formulas=formulas_ions, normalize=True)
#dict_formulas = create_formulas_dict(top_peaks, formulas_list)
#plot gases and temperature profile, adjust time for each sample
#fig_miro = trace_gas_visualizer_temp_second(
   # merged_temp_gases, gases=gases,
   # start_time="12:09",
   # end_time="12:28",
  #  figsize=(14, 7),
 #   normalize=True
#)


#plot gases and temperature profile, adjust time for each sample
#fig_miro_2 = trace_gas_visualizer_temp_second_soft(
 #   merged_temp_gases, gases=gases,
  #  start_time="13:36",
   # end_time="16:55",
   # figsize=(14, 7),
   # normalize=True,  gas_smoothing="rolling_median",
    # rolling_window=60
#)






###############################################################################


#final figure 
#combine_figures(fig_miro, fig_ts, fig_ms, fig_hm)







###This figure working in progress ########

#fig_complete = combined_plotly_ions_gases_2D_MS(cleaned_df, merged_temp_gases, cleaned_df2, formulas=formulas_ions,
 #                                    top_n=5, gases=['NO wet'], mz_min=50, mz_max=250,
  #                                   use_log=False, normalize=False, times=times,
   #                                  auto_show=True, title=None, dict_formulas=dict_formulas, save_path="complete_nicotine.png")