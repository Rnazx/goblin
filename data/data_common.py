import numpy as np
import pandas as pd
import os
from data_helpers import *
current_directory = str(os.getcwd())

#converting from 2nd unit to 1st
pc_kpc = 1e3  # number of pc in one kpc
cm_km = 1e5  # number of cm in one km
s_day = 24*3600  # number of seconds in one day
s_min = 60  # number of seconds in one hour
s_hr = 3600  # number of seconds in one hour
cm_Rsun = 6.957e10  # solar radius in cm
g_Msun = 1.989e33  # solar mass in g
cgs_G = 6.674e-8
cms_c = 2.998e10
g_mH = 1.6736e-24
g_me = 9.10938e-28
cgs_h = 6.626e-27
deg_rad = 180e0/np.pi
arcmin_deg = 60e0
arcsec_deg = 3600e0
cm_kpc = 3.086e+21  # number of centimeters in one parsec
cm_pc = cm_kpc/1e+3
s_Myr = 1e+6*(365*24*60*60)  # megayears to seconds

###########################################################################################################################################
raw_data = pd.read_csv('formatted_data.csv')

raw_data = vcirc_to_qomega(raw_data)

radii_df = keep_substring_columns(raw_data, 'r ')[0]

# Find the column with the maximum number of NaN values
coarsest_radii_mask = radii_df.isnull().sum().idxmax()

print("Coarsest radii is {} and the data it corresponds to is {}:".format(coarsest_radii_mask,get_adjacent_column(raw_data,coarsest_radii_mask)))
kpc_r = radii_df[coarsest_radii_mask].to_numpy()
#print(raw_data)
interpolated_df = df_interpolation(raw_data,radii_df, kpc_r)
nan_mask = np.isnan(interpolated_df)
interpolated_df = interpolated_df[~(nan_mask.sum(axis=1)>0)]
#interpolated_df.dropna()

interpolated_df = molfrac_to_H2(interpolated_df)
add_temp(0.017e+4,0.5e+4,interpolated_df)

conv_factors=np.array([1, (g_Msun/(cm_pc**2) ), g_Msun/(cm_pc**2), g_Msun/(cm_pc**2), 1,cm_km/cm_kpc,
              g_Msun/((s_Myr*1e3)*(cm_pc**2)),1])
interpolated_df = interpolated_df*conv_factors
#interpolated_df= replace_conversion(interpolated_df, 'kpc', 'cm')
#interpolated_df= replace_conversion(interpolated_df, 'kms', 'cms')

print(interpolated_df)
interpolated_df.to_csv('data_interpolated.csv', index = False)