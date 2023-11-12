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
galaxy_name = os.environ.get('galaxy_name')
base_path = os.environ.get('MY_PATH')
# l = ['m31','m33','m51', 'ngc6946']
# for galaxy_name in l:

os.chdir(os.path.join(base_path, 'data','model_data', f'{galaxy_name}_data'))

raw_data = pd.read_csv(f'combined_data_{galaxy_name}.csv', skiprows=1)
corrections = pd.read_csv(f'correction_data_{galaxy_name}.csv', skiprows=1, index_col=0)

#distance correction
new_dist= corrections.iloc[-1,0]
old_dist = corrections.iloc[:-1,0].values

#inclination correction
new_i= corrections.iloc[-1,1] #deg
old_i= corrections.iloc[:-1,1].values #used i as no inclination correction is needed for Claude data

raw_data = incl_distance_correction(raw_data, distance_new=new_dist, distance_old=old_dist,\
                        i_new=np.radians(new_i), i_old=np.radians(old_i))

#data to be removed
try:
    data_rem = pd.read_csv(f'removed_data_{galaxy_name}.csv', dtype=str)
except:
    data_rem = []
for d in data_rem:
    remove_data(raw_data, d)

temp_fit = np.genfromtxt(f'temp_{galaxy_name}.csv', skip_header = 1,delimiter=',')

###################################################################################################################
#Finished Distance and inclination corrections
raw_data = vcirc_to_qomega(raw_data)
radii_df = keep_substring_columns(raw_data, 'r ')[0]

# Find the column with the maximum number of NaN values
coarsest_radii_mask = radii_df.isnull().sum().idxmax()

print("Coarsest radii is {} and the data it corresponds to is {}:".format(coarsest_radii_mask,get_adjacent_column(raw_data,coarsest_radii_mask)))
kpc_r = radii_df[coarsest_radii_mask].to_numpy()

#print(raw_data)
interpolated_df = df_interpolation(raw_data,radii_df, kpc_r)
interpolated_df = molfrac_to_H2(interpolated_df)
add_temp(temp_fit,interpolated_df)
nan_mask = np.isnan(interpolated_df)
interpolated_df = interpolated_df[~(nan_mask.sum(axis=1)>0)]
#interpolated_df.dropna()
# Changed for m51 and ngc6946
if galaxy_name == 'm31' or galaxy_name == 'm33':
    m_to_gconv = 1e3
else:
    m_to_gconv = 1
conv_factors=np.array([1, (g_Msun/(cm_pc**2) ), g_Msun/(cm_pc**2), g_Msun/(cm_pc**2), 1,cm_km/cm_kpc,
            g_Msun/((s_Myr*m_to_gconv)*(cm_pc**2)),1])
interpolated_df = interpolated_df*conv_factors
#interpolated_df= replace_conversion(interpolated_df, 'kpc', 'cm')
#interpolated_df= replace_conversion(interpolated_df, 'kms', 'cms')
print(interpolated_df)
os.chdir(os.path.join(base_path, 'data'))
interpolated_df.to_csv('data_interpolated.csv', index = False)