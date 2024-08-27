# This file is used to interpolate the data and convert it to cgs units. 
# The data is then saved in a csv file which can be used for further analysis.

if __name__ == '__main__': 

    print('#####  Data common file running #####')

import numpy as np
import pandas as pd
import os
from data_helpers import *
# from helper_functions import parameter_read

current_directory = str(os.getcwd())

#converting from 2nd unit to 1st
pc_kpc     = 1e3       # number of pc in one kpc
cm_km      = 1e5       # number of cm in one km
s_day      = 24*3600   # number of seconds in one day
s_min      = 60        # number of seconds in one hour
s_hr       = 3600      # number of seconds in one hour
cm_Rsun    = 6.957e10  # solar radius in cm
g_Msun     = 1.989e33  # solar mass in g
cgs_G      = 6.674e-8
cms_c      = 2.998e10
g_mH       = 1.6736e-24
g_me       = 9.10938e-28
cgs_h      = 6.626e-27
deg_rad    = 180e0/np.pi
arcmin_deg = 60e0
arcsec_deg = 3600e0
cm_kpc     = 3.086e+21            # number of centimeters in one parsec
cm_pc      = cm_kpc/1e+3
s_Myr      = 1e+6*(365*24*60*60)  # megayears to seconds

def parameter_read(filepath):
#opening these files and making them into dictionaries
    params = {}
    with open(filepath, 'r') as FH:
        for file in FH.readlines():
            line = file.strip()
            try:
                par_name, value = line.split('= ')
            except ValueError:
                print("Record: ", line)
                raise Exception(
                    "Failed while unpacking. Not enough arguments to supply.")
            try:
                params[par_name] = np.float64(value)
            except ValueError: #required cz of 14/11 in parameter.in file
                try:
                    num, denom = value.split('/')
                    params[par_name] = np.float64(num) / np.float64(denom)
                except ValueError:
                    params[par_name] = value
            
    return params
###########################################################################################################################################
galaxy_name = os.environ.get('galaxy_name')
base_path   = os.environ.get('MY_PATH')

switch = parameter_read(os.path.join(base_path,'inputs','switches.in'))

os.chdir(os.path.join(base_path, 'data','model_data', f'{galaxy_name}_data'))

raw_data = pd.read_csv(f'combined_data_{galaxy_name}.csv', skiprows=1)

# checking wei+21 data for m51
# raw_data = pd.read_csv(f'combined_data_{galaxy_name} - Copy.csv', skiprows=1)
corrections = pd.read_csv(f'correction_data_{galaxy_name}.csv', skiprows=1, index_col=0)

# distance correction
new_dist = corrections.iloc[-1,0]
old_dist = corrections.iloc[:-1,0].values

# inclination correction
new_i = corrections.iloc[-1,1] #deg
old_i = corrections.iloc[:-1,1].values #used new_i as no inclination correction is needed for Claude data

# converts all radii units to kpc
# inclination and distance correction

raw_data = incl_distance_correction(raw_data, distance_new = new_dist, distance_old = old_dist,\
                        i_new = np.radians(new_i), i_old = np.radians(old_i))

# data to be removed
# M31- Chemin data removed, M33- Koch and upsilon = 72 data removed
# No data removed for M51 and NGC 6946
try:
    data_rem = pd.read_csv(f'removed_data_{galaxy_name}.csv', dtype=str)
except:
    data_rem = []
for d in data_rem:
    remove_data(raw_data, d)

# read slope and intercept for temperature fits
temp_fit = np.genfromtxt(f'temp_{galaxy_name}.csv', skip_header = 1, delimiter=',')

###################################################################################################################

# Calculate omega and q from vcirc and drop vcirc column
raw_data = vcirc_to_qomega(raw_data)

# Obtain dataframe containing all radii
radii_df = keep_substring_columns(raw_data, 'r ')[0]

# Find the column with the maximum number of NaN values
# OG
# coarsest_radii_mask = radii_df.isnull().sum().idxmax()
# print("Coarsest radii is {} and the data it corresponds to is {}:".format(coarsest_radii_mask,get_adjacent_column(raw_data,coarsest_radii_mask)))

# if mol data isn't included in the calculation, 
# drop sigma_H2 before selecting coarsest data
if switch['incl_moldat'] == 'No':
    # drop sigma_H2 column from raw_data and make new copy
    raw_data_drop_sigmaH2 = raw_data.copy()
    raw_data_drop_sigmaH2 = remove_data(raw_data_drop_sigmaH2, 'sigma_H2')
    radii_df_drop_sigmaH2 = keep_substring_columns(raw_data_drop_sigmaH2, 'r ')[0]
    coarsest_radii_mask = radii_df_drop_sigmaH2.isnull().sum().idxmax()
    kpc_r = radii_df_drop_sigmaH2[coarsest_radii_mask].to_numpy()

    if __name__ == '__main__': 
        print("Coarsest radii is {} and the data it corresponds to is {}:".format(coarsest_radii_mask,get_adjacent_column(raw_data_drop_sigmaH2,coarsest_radii_mask)))

else:
    coarsest_radii_mask = radii_df.isnull().sum().idxmax()
    kpc_r = radii_df[coarsest_radii_mask].to_numpy()

    if __name__ == '__main__': 
        print("Coarsest radii is {} and the data it corresponds to is {}:".format(coarsest_radii_mask,get_adjacent_column(raw_data,coarsest_radii_mask)))




# interpolate the data 
# calculate sigma_H2 and drop molfrac columns for M31
interpolated_df = df_interpolation(raw_data,radii_df, kpc_r)
interpolated_df = molfrac_to_H2(interpolated_df)

# add temperature column
add_temp(temp_fit,interpolated_df)

# remove NaN 
nan_mask                    = np.isnan(interpolated_df)
interpolated_df             = interpolated_df[~(nan_mask.sum(axis=1)>0)]
interpolated_df_astro_units = interpolated_df.copy()

# converting to cgs units
if galaxy_name == 'm31' or galaxy_name == 'm33':
    m_to_gconv = 1e3 # needed since unit of sigma_SFR is different for these galaxies
else:
    m_to_gconv = 1

conv_factors    = np.array([1, g_Msun/(cm_pc**2), g_Msun/(cm_pc**2), g_Msun/(cm_pc**2), 1,cm_km/cm_kpc,
            g_Msun/((s_Myr*m_to_gconv)*(cm_pc**2)),1])

interpolated_df = interpolated_df*conv_factors

os.chdir(os.path.join(base_path, 'data'))

# customising file name so that it can be directly loaded to output comparison file
interpolated_df.to_csv('data_interpolated_{}.csv'.format(galaxy_name), index = False)
interpolated_df_astro_units.to_csv('data_interpolated_astro_units_{}.csv'.format(galaxy_name), index = False)

if __name__ == '__main__': 
    print('#####  Data interpolation done #####')