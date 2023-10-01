import numpy as np
from sympy import *
from fractions import Fraction
import pickle
from scipy.interpolate import griddata
import pandas as pd
import os

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

#REQUIRED FUNCTIONS
###########################################################################################################################################
#extrapolation
def interpolation(list1,list2,standard):
    interpolated_data = griddata(list1, list2, standard, method='linear', fill_value=nan, rescale=False)
    return interpolated_data

def find_and_multiply_column(dataframe, substring, multiplier):
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = dataframe.copy()
    i = 0
    for col in result_df.columns:
        if substring in col:
            try:
                result_df[col] = result_df[col] * multiplier[i]
            except:
                result_df[col] = result_df[col] * multiplier
            i+=1
    return result_df

def keep_substring_columns(dataframe, substring):
    # Get the columns that contain the specified substring
    filtered_columns = [col for col in dataframe.columns if substring in col]
    
    # Create a new DataFrame with only the filtered columns
    result_df = dataframe[filtered_columns].copy()
    
    return result_df, filtered_columns

# get the next or previous column of the dataframe
def get_adjacent_column(df, col, next = True):
    index_of_target = df.columns.get_loc(col)
    if index_of_target < len(df.columns) - 1:
        if next:
            return df.columns[index_of_target+1]
        else:
            return df.columns[index_of_target-1]
    else:
        print("The target column is the last column.")

def df_interpolation(df, radii_df, standard):
    result_df = df.copy()
    for cols in radii_df.columns:
        result_df[get_adjacent_column(df, cols)] = interpolation(df[cols],df[get_adjacent_column(df, cols)],standard)
        result_df.drop(columns=[cols], inplace=True)
    result_df.insert(0, 'kpc_r', standard)
    return result_df

def inclination_correction(df, i_new, i_old):
    df_corr = df.copy()
    df_corr.iloc[:,1:].multiply(np.cos(i_new)/np.cos(i_old), axis = 1)
    return df_corr

def molfrac_to_H2(df):
    df = df.copy()
    molfrac_data = keep_substring_columns(df, 'molfrac')
    if molfrac_data[0].empty:
        return
    else:
        HI_data = keep_substring_columns(interpolated_df, 'HI')
        sigma_H2 = HI_data[0].multiply((1/(1-molfrac_data[0])).values, axis = 0)
        index_of_HI = df.columns.get_loc(HI_data[1][0])
        df.insert(index_of_HI+1, 'sigma_H2', sigma_H2)
        df.drop(columns=molfrac_data[1], inplace=True)
        return df
    
def add_temp(m, c, df):
    r = df.iloc[:,0].to_numpy().flatten()
    T = m*r +c
    df.insert(len(df.columns), 'T', T)
    return

def replace_conversion(df, substring_to_replace, replacement_string):
    # Create a dictionary for column renaming
    rename_dict = {col: col.replace(substring_to_replace, replacement_string) for col in df.columns}
    # Rename the columns using the dictionary
    updated_df = df.rename(columns=rename_dict)

    return updated_df

def vcirc_to_qomega(df):
    df = df.copy()
    vcirc_data = keep_substring_columns(df, 'vcirc')
    if vcirc_data[0].empty:
        return
    else:
        r = df.iloc[:,0].to_numpy().flatten()*cm_kpc
        Om = vcirc_data[0].to_numpy().flatten()/r
        q = -1 * r/Om* np.gradient(Om)/np.gradient(r)
        index_of_vcirc = df.columns.get_loc(vcirc_data[1][0])
        df.insert(index_of_vcirc, 'q', q)
        df.insert(index_of_vcirc, '\Omega', Om)
        df.drop(columns=vcirc_data[1], inplace=True)
        return df


###########################################################################################################################################
os.chdir(current_directory+'\M31_data')
raw_data = pd.read_csv('combined_data_m31.csv', skiprows=1)

radii_df = keep_substring_columns(raw_data, 'r ')[0]
#radii_df = radii_df.drop(columns='error kms')

#to obtain radius data from every df
distance_m31= 0.78 #Mpc
distances_Mpc=np.array([0.785,0.785,0.780,0.780])

#convert arcmin to kpc
radii_df = find_and_multiply_column(radii_df, 'r arcmin', ((distance_m31*1000)/(arcmin_deg*deg_rad)))
#convert arcsec to kpc
radii_df = find_and_multiply_column(radii_df, 'r arcsec', ((distance_m31*1000)/(arcsec_deg*deg_rad)))
#distance correction
radii_df = find_and_multiply_column(radii_df, 'r kpc', distance_m31/distances_Mpc)

# print("Original DataFrame:")
#print(raw_data)
print("\nDataFrame after multiplication:")

raw_data[radii_df.columns] = radii_df

# Find the column with the maximum number of NaN values
coarsest_radii_mask = radii_df.isnull().sum().idxmax()

print("Coarsest radii is {} and the data it corresponds to is {}:".format(coarsest_radii_mask,get_adjacent_column(raw_data,coarsest_radii_mask)))
kpc_r = radii_df[coarsest_radii_mask].to_numpy()

interpolated_df = df_interpolation(raw_data,radii_df, kpc_r)
nan_mask = np.isnan(interpolated_df)
interpolated_df = interpolated_df[~(nan_mask.sum(axis=1)>0)]
#interpolated_df.dropna()
######################################################################################

################SPECIFIC TO GALAXIES###############################
#M31
i_m31= 75 #deg
inclinations=np.array([i_m31,77,i_m31,77,i_m31,75,77.5]) #used i_m31 as no inclination correction is needed for Claude data
interpolated_df = inclination_correction(interpolated_df, np.radians(i_m31), np.radians(inclinations))

data_choices = ['chemin', 'claude']
data_chosen=data_choices[0] #Claude data chosen
interpolated_df.drop(columns=keep_substring_columns(interpolated_df, data_chosen)[1],inplace=True)
################################################################################

interpolated_df = molfrac_to_H2(interpolated_df)


add_temp(0.017e+4,0.5e+4,interpolated_df)
conv_factors=np.array([1, (g_Msun/(cm_pc**2) ), g_Msun/(cm_pc**2), g_Msun/(cm_pc**2), cm_km,
              g_Msun/((s_Myr*1e3)*(cm_pc**2)),1])
interpolated_df = interpolated_df*conv_factors
#interpolated_df= replace_conversion(interpolated_df, 'kpc', 'cm')
interpolated_df= replace_conversion(interpolated_df, 'kms', 'cms')

interpolated_df = vcirc_to_qomega(interpolated_df)
print(interpolated_df)
os.chdir("..")
interpolated_df.to_csv('data_interpolated.csv')