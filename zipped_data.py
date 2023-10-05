import numpy as np
from sympy import *
import pickle
import os
import pandas as pd

current_directory = str(os.getcwd())

params = {}
switch = {}

#opening these files and making them into dictionaries
with open(current_directory+'\parameter_file.in', 'r') as FH:
    for file in FH.readlines():
        line = file.strip()
        try:
            par_name, value = line.split('=')
        except ValueError:
            print("Record: ", line)
            raise Exception(
                "Failed while unpacking. Not enough arguments to supply.")
        try:
            params[par_name] = np.float64(value)
        except ValueError: #required cz of 14/11 in parameter.in file
            num, denom = value.split('/')
            params[par_name] = np.float64(num) / np.float64(denom)

with open(current_directory+'\switches.in', 'r') as FH:
    for file in FH.readlines():
        line = file.strip()
        try:
            sw_name, value = line.split('=')
        except ValueError:
            print("Record: ", line)
            raise Exception(
                "Failed while unpacking. Not enough arguments to supply.")
        switch[sw_name] = value

print('Succesfully read the parameters and switches')

#choose galaxy name in the next line to choose the galaxy
os.chdir(current_directory+'\data')
data = (pd.read_csv('data_interpolated.csv')) 

kpc_r = data.iloc[:, 0].values
# finding total gas density
# adds HI and H2 data based on switch 
if switch['incl_moldat'] == 'Yes': #this is set to 'NO' in switches.in file
    data[data.columns[2]] = data.iloc[:, 2] + data.iloc[:, 3]

data.drop(data.columns[3], axis=1, inplace=True)
data.rename(columns={data.columns[2]: '\sigma_gas'}, inplace=True)
#use this for m51 and 6946
#difference as here we have to mandatorily consider molecular gas density
# kpc_r, dat_sigmatot, dat_sigma, dat_q, dat_omega, dat_sigmasfr, T= data
r = kpc_r.size  # common radius of the interpolated data

#mu= 14/11 set in parameters.in file
data['\sigma_gas']*= (3*params['mu'])/(4-params['mu'])
#####################################################################################################################

#####################################################################################################################
#to apply kennicut-schmidt relation
ks_exp = params['ks_exp']
ks_const = (data['sigma_sfr']/(data['\sigma_gas'])**(ks_exp)).mean()

ks_split = switch['force_kennicut_scmidt'].split() #currently this is set to 'No, sigmadata'
if ks_split[0] == 'Yes':
    if ks_split[1] == 'sigmasfrdata':
        data['\sigma_gas'] = (data['sigma_sfr']/ks_const)**(1/ks_exp)
    else:
        data['sigma_sfr'] = ks_const*(data['\sigma_gas'])**(ks_exp)
#####################################################################################################################
params_df = pd.DataFrame({key: [value] * r for key, value in params.items() if key != 'ks_exp'})
data_params = data.join(params_df)

data_listoftup = list(data_params[data_params.columns[1:]].to_records(index=False))
# zip function makes array of corresponding elements of each array passed into it
data_pass = kpc_r, data_listoftup
# (data_pass[1]).sort()
# (list(data_params[data_params.columns[1:]].to_records(index=False))).sort()
#if data_pass[1] == list(data_params[data_params.columns[1:]].to_records(index=False)): print('You are a God rion')

#dont change this pickle file name
with open(current_directory+ '\data\zip_data.pickle', 'wb') as f:
    pickle.dump(data_pass, f)

print('Succesfully zipped the data and the parameters, and made pickle file')

