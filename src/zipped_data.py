# This file is used to zip the data and parameters into a pickle file, 
# which is then used in the main code to run the model.

import matplotlib

print('#####  Zipped data file running #####')

import numpy as np
from sympy import *
import pickle
import os
import pandas as pd
from helper_functions import parameter_read
import matplotlib.pyplot as plt

galaxy_name = os.environ.get('galaxy_name')

current_directory = str(os.getcwd())
base_path         = os.environ.get('MY_PATH')

params = parameter_read(os.path.join(base_path,'inputs','parameter_file.in'))
switch = parameter_read(os.path.join(base_path,'inputs','switches.in'))

if __name__ == '__main__': 
    print('Succesfully read the parameters and switches')

os.chdir(os.path.join(base_path,'data'))
data  = (pd.read_csv('data_interpolated_{}.csv'.format(galaxy_name))) 
kpc_r = data.iloc[:, 0].values

# finding total gas density
# adds HI and H2 data based on switch 
if switch['incl_moldat'] == 'Yes': 
    data[data.columns[2]] = (3*params['mu']/(4-params['mu']))*data.iloc[:, 2] + (params['mu_prime']/(4-params['mu_prime']))*data.iloc[:, 3]
else:
    data[data.columns[2]] = (3*params['mu']/(4-params['mu']))*data.iloc[:, 2] 

data.drop(data.columns[3], axis=1, inplace=True)

# order of data: kpc_r, dat_sigmatot, dat_sigma, dat_q, dat_omega, dat_sigmasfr, T= data
data.rename(columns={data.columns[2]: '\sigma_gas'}, inplace=True)

# saving ratio of gas to stellar surface density
ratio = data[data.columns[2]]/data[data.columns[1]]

os.chdir(os.path.join(base_path,'data','supplementary_data'))
# so that new directories wont be made when the directory name is imported from this file
if __name__ == '__main__': 
    try:
        os.makedirs('gas_stars ratio')
        os.chdir('gas_stars ratio')
    except FileExistsError:
        # Handle the case where the directory already exists
        # print(f"The directory 'gas_stars ratio' already exists, ratio of gas to stellar surface density saved in it.")
        os.chdir('gas_stars ratio')
        #anything in this folder before will be re-written
    except OSError as e:
        # Handle other OSError exceptions if they occur
        print(f"An error occurred while creating the directory: {e}")

# plot ratio vs r
plt.figure(figsize=(11, 8))
# axis label size
plt.rcParams.update({'font.size': 25})
# fonts and ticks
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
# major and minor ticks on
plt.minorticks_on()
plt.tick_params(axis='both', which='major', direction='in', length=10, width=1)
plt.tick_params(axis='both', which='minor', direction='in', length=5, width=1)
# major and minor ticks on top and right
plt.tick_params(axis='both', which='both', top=True, right=True)

plt.plot(kpc_r, ratio, marker = 'o', markersize = 4, label = 'Include moldat = {}'.format(switch['incl_moldat']))
if switch['incl_moldat'] == 'Yes':
    plt.title('{}: Molecular gas included'.format(galaxy_name.upper()), fontsize = 25, fontweight = 'bold')
else:
    plt.title('{}: Molecular gas excluded'.format(galaxy_name.upper()), fontsize = 25, fontweight = 'bold')
plt.xlabel(r'r [kpc]')
plt.ylabel(r'$\Sigma/\Sigma_\mathrm{tot}$')
plt.legend(fontsize = 20)
plt.grid(True)
plt.savefig('{}, Moldat = {}'.format(galaxy_name, switch['incl_moldat']))

# save r vs ratio data as csv in the same folder
ratio_data = pd.DataFrame({'r': kpc_r, 'ratio': ratio})
ratio_data.to_csv('ratio_data_{}_moldat{}.csv'.format(galaxy_name, switch['incl_moldat']), index=False)

# return to the data folder
os.chdir(os.path.join(base_path,'data'))

r = kpc_r.size  # common radius of the interpolated data

# to apply kennicut-schmidt relation
ks_exp   = params['ks_exp']
ks_const = (data.iloc[:, -2]/(data['\sigma_gas'])**(ks_exp)).mean()

ks_split = switch['force_kennicut_scmidt'].split(',') #currently this is set to 'No, sigmadata'
if ks_split[0] == 'Yes':
    if ks_split[1] == 'sigmasfrdata':
        data['\sigma_gas'] = (data.iloc[:, -2]/ks_const)**(1/ks_exp)
    else:
        data.iloc[:, -2] = ks_const*(data['\sigma_gas'])**(ks_exp)

params_df      = pd.DataFrame({key: [value] * r for key, value in params.items() if key != 'ks_exp'})#and key != 'mu_prime'
data_params    = data.join(params_df)
data_listoftup = list(data_params[data_params.columns[1:]].to_records(index=False))
data_pass      = kpc_r, data_listoftup

# save kpc_r as a txt file in data folder
os.chdir(os.path.join(base_path,'data'))
# make a folder for kpc_r if it doesn't exist already
if not os.path.exists('kpc_r'):
    os.mkdir('kpc_r')
os.chdir('kpc_r')
np.savetxt('kpc_r_{}.txt'.format(galaxy_name), kpc_r)

with open(os.path.join(base_path,'inputs','zip_data.in'), 'wb') as f:
    pickle.dump(data_pass, f)

print('Succesfully zipped the data and the parameters, and made pickle file')
