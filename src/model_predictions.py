# This file is used to calculate the model predictions for the given galaxy.
# Actual values for magnetic and kinematic properties are calculated here
if __name__ == '__main__': 
    print('#####  Model predictions #####')

from helper_functions import datamaker, root_finder, exp_analytical_data, parameter_read
import numpy as np
from sympy import *
import pickle
import os
import pandas as pd

#reading the parameters
base_path   = os.environ.get('MY_PATH')
galaxy_name = os.environ.get('galaxy_name')

# variables introduced to customise the scale height and velocity dispersion values used in the model
# choose to use datamaker fn or actual velocity dispersion data for u_f
# u_data_choose = 'data' # 'data' or 'datamaker' 
# scale height customisation 
# h_data_choose = 'root_find' # 'root_find' or 'exponential'

def h_exp(kpc_r):
    # required constants
    deg_rad    = 180e0/np.pi
    arcmin_deg = 60e0

    # Hyper-LEDA values
    log_d25_arcmin_paper2 = [3.25, 2.79, 2.14, 2.06]
    d25_arcmin_paper2     = 0.1*(10**np.array(log_d25_arcmin_paper2))
    r_25_arcmin_paper2    = d25_arcmin_paper2/2
    dist_Mpc_paper2       = [0.78, 0.84, 8.5, 7.72]
    # convert arcmin to radius in kpc using distance to these galaxies
    r_25_kpc_paper2       = [r*dist_Mpc_paper2[i]*1000/(arcmin_deg*deg_rad) for i,r in enumerate(r_25_arcmin_paper2)]

    if galaxy_name   == 'm31':
        h_scaled      = 0.18*np.exp(kpc_r/((10/16)*r_25_kpc_paper2[0])) # in kpc
    elif galaxy_name == 'm33':
        h_scaled      = 0.18*np.exp(kpc_r/((10/16)*r_25_kpc_paper2[1])) # in kpc
    elif galaxy_name == 'm51':
        h_scaled      = 0.18*np.exp(kpc_r/((10/16)*r_25_kpc_paper2[2])) # in kpc
    else:
        h_scaled      = 0.18*np.exp(kpc_r/((10/16)*r_25_kpc_paper2[3])) # in kpc

    # convert kpc to cm
    h_scaled = h_scaled*3.086e+21
    return h_scaled

######################################################################################################################################

current_directory = str(os.getcwd())

base_path   = os.environ.get('MY_PATH')
galaxy_name = os.environ.get('galaxy_name')

params = parameter_read(os.path.join(base_path,'inputs','parameter_file.in'))
switch = parameter_read(os.path.join(base_path,'inputs','switches.in'))

os.chdir(os.path.join(base_path,'inputs'))

with open('zip_data.in', 'rb') as f:
    kpc_r, data_pass = pickle.load(f)

# extracting the expressions
os.chdir(os.path.join(base_path,'expressions'))

with open('turb_exp.pickle', 'rb') as f:
    hg, h_vdisp, rho, nu, u, l, taue, taur, alphak1, alphak2, alphak3 = pickle.load(f)

with open('mag_exp.pickle', 'rb') as f:
    biso, bani, Bbar, tanpb, tanpB, Beq, eta, cs_exp, Dk, Dc = pickle.load(f)

os.chdir(os.path.join(base_path,'data'))
data  = (pd.read_csv('data_interpolated_{}.csv'.format(galaxy_name))) 

os.chdir(current_directory)
cs_f     = exp_analytical_data(cs_exp, np.array(data_pass))
print(cs_f)
if switch['incl_moldat'] == 'Yes':
    S_g = (3*params['mu']/(4-params['mu']))*data.iloc[:, 2] + (params['mu_prime']/(4-params['mu_prime']))*data.iloc[:, 3]
    cs_f = np.sqrt(np.array(((3*params['mu']/(4-params['mu']))*data.iloc[:, 2]*(cs_f)**2 + (params['mu_prime']/(4-params['mu_prime']))*data.iloc[:, 3]*(cs_f/10)**2)/S_g, dtype=np.float64))
print(cs_f)


# obtain 3D velocity dispersion data
vdisp_df = pd.read_csv(os.path.join(base_path, 'data','supplementary_data', f'{galaxy_name}',f'{galaxy_name}_veldisp_ip.csv'))
vdisp    = vdisp_df["v disp"].values # in cgs units

h_init_trys = [1e+15,1e+25,1e+25]
for i,hi in enumerate(h_init_trys):
    try:
        if switch['u'] != 'datamaker':
            h_f = datamaker(h_vdisp, data_pass, np.ones(len(vdisp)),None, None, vdisp)  
        else:
            if switch['h'] == 'root_find': #'root_find' or 'exponential'
                if __name__ == '__main__': 
                    print('Try {} for initial guess of h as {:e} cm'.format(i,np.round(hi)))
                
                h_f = root_finder(exp_analytical_data(hg, data_pass,cs_f), hi)
                # print(h_f)
                
                if __name__ == '__main__': 
                    print('Root found succesfully')
            else:
                h_f = h_exp(kpc_r)

        l_f = datamaker(l, data_pass, h_f,None,None,None,None,cs_f)
        # print(l_f)

        # choose to use datamaker fn or actual velocity dispersion data for u_f
        if switch['u'] == 'datamaker':
            u_f = datamaker(u, data_pass, h_f,None,None,None,None,cs_f)
        else: # call the velocity dispersion data from supplemetary data folder in the data folder
            u_f = vdisp


        taue_f = datamaker(taue, data_pass, h_f, None, None, u_f, l_f,cs_f)
        taur_f = datamaker(taur, data_pass, h_f, None, None, u_f, l_f,cs_f)
        if switch['tau']=='taue':
            tau_f = taue_f 
        elif switch['tau']=='taur':
            tau_f = taur_f
        else:
            tau_f = np.minimum(taue_f, taur_f)  

        omega  = Symbol('\Omega')
        kalpha = Symbol('K_alpha')
        calpha = Symbol('C_alpha')

        omt = datamaker(omega, data_pass, h_f, tau_f, None, u_f, l_f,cs_f)*tau_f
        kah = datamaker(kalpha/calpha, data_pass, h_f, tau_f, None, u_f, l_f,cs_f)*(h_f/(tau_f*u_f))

        alphak_f = []

        for i in range(len(omt)):
            if min(1, kah[i]) >= omt[i]:
                alpha_k = alphak1
            elif min(omt[i], kah[i]) >= 1:
                alpha_k = alphak2
            else:
                alpha_k = alphak3
            alphak_f.append(datamaker(alpha_k, [data_pass[i]], np.array(
                [h_f[i]]), np.array([tau_f[i]]), None, u_f, l_f,cs_f)[0])

        alphak_f = np.array(alphak_f)


        biso_f = datamaker(biso, data_pass, h_f, tau_f, None, u_f, l_f,cs_f)
        bani_f = datamaker(bani, data_pass, h_f, tau_f, None, u_f, l_f,cs_f)

        dkdc_f  = datamaker((Dk/Dc), data_pass, h_f, tau_f, alphak_f, u_f, l_f,cs_f)
        alpham_f = alphak_f*((1/dkdc_f)-1)

        Bbar_f = datamaker(Bbar, data_pass, h_f, tau_f, alphak_f, u_f, l_f,cs_f)

        tanpB_f = datamaker(tanpB, data_pass, h_f, tau_f, None, u_f, l_f,cs_f)
        tanpb_f = datamaker(tanpb, data_pass, h_f, tau_f, None, u_f, l_f,cs_f)
        mag_obs = kpc_r, h_f, l_f, u_f, np.float64(cs_f), alphak_f, taue_f, taur_f, biso_f, bani_f, Bbar_f, tanpB_f, tanpb_f , dkdc_f #, alpham_f, omt, kah
        os.chdir(os.path.join(base_path,'outputs'))

        with open(f'{galaxy_name}output_ca_'+str(params[r'C_\alpha'])+'K_'+str(params[r'K'])+'z_'+str(params[r'\zeta'])+'psi_'+str(params[r'\psi'])+'b_'+str(params[r'\beta'])+'.out', 'wb') as f:
            pickle.dump(mag_obs, f)
            break

    except Exception as e:
        print(e)
        continue

else: 
    print('*************************************************************************************')
    print('Please change the values of the initial guesses')
    print('*************************************************************************************')

if __name__ == '__main__': 
    print('#####  Model prediction done #####')

