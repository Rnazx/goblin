# This file calculates the errors in the observables using the scaling relations. 
# The errors are then saved in a pickle file and a csv file.

print('#####  Error calculation ######')

import numpy as np
import pickle
import os
import pandas as pd
from helper_functions import parameter_read
from data_helpers import *
from datetime import date
import csv #for saving err data as csv file
import icecream as ic

# converting from 2nd unit to 1st
pc_kpc     = 1e3      # number of pc in one kpc
cm_km      = 1e5      # number of cm in one km
s_day      = 24*3600  # number of seconds in one day
s_min      = 60       # number of seconds in one hour
s_hr       = 3600     # number of seconds in one hour
cm_Rsun    = 6.957e10 # solar radius in cm
g_Msun     = 1.989e33 # solar mass in g
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

#########################################################################################
base_path   = os.environ.get('MY_PATH')
galaxy_name = os.environ.get('galaxy_name')

#get parameter values
params = parameter_read(os.path.join(base_path,'inputs','parameter_file.in'))
switch = parameter_read(os.path.join(base_path,'inputs','switches.in'))

current_directory = str(os.getcwd())
os.chdir(os.path.join(base_path,'outputs'))

with open(f'{galaxy_name}output_ca_'+str(params[r'C_\alpha'])+'K_'+str(params[r'K'])+'z_'+
          str(params[r'\zeta'])+'psi_'+str(params[r'\psi'])+'b_'+str(params[r'\beta'])+'.out', 'rb') as f:
                model_f = pickle.load(f)

os.chdir(os.path.join(base_path,'inputs'))

# getting available data and error
os.chdir(os.path.join(base_path, 'data','model_data', f'{galaxy_name}_data'))

# here the units of observables are as used in source papers
# radius may be in kpc or arcsec/arcmin
raw_data_astro_units = pd.read_csv(f'combined_data_{galaxy_name}.csv', skiprows = 1) # needed to obtain vcirc values which isnt present in interpolated_data.csv
err_data_astro_units = pd.read_csv(f'error_combined_{galaxy_name}.csv')
corrections_kpc_deg  = pd.read_csv(f'correction_data_{galaxy_name}.csv', skiprows = 1, index_col = 0)

# data to be removed
corrections_kpc_deg = corrections_kpc_deg.T # taking transpose of corrections df
try:
    data_rem = pd.read_csv(f'removed_data_{galaxy_name}.csv', dtype=str)
except:
    data_rem = []
for d in data_rem:
    remove_data(corrections_kpc_deg, d, False)
    remove_data(raw_data_astro_units, d)
corrections_kpc_deg = corrections_kpc_deg.T # reverting the transpose
#################################################################################################################
# error in distance and inclination

# order of quantities in corrections file: sigma_tot, sigma_HI, sigma_H2, vcirc, sigma_sfr
# distance correction
nDIST_kpc       = corrections_kpc_deg.iloc[-1,0]
err_nDIST_kpc   = corrections_kpc_deg.iloc[-1,2]
oDIST_kpc       = corrections_kpc_deg.iloc[:-1,0].values
err_oDIST_kpc   = corrections_kpc_deg.iloc[:-1,2].values

# inclination correction
nINC_deg        = corrections_kpc_deg.iloc[-1,1] # deg
err_nINC_deg    = corrections_kpc_deg.iloc[-1,3] # deg
oINC_deg        = corrections_kpc_deg.iloc[:-1,1].values # used new_i as no inclination correction is needed for Claude data
err_oINC_deg    = corrections_kpc_deg.iloc[:-1,3].values

# galaxy-wise inc and D for correcting the errors
# should be customised when new data/galaxies are added for error
if galaxy_name == 'm31':
       inc_for_err  = np.radians([77, 77.5])
       dist_for_err = np.array([0.78, 0.78])
elif galaxy_name == 'm33':
       inc_for_err  = np.radians([52, 56])
       dist_for_err = np.array([0.84, 0.84])
elif galaxy_name == 'm51':
       inc_for_err  = np.radians([20])
       dist_for_err = np.array([9.6])
else:
       inc_for_err  = np.radians([30])
       dist_for_err = np.array([5.5])

#################################################################################################################

# raw_data from combined_data.csv is converted to correct units and corrected for distance and inclination
# raw_data_astro_units_kpc implies that radii are in kpc and observables in astro units
raw_data_astro_units_kpc = incl_distance_correction(raw_data_astro_units, distance_new = nDIST_kpc, distance_old = oDIST_kpc,\
                        i_new = np.radians(nINC_deg), i_old = np.radians(oINC_deg))

# obtain slope and intercept for temperature fit
temp_fit = np.genfromtxt(f'temp_{galaxy_name}.csv', skip_header = 1, delimiter = ',')
#################################################################################################################

raw_data_astro_units_kpc = vcirc_to_qomega(raw_data_astro_units_kpc, False) #raw_data contain v_circ also
raw_data_radii_df_kpc    = keep_substring_columns(raw_data_astro_units_kpc, 'r ')[0]

# error data D and i correction
err_data_astro_units_kpc = incl_distance_correction(err_data_astro_units, distance_new = nDIST_kpc, distance_old = dist_for_err,\
                            i_new = np.radians(nINC_deg), i_old = inc_for_err) #correcting errors for D and i
err_radii_df_kpc         = keep_substring_columns(err_data_astro_units_kpc, 'R')[0]

# coarsest_radii_mask = raw_data_radii_df_kpc.isnull().sum().idxmax()

# choosing coarsest radius depending on moldata switch
if switch['incl_moldat'] == 'No':
        # drop sigma_H2 column from raw_data and make new copy
        raw_data_drop_sigmaH2 = raw_data_astro_units_kpc.copy()
        raw_data_drop_sigmaH2 = remove_data(raw_data_drop_sigmaH2, 'sigma_H2')
        radii_df_drop_sigmaH2 = keep_substring_columns(raw_data_drop_sigmaH2, 'r ')[0]
        coarsest_radii_mask   = radii_df_drop_sigmaH2.isnull().sum().idxmax()
        kpc_r                 = radii_df_drop_sigmaH2[coarsest_radii_mask].to_numpy()
else:
        # moldata is included and no need to remove sigma_H2 column
        # proceed to next step 
        coarsest_radii_mask = raw_data_radii_df_kpc.isnull().sum().idxmax()
        kpc_r               = raw_data_radii_df_kpc[coarsest_radii_mask].to_numpy()

interpolated_df_astro_units_kpc = df_interpolation(raw_data_astro_units_kpc,raw_data_radii_df_kpc, kpc_r)

###############################################################################################

# error data length is matched with raw data length by adding nan
max_rows_A = max(len(err_data_astro_units_kpc[col]) for col in err_data_astro_units_kpc.columns)
max_rows_B = max(len(interpolated_df_astro_units_kpc[col]) for col in interpolated_df_astro_units_kpc.columns)

err_data_long = pd.DataFrame(0, index=range(max(max_rows_A,max_rows_B)), columns=err_data_astro_units_kpc.columns)

if max_rows_A > max_rows_B: # err_data has greater length. nan added to interpolated_df
#        print('err_data has greater length. nan added to interpolated_df')
       for i in range(len(interpolated_df_astro_units_kpc.columns)):
                interpolated_df_astro_units_kpc.iloc[abs(max_rows_A-max_rows_B):, i] = np.nan
else:
#        print('interpolated_df has greater length. nan added to err_data')
       for i in range(len(err_data_astro_units_kpc.columns)):
                err_data_col_list       = np.array(err_data_astro_units_kpc.iloc[:,i])
                err_add_nan             = np.concatenate((err_data_col_list, np.full(abs(max_rows_A-max_rows_B), np.nan)))
                err_data_long.iloc[:,i] = err_add_nan

err_radii_df_kpc                    = keep_substring_columns(err_data_long, 'R')[0]
err_interpolated_df_astro_units_kpc = df_interpolation(err_data_long,err_radii_df_kpc, kpc_r)
################################################################################################

# adding sigma_H2 data to interpolated_df for M31 & adding temperature data
interpolated_df_astro_units_kpc = molfrac_to_H2(interpolated_df_astro_units_kpc, False)
add_temp(temp_fit,interpolated_df_astro_units_kpc)

# remove NaN values from interpolated_df and err_interpolated_df
nan_mask                         = np.isnan(interpolated_df_astro_units_kpc)
interpolated_df_astro_units_kpc  = interpolated_df_astro_units_kpc[~(nan_mask.sum(axis=1)>0)]

nan_mask                            = np.isnan(err_interpolated_df_astro_units_kpc)
err_interpolated_df_astro_units_kpc = err_interpolated_df_astro_units_kpc[~(nan_mask.sum(axis=1)>0)]

#########################################################################################

# difference in length between model_f and interpolated_df
if len(interpolated_df_astro_units_kpc.iloc[:,0]) != len(model_f[0]):
        if len(interpolated_df_astro_units_kpc.iloc[:,0]) > len(model_f[0]):
                interpolated_df_astro_units_kpc = interpolated_df_astro_units_kpc.iloc[:len(model_f[0]), :]
        else:
                for i in range(len(model_f)):
                      # Initialize an empty list to store the modified elements
                        new_model_f = []

                        # Loop over each element in model_f
                        for i in range(len(model_f)):
                        # Slice model_f[i] to match the length of interpolated_df.iloc[:,0]
                        # and append it to new_model_f
                                new_model_f.append(model_f[i][:len(interpolated_df_astro_units_kpc.iloc[:,0])])

                        # Convert new_model_f to a tuple
                        model_f = tuple(new_model_f)
#########################################################################################

# unit conversions for data
# Changed for m51 and ngc6946
if galaxy_name == 'm31' or galaxy_name == 'm33':
    m_to_gconv = 1e3
else:
    m_to_gconv = 1

# Order followed: kpc_r,  sigma_tot, sigma_HI, sigma_H2, q, \Omega,  sigma_sfr, T, extras
if galaxy_name   == 'm31': # since there is extra molfrac list here at 2nd last element
    conv_factors = np.array([1, (g_Msun/(cm_pc**2) ), g_Msun/(cm_pc**2), g_Msun/(cm_pc**2), 1,cm_km/cm_kpc,cm_km,
            g_Msun/((s_Myr*m_to_gconv)*(cm_pc**2)),1,1])
else:
    conv_factors = np.array([1, (g_Msun/(cm_pc**2) ), g_Msun/(cm_pc**2), g_Msun/(cm_pc**2), 1,cm_km/cm_kpc,cm_km,
            g_Msun/((s_Myr*m_to_gconv)*(cm_pc**2)),1])

interpolated_df_cgs_kpc = interpolated_df_astro_units_kpc*conv_factors

# unit conversions for error data
if galaxy_name   == 'm31':
        conv_factors=np.array([1,cm_km,1])
elif galaxy_name == 'm33':
        conv_factors = np.array([1,cm_km,g_Msun/(cm_pc**2)])
else:
        conv_factors = np.array([1,cm_km])

err_interpolated_df_cgs = err_interpolated_df_astro_units_kpc*conv_factors

# in data folder, save the interpolated data in cgs units
os.chdir(os.path.join(base_path,'data'))
err_interpolated_df_cgs.to_csv(f'error_interpolated_{galaxy_name}.csv', index = False)
# return to base path
os.chdir(base_path)
#########################################################################################

# unit conversions for all functions
oDIST_cm     = oDIST_kpc*cm_kpc*(10**3) # Mpc to cm conversion
nDIST_cm     = nDIST_kpc*cm_kpc*(10**3)
        
oINC_rad     = np.radians(oINC_deg)
nINC_rad     = np.radians(nINC_deg)

err_nDIST_cm = err_nDIST_kpc*cm_kpc*(10**3) # Mpc to cm conversion
err_oDIST_cm = err_oDIST_kpc*cm_kpc*(10**3)

err_nINC_rad = np.radians(err_nINC_deg)
err_oINC_rad = np.radians(err_oINC_deg)

kpc_r        = interpolated_df_cgs_kpc['kpc_r']
cm_r         = kpc_r*cm_kpc # kpc to cm conversion
#########################################################################################

# obtaining data with no inc & D correction
# to be used in relative error calculation
sigmatot_reverted_cgs = interpolated_df_cgs_kpc.iloc[:,1]*(np.cos(oINC_rad[0])/np.cos(nINC_rad))
vcirc_reverted_cgs    = interpolated_df_cgs_kpc.iloc[:,6]*(np.sin(oINC_rad[-2])/np.sin(nINC_rad))*(nDIST_cm/oDIST_cm[-2])
sigmaHI_reverted_cgs  = interpolated_df_cgs_kpc.iloc[:,2]*(np.cos(oINC_rad[1])/np.cos(nINC_rad))

if galaxy_name == 'm31':
        # indexing different for m31 as molfrac is used to calculate sigma_H2
        # molfrac has -1 index, sigma_sfr has -2 index 
        # sigma_HI has same index as other galaxies, but has different col name due to Claude/Chemin switch
        molfrac_reverted_cgs  = interpolated_df_cgs_kpc['molfrac']*(np.cos(oINC_rad[-1])/np.cos(nINC_rad))
        sigmaSFR_reverted_cgs = interpolated_df_cgs_kpc['sigma_sfr']*(np.cos(oINC_rad[-2])/np.cos(nINC_rad))

else:
        sigmaH2_reverted_cgs  = interpolated_df_cgs_kpc['sigma_H2']*(np.cos(oINC_rad[2])/np.cos(nINC_rad))
        sigmaSFR_reverted_cgs = interpolated_df_cgs_kpc['sigma_sfr']*(np.cos(oINC_rad[-1])/np.cos(nINC_rad))


#########################################################################################

# Functions for error propagation

# error in omega
def err_omega(omega, vcirc, sigma_v):
        
        sigma_v = [sigma_v.iloc[i] for i in range(len(kpc_r))]

        if galaxy_name == 'm31':
                u = 2 # different as molfrac used for sigma_H2 is at -1 index
        else:
                u = 3
        
        # Equation C9 in appendix of Nazareth+24
        term1 = [(oDIST_cm[u]*np.sin(oINC_rad[u])*sigma_v[i])/(nDIST_cm*cm_r.iloc[i]*np.sin(nINC_rad)) for i in range(len(kpc_r))]
        term2 = [(vcirc.iloc[i]*oDIST_cm[u]*np.cos(oINC_rad[u])*err_oINC_rad[u])/((np.sin(nINC_rad))*cm_r.iloc[i]*nDIST_cm) for i in range(len(kpc_r))]       
        term3 = [(vcirc.iloc[i]*oDIST_cm[u]*np.sin(oINC_rad[u])*np.cos(nINC_rad)*err_nINC_rad)/(((np.sin(nINC_rad))**2)*cm_r.iloc[i]*nDIST_cm) for i in range(len(kpc_r))]
        term4 = [(vcirc.iloc[i]*np.sin(oINC_rad[u])*err_oDIST_cm[u])/((np.sin(nINC_rad))*cm_r.iloc[i]*nDIST_cm) for i in range(len(kpc_r))]
        term5 = [(vcirc.iloc[i]*np.sin(oINC_rad[u])*oDIST_cm[u]*err_nDIST_cm)/((np.sin(nINC_rad))*cm_r.iloc[i]*(nDIST_cm**2)) for i in range(len(kpc_r))]
        
        err_omega = [np.sqrt(term1[i]**2 +term2[i]**2 + term3[i]**2 + term4[i]**2 + term5[i]**2) for i in range(len(kpc_r))]
        
        rel_err_omega = err_omega/omega
        return rel_err_omega

# error in sigma_HI
def err_sigmaHI(sigmaHI_corrected, sigmaHI_not_corrected, percent_sigmaHI_err): # sigmaHI_err = 6 percent, sigmaHI = columns_as_arrays[2]
        
        if galaxy_name == 'm31' or galaxy_name == 'm33':
                u = 2
        else:
                u = 1

        # Equation C10 in appendix of Nazareth+24
        sigmaHI_err = sigmaHI_not_corrected*percent_sigmaHI_err
        term1       = [(sigmaHI_err.iloc[i]*np.cos(nINC_rad))/np.cos(oINC_rad[u]) for i in range(len(kpc_r))]        # print('1',term1)
        term2       = [(sigmaHI_not_corrected.iloc[i]*np.sin(nINC_rad)*err_nINC_rad)/np.cos(oINC_rad[u]) for i in range(len(kpc_r))]        # print('2',term2)
        term3       = [(sigmaHI_not_corrected.iloc[i]*np.cos(nINC_rad)*err_oINC_rad[u]*np.sin(oINC_rad[u]))/((np.cos(oINC_rad[2]))**2) for i in range(len(kpc_r))]        # print('3',term3)
        
        err_sigmaHI     = [np.sqrt(term1[i]**2 + term2[i]**2 + term3[i]**2) for i in range(len(kpc_r))]        
        rel_err_sigmaHI = [err_sigmaHI[i]/sigmaHI_corrected.iloc[i] for i in range(len(kpc_r))]
        
        return err_sigmaHI, rel_err_sigmaHI

# no need of corrected sigmas here as relative error isnt required
def err_sigmaH2(sigmaHI_not_corrected, molfrac_or_sigmaH2, percent_sigmaHI_err, molfrac_err, percent_sigmaH2_err): 
        
        if switch['incl_moldat'] == 'No':
                # set err_sigmaH2 to 0
                err_sigmaH2 = [0 for i in range(len(kpc_r))]
        else: # molecular gas is included
                if galaxy_name == 'm31': # sigma_H2 error is calculated using molfrac data
                        molfrac         = molfrac_or_sigmaH2
                        # error in molfrac due to uncertainty in inclination, distance and molfrac data
                        molfrac_err_tot = [np.sqrt(((molfrac[i]*np.sin(nINC_rad)*err_nINC_rad)/(np.cos(oINC_rad[-1])))**2 + 
                                                   ((molfrac[i]*np.cos(nINC_rad)*err_oINC_rad[-1]*np.sin(oINC_rad[-1]))/(np.cos(oINC_rad[-1]))**2)**2 + 
                                                   (molfrac_err[i]*np.cos(nINC_rad))/np.cos(oINC_rad[-1])**2) for i in range(len(kpc_r))]
                        sigmaHI_err     = sigmaHI_not_corrected*percent_sigmaHI_err
                        
                        # Equation C11 in appendix of Nazareth+24
                        term1       = [(sigmaHI_err.iloc[i]*molfrac.iloc[i])/(1-molfrac.iloc[i]) for i in range(len(kpc_r))]                        
                        term2       = [(sigmaHI_not_corrected.iloc[i]*molfrac_err_tot[i])/((1-molfrac.iloc[i])**2) for i in range(len(kpc_r))]                        
                        err_sigmaH2 = [np.sqrt(term1[i]**2 + term2[i]**2) for i in range(len(kpc_r))] 
                else:
                        if galaxy_name == 'm33': # there is data for sigma_H2 error
                                u = 3 # extra column due to Koch data
                                sigmaH2_not_corrected = molfrac_or_sigmaH2
                                sigmaH2_err           = err_interpolated_df_cgs['error sigma_H2']

                        else: # M51 & NGC 6946: no data for sigma_H2 error, assume 6% error 
                                u = 2
                                sigmaH2_not_corrected = molfrac_or_sigmaH2
                                percent_sigmaH2_err   = 0.06
                                sigmaH2_err           = sigmaH2_not_corrected*percent_sigmaH2_err
                        
                        # Equation C10 in appendix of Nazareth+24
                        term1 = [(sigmaH2_err.iloc[i]*np.cos(nINC_rad))/np.cos(oINC_rad[u]) for i in range(len(kpc_r))]
                        term2 = [(sigmaH2_not_corrected.iloc[i]*np.sin(nINC_rad)*err_nINC_rad)/np.cos(oINC_rad[u]) for i in range(len(kpc_r))]
                        term3 = [(sigmaH2_not_corrected.iloc[i]*np.cos(nINC_rad)*err_oINC_rad[u]*np.sin(oINC_rad[u]))/(np.cos(oINC_rad[u]))**2 for i in range(len(kpc_r))]
                        
                        err_sigmaH2 = [np.sqrt(term1[i]**2 + term2[i]**2 + term3[i]**2) for i in range(len(kpc_r))]
        return err_sigmaH2

# error in sigma_gas
# combine sigma_HI and sigma_H2 errors
def err_sigmagas(sigmaHI_corr, sigmaH2_corr, sigmaHI_no_corr, sigmaH2_or_molfrac_no_corr, err_mu, err_mu_prime):
        
        mu       = params['mu']       # 14/11
        mu_prime = params['mu_prime'] # 2.4

        # to choose corrected sigmagas
        if galaxy_name == 'm31':
                molfrac_no_corr = sigmaH2_or_molfrac_no_corr
                sigmaH2_err     = err_sigmaH2(sigmaHI_no_corr, molfrac_no_corr, 0.06, err_interpolated_df_astro_units_kpc['error molfrac'], 0.06)
        else:
                sigmaH2_no_corr = sigmaH2_or_molfrac_no_corr
                sigmaH2_err     = err_sigmaH2(sigmaHI_no_corr, sigmaH2_no_corr, 0.06, 0, 0.06)
               
        sigmaHI_err = err_sigmaHI(sigmaHI_corr, sigmaHI_no_corr, 0.06)[0]

        # if moldat is included, sigma_gas equation is modified
        if switch['incl_moldat'] == 'Yes':
                sigmagas_corr  = (3*mu/(4-mu))*sigmaHI_corr + (mu_prime/(4-mu_prime))*sigmaH2_corr #columns_as_arrays[2]+columns_as_arrays[3]
                if galaxy_name == 'm31':
                        sigmaH2_no_corr = sigmaHI_no_corr*(molfrac_no_corr/(1-molfrac_no_corr))
        else:
                sigmagas_corr  = sigmaHI_corr

        mu_err           = [err_mu*mu for i in range(len(kpc_r))]
        mu_prime_err     = [err_mu_prime*mu_prime for i in range(len(kpc_r))]

        # Equation C12 in appendix of Nazareth+24
        term1            = [(sigmaHI_err[i]*(3*mu/(4-mu))) for i in range(len(kpc_r))]
        term2            = [(sigmaHI_no_corr.iloc[i]*(12*mu_err[i]/(4-mu)**2)) for i in range(len(kpc_r))]
        term3            = [(sigmaH2_err[i]*(mu_prime/(4-mu_prime))) for i in range(len(kpc_r))]
        if switch['incl_moldat'] == 'Yes':
                term4            = [(sigmaH2_no_corr.iloc[i]*(4*mu_prime_err[i]/(4-mu_prime)**2)) for i in range(len(kpc_r))]
        else:
                term4            = [0 for i in range(len(kpc_r))]
        
        err_sigmagas     = [np.sqrt((term1[i]**2 + term2[i]**2 + term3[i]**2 + term4[i]**2)) for i in range(len(kpc_r))]
        rel_err_sigmagas = [err_sigmagas[i]/sigmagas_corr.iloc[i] for i in range(len(kpc_r))]
        return rel_err_sigmagas

# error in sigma_SFR
def err_sigmaSFR(sigmaSFR_corrected, sigmaSFR_not_corrected, percent_sigmaSFR_err): # arbitrarily assuming 10% error in sigma_SFR
        
        sigmaSFR_err   = sigmaSFR_not_corrected*percent_sigmaSFR_err
        if galaxy_name == 'm31':
                u = -2
        else:
                u = -1
        
        # Equation C10 in appendix of Nazareth+24
        term1            = [(sigmaSFR_err.iloc[i]*np.cos(nINC_rad))/np.cos(oINC_rad[u]) for i in range(len(kpc_r))]
        term2            = [(sigmaSFR_not_corrected.iloc[i]*np.sin(nINC_rad)*err_nINC_rad)/np.cos(oINC_rad[u]) for i in range(len(kpc_r))]
        term3            = [(sigmaSFR_not_corrected.iloc[i]*np.cos(nINC_rad)*err_oINC_rad[u]*np.sin(oINC_rad[u]))/(np.cos(oINC_rad[u]))**2 for i in range(len(kpc_r))]
        
        err_sigmaSFR     = [np.sqrt(term1[i]**2 + term2[i]**2 + term3[i]**2) for i in range(len(kpc_r))]
        rel_err_sigmaSFR = [err_sigmaSFR[i]/sigmaSFR_corrected.iloc[i] for i in range(len(kpc_r))]
        return rel_err_sigmaSFR

# error in sigma_tot
def err_sigmatot(sigmatot_corrected, sigmatot_not_corrected, percent_sigmatot_err):
        
        # Equation C13 in appendix of Nazareth+24
        if galaxy_name == 'm33': # does not use percent_sigmatot_error
                # use formula from Kam+15/17 for error propagation
                gamma     = 0.52*g_Msun/((cm_pc)**2) # converting to cgs units
                err_gamma = 0.1*g_Msun/((cm_pc)**2)  # converting to cgs units

                # values as in Kam+15
                mu0       = 18.01 # mag arsec^2
                sigma_mu0 = 0.03  # mag arsec^2
                kpc_Rd    = 1.82  # kpc
                cm_Rd     = kpc_Rd*cm_kpc # cm
                cm_Rd_err = 0.02*cm_kpc   # cm
                c36       = 24.8
                mu        = [(mu0+(1.10857*(cm_r.iloc[i]/cm_Rd))) for i in range(len(kpc_r))] # Kam+15

                term1_sigma_mu = [sigma_mu0 for i in range(len(kpc_r))]
                term2_sigma_mu = [(1.10857*cm_r.iloc[i]*cm_Rd_err/cm_Rd**2) for i in range(len(kpc_r))]
                term3_sigma_mu = [((1.10857)/cm_Rd)*cm_r.iloc[i]*np.sqrt(((err_nDIST_cm)/(nDIST_cm))**2 + ((nDIST_cm*err_oDIST_cm[0])/(oDIST_cm[0])**2)**2) for i in range(len(kpc_r))]
                
                # no squaring as mu0 is an additive constant
                sigma_mu       = [term1_sigma_mu[i] + np.sqrt((term2_sigma_mu[i])**2 + (term3_sigma_mu[i])**2) for i in range(len(kpc_r))]
                a              = [10**(-0.4*(mu[i]-c36)) for i in range(len(kpc_r))]
                da_dt          = [(np.log(10))*(-0.4)*a[i] for i in range(len(kpc_r))]
                sigma_a        = [da_dt[i]*sigma_mu[i] for i in range(len(kpc_r))]
                term1          = [err_gamma*a[i] for i in range(len(kpc_r))]
                term2          = [gamma*sigma_a[i] for i in range(len(kpc_r))]
                err_sigmatot   = [np.sqrt(term1[i]**2 + term2[i]**2) for i in range(len(kpc_r))]
                
        else:
                sigmatot_err   = percent_sigmatot_err*sigmatot_not_corrected
                term1          = [(sigmatot_err.iloc[i]*np.cos(nINC_rad)/np.cos(oINC_rad[0])) for i in range(len(kpc_r))]
                term2          = [(sigmatot_not_corrected.iloc[i]*np.sin(nINC_rad)*err_nINC_rad/np.cos(oINC_rad[0])) for i in range(len(kpc_r))]
                term3          = [(sigmatot_not_corrected.iloc[i]*np.cos(nINC_rad)*err_oINC_rad[0]*np.sin(oINC_rad[0]))/((np.cos(oINC_rad[0]))**2) for i in range(len(kpc_r))]
                err_sigmatot   = [np.sqrt(term1[i]**2 + term2[i]**2 + term3[i]**2) for i in range(len(kpc_r))]
        
        rel_err_sigmatot       = [err_sigmatot[i]/sigmatot_corrected.iloc[i] for i in range(len(kpc_r))]
        return rel_err_sigmatot

# error in temperature and q
def err_T_q(T_corrected, q_corrected):

        err_T     = np.std(T_corrected)
        err_q     = np.std(q_corrected)

        rel_err_T = [err_T/T_corrected.iloc[i] for i in range(len(kpc_r))]
        rel_err_q = [err_q/q_corrected.iloc[i] for i in range(len(kpc_r))]
        
        return rel_err_T, rel_err_q
#################################################################################################################

# no D and i correction done
hregs = ['subsonic', 'supersonic']
for hreg in hregs:
        os.chdir(os.path.join(base_path,'inputs'))
        exps = np.load(f'scal_exponents_{hreg}.npy')
        r = kpc_r.size

        # calculate relative error in omega
        rel_err_omega = err_omega(interpolated_df_cgs_kpc['\Omega'],vcirc_reverted_cgs,err_interpolated_df_cgs['error vcirc kms'])

        # calculate relative error in sigma_tot
        if galaxy_name           == 'm33': # Kam+15/17 prescription
                rel_err_sigmatot = err_sigmatot(interpolated_df_cgs_kpc['sigma_tot_up52'],sigmatot_reverted_cgs, None)
        else: # 10% error in sigma_tot
                rel_err_sigmatot = err_sigmatot(interpolated_df_cgs_kpc['sigma_tot']     ,sigmatot_reverted_cgs, 0.1)
        
        # calculate relative error in sigma_gas
        if galaxy_name           == 'm31':
                rel_err_sigmagas = err_sigmagas(interpolated_df_cgs_kpc['sigma_HI_claude'],interpolated_df_cgs_kpc['sigma_H2'], sigmaHI_reverted_cgs, molfrac_reverted_cgs, 0.1, 0.1)
        else:
                rel_err_sigmagas = err_sigmagas(interpolated_df_cgs_kpc['sigma_HI'],       interpolated_df_cgs_kpc['sigma_H2'], sigmaHI_reverted_cgs, sigmaH2_reverted_cgs, 0.1, 0.1)
        
        # calculate relative error in sigma_SFR
        rel_err_sigmasfr    = err_sigmaSFR(interpolated_df_cgs_kpc['sigma_sfr'], sigmaSFR_reverted_cgs, 0.1)
        
        # calculate relative error in T and q
        rel_err_T, rel_err_q = err_T_q(interpolated_df_cgs_kpc['T'],interpolated_df_cgs_kpc['q'])
        
        # combine all relative errors 
        rel_err = np.array([rel_err_q, rel_err_omega, rel_err_sigmagas, rel_err_sigmatot,rel_err_sigmasfr, rel_err_T])

        # calculate error in quantities
        relerr_quan    = np.sqrt(np.matmul(exps**2,rel_err**2))
        err_quantities = model_f[1:]*relerr_quan

        # inputting errors into a pickle file
        os.chdir(os.path.join(base_path,'outputs'))
        # load errors in super- and sub-sonic regimes for later comparison
        with open(f'errors_{hreg}.out', 'wb') as f:
                pickle.dump(err_quantities, f)

# saving relerr_quan to a csv file
save_files_dir = os.path.join(base_path + r'\data\supplementary_data\{}'.format(galaxy_name))
filename       = r'\{}_rel_err_observables_moldat_{},taue,z_{},psi_{},ca_{},beta_{},A_{}.csv'.format(galaxy_name,switch['incl_moldat'],params[r'\zeta'],params[r'\psi'],
                                params[r'C_\alpha'],params[r'\beta'],params['A'])

os.chdir(save_files_dir)

rel_err_transpose = list(zip(*rel_err))
column_names      = ['rel_err_q', 'rel_err_omega', 'rel_err_sigmagas', 'rel_err_sigmatot','rel_err_sigmasfr', 'rel_err_T']

# Writing to the file
with open(save_files_dir+filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(column_names)
    csvwriter.writerows(rel_err_transpose)

print('Found the errors from the scaling relations')
