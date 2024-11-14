# This file is used to plot the data from the output files

print('#####  Plotting starts #####')

import matplotlib
from helper_functions import datamaker, parameter_read, analytical_pitch_angle_integrator, plot_rectangle, fill_error, new_pitch_angle_integrator
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import pickle
import os
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import griddata
import sys
from datetime import date
import csv 
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)
import matplotlib.patches as patches
import pandas as pd
from icecream import ic

today = date.today() # to set the date for the folder name

# Defining the Observables
q        = Symbol('q')
omega    = Symbol('\Omega')
sigma    = Symbol('\Sigma')
sigmatot = Symbol('Sigma_tot')
sigmasfr = Symbol('Sigma_SFR')
T        = Symbol('T')

# Defining the Constants
gamma = Symbol('gamma')
boltz = Symbol('k_B')
mu    = Symbol('mu')
mh    = Symbol('m_H')

# Defining the general parameters
u   = Symbol('u')
tau = Symbol('tau')
l   = Symbol('l')
h   = Symbol('h')

# conversion factors
pc_kpc     = 1e3  # number of pc in one kpc
cm_km      = 1e5  # number of cm in one km
cm_kpc     = 3.086e+21  # number of centimeters in one parsec
s_Myr      = 1e+6*(365*24*60*60)  # megayears to seconds
deg_rad    = 180e0/np.pi
arcmin_deg = 60e0
arcsec_deg = 3600e0

# reading the parameters
base_path   = os.environ.get('MY_PATH')
galaxy_name = os.environ.get('galaxy_name')

params      = parameter_read(os.path.join(base_path,'inputs','parameter_file.in'))
switch      = parameter_read(os.path.join(base_path,'inputs','switches.in'))

sys.path.append(os.path.join(base_path,'data','supplementary_data', galaxy_name))
# importing literature data for field strength, pitch angle, velocity dispersion and scale height 
from observables import * 

current_directory = str(os.getcwd())

os.chdir(os.path.join(base_path,'outputs'))

with open(f'{galaxy_name}output_ca_'+str(params[r'C_\alpha'])+'K_'+str(params[r'K'])+'z_'+str(params[r'\zeta'])+'psi_'+str(params[r'\psi'])+'b_'+str(params[r'\beta'])+'.out', 'rb') as f:
    kpc_r, h_f, l_f, u_f, cs_f, alphak_f, taue_f, taur_f, biso_f, bani_f, Bbar_f, tanpB_f, tanpb_f , dkdc_f = pickle.load(
        f)
with open('errors_subsonic.out', 'rb') as f:
        subsonic_errors= pickle.load(f)
with open('errors_supersonic.out', 'rb') as f:
        supersonic_errors= pickle.load(f)
h_err, l_err, u_err, cs_err, alphak_err, tau_err, \
        taur_err, biso_err, bani_err, Bbar_err, \
                tanpB_err, tanpb_err, dkdc_err = [np.maximum(sub, sup) for sub,sup in zip(subsonic_errors, supersonic_errors)]

# delete this file from the outputs folder
# os.remove(f'{galaxy_name}output_ca_'+str(params[r'C_\alpha'])+'K_'+str(params[r'K'])+'z_'+str(params[r'\zeta'])+'psi_'+str(params[r'\psi'])+'b_'+str(params[r'\beta'])+'.out')

# compare errors in sub/super-sonic regimes and save the maximum of the two
err_quant_list = [np.maximum(sub, sup) for sub,sup in zip(subsonic_errors, supersonic_errors)]

os.chdir(os.path.join(base_path,'inputs'))

with open('zip_data.in', 'rb') as f:
    kpc_r, data_pass = pickle.load(f)

# interpolate velocity and scale height data as per availability
dat_u = griddata(kpc_radius, np.sqrt(3)*kms_sigmaLOS, kpc_r, method='linear',
                 fill_value=nan, rescale=False)*1e+5
try: # only for M31
    dat_u_warp = griddata(kpc_radius, np.sqrt(3)*kms_sigmaLOS_warp, kpc_r, method='linear',
                    fill_value=nan, rescale=False)*1e+5
except NameError:
    pass

try: # Bacchini+19 data for NGC 6946
    dat_u_bacchini = griddata(kpc_radius_b, np.sqrt(3)*kms_sigmaLOS_b, kpc_r, method='linear',
                    fill_value=nan, rescale=False)*1e+5
except NameError:
    pass

try: # Bacchini+19 data for NGC 6946 h
    dat_h_bacchini = griddata(kpc_radius_h_b, h_b, kpc_r, method='linear',
                    fill_value=nan, rescale=False)
except NameError:
    pass

os.chdir(current_directory)

# # calculate pitch angles and errors
# pB, po, pb, pB_err, po_err, pb_err = analytical_pitch_angle_integrator(kpc_r, tanpB_f,tanpb_f, \
#                                    Bbar_f, bani_f, tanpB_err,tanpb_err, Bbar_err, bani_err)

# calculate pitch angles and errors
pB, po, pb, pB_err, po_err, pb_err = new_pitch_angle_integrator(kpc_r, tanpB_f,tanpb_f, \
                                   Bbar_f, bani_f, tanpB_err,tanpb_err, Bbar_err, bani_err, taue_f, data_pass)

# print(pB,pb,po)