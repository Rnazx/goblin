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
    model_f = pickle.load(
        f)
with open(f'{galaxy_name}output_ca_'+str(params[r'C_\alpha'])+'K_'+str(params[r'K'])+'z_'+str(params[r'\zeta'])+'psi_'+str(params[r'\psi'])+'b_'+str(params[r'\beta'])+'.out', 'rb') as f:
    kpc_r, h_f, l_f, u_f, cs_f, alphak_f, taue_f, taur_f, biso_f, bani_f, Bbar_f, tanpB_f, tanpb_f , dkdc_f = pickle.load(
        f)

os.chdir(os.path.join(base_path,'inputs'))

with open('zip_data.in', 'rb') as f:
    kpc_r, data_pass = pickle.load(f)


# calculate pitch angles and errors
# pB, po, pb, pB_err, po_err, pb_err = new_pitch_angle_integrator(kpc_r, tanpB_f,tanpb_f, \
#                                    Bbar_f, bani_f, tanpB_err,tanpb_err, Bbar_err, bani_err, taue_f, data_pass)
os.chdir(os.path.join(base_path,'outputs'))

with open(f'{galaxy_name}output_ca_'+str(params[r'C_\alpha'])+'K_'+str(params[r'K'])+'z_'+str(10.0)+'psi_'+str(params[r'\psi'])+'b_'+str(params[r'\beta'])+'.out', 'rb') as f:
    model_f2 = pickle.load(
        f)

print('kpc_r, h_f, l_f, u_f, cs_f, alphak_f, taue_f, taur_f, biso_f, bani_f, Bbar_f, tanpB_f, tanpb_f , dkdc_f')
for i,j in zip(model_f,model_f2):
    print(np.average(np.log2(i/j)))