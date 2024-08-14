import numpy as np
import pandas as pd
import os
from scipy.interpolate import griddata
from numpy import nan
from data_common import interpolated_df

# get first column from interpolated_df
kpc_r = interpolated_df.iloc[:,0].values
# conversion factors

pc_kpc     = 1e3  # number of pc in one kpc
cm_km      = 1e5  # number of cm in one km
cm_kpc     = 3.086e+21  # number of centimeters in one parsec
s_Myr      = 1e+6*(365*24*60*60)  # megayears to seconds
deg_rad    = 180e0/np.pi
arcmin_deg = 60e0
arcsec_deg = 3600e0

kpc_gal_dist = [780, 840, 8500, 7720] # in kpc
#####################################################################################################################################
galaxy_name = os.environ.get('galaxy_name')
base_path = os.environ.get('MY_PATH')

os.chdir(os.path.join(base_path, 'data','supplementary_data', f'{galaxy_name}'))

vdisp_df = pd.read_csv(f'{galaxy_name}_veldisp.csv')

# no distance correction made for r 
if galaxy_name == 'm31':
    kms_sigmaLOS      = vdisp_df["vdisp"].values
    kms_sigmaLOS_warp = vdisp_df["vdisp warp"].values
    arcsec_r          = vdisp_df["r arcsec"].values
    kpc_gal_dist      = 780e0  # from Beck et al
    kpc_radius        = kpc_gal_dist*arcsec_r/(arcsec_deg*deg_rad)
elif galaxy_name == 'm33':
    kms_sigmaLOS = vdisp_df["vel disp kms"].values
    kpc_radius   = vdisp_df["r kpc"].values
else:
    kms_sigmaLOS = vdisp_df["v disp"].values
    kpc_radius   = vdisp_df["r"].values

dat_u = griddata(kpc_radius, np.sqrt(3)*kms_sigmaLOS, kpc_r, method='linear',
                 fill_value=nan, rescale=False)*1e+5 # in cm/s
try:
    dat_u_warp = griddata(kpc_radius, np.sqrt(3)*kms_sigmaLOS_warp, kpc_r, method='linear',
                    fill_value=nan, rescale=False)*1e+5 # in cm/s
except NameError:
    pass

# save kpc_r, dat_u, dat_u_warp to a csv file
df = pd.DataFrame({'r': kpc_r, 'v disp': dat_u})
df.to_csv(f'{galaxy_name}_veldisp_ip.csv', index=False)
try:
    df = pd.DataFrame({'r': kpc_r, 'v disp warp': dat_u_warp})
    df.to_csv(f'{galaxy_name}_veldisp_warp_ip.csv', index=False)
except NameError:
    pass

try:
    df = pd.DataFrame({'r_bacchini': kpc_r, 'v disp_bacchini': dat_u_warp})
    df.to_csv(f'{galaxy_name}_veldisp_baccini_ip.csv', index=False)
except NameError:
    pass