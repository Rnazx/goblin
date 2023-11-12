import numpy as np
import pandas as pd
import os

# conversion factors
pc_kpc = 1e3  # number of pc in one kpc
cm_km = 1e5  # number of cm in one km
cm_kpc = 3.086e+21  # number of centimeters in one parsec
s_Myr = 1e+6*(365*24*60*60)  # megayears to seconds
deg_rad = 180e0/np.pi
arcmin_deg = 60e0
arcsec_deg = 3600e0
#####################################################################################################################################
galaxy_name = os.environ.get('galaxy_name')
base_path = os.environ.get('MY_PATH')

os.chdir(os.path.join(base_path, 'data','supplementary_data', f'{galaxy_name}'))

vdisp_df = pd.read_csv(f'{galaxy_name}_veldisp.csv')
kms_sigmaLOS = vdisp_df["v disp"].values

kpc_radius = vdisp_df["r"].values

#no RM data for M51

# Magnetic field data from Beck et. al. 2019 Tables 3 and 4
# *(8.5/7.6) is correction for distance to M51 chosen
G_dat_Bord = np.array([8.6,7.6,7.6,7.8])
G_dat_Breg = np.array([1.3,1.8,2.6,3.2])
G_dat_Btot = np.array([17,16,15,13])

# pitch angle data #RM= rotation measure data, M= fitting polarisation angles

RM_dat_po = np.array([20,27,19]) * np.pi/180 #pitch angle of ordered field, not really found using RM
err_RM_dat_po = np.array([2,2,5]) * np.pi/180 #error in po
rmdat_tanpo = np.tan(RM_dat_po)
rm_errdat_tanpo = 1/(np.cos(err_RM_dat_po))**2

M_dat_pb = np.array([20,24,22,18]) * np.pi/180 #pitch angle of regular field
err_M_dat_pb = np.array([1,4,4,1,]) * np.pi/180
m_errdat_tanpb = 1/(np.cos(err_M_dat_pb))**2

#radial ranges
range_po=np.array([1.7,3.0,7.8])*(8.5/7.6) #for po
mrange_endps = np.array([2.4,3.6,4.8,6.0,7.2])*(8.5/7.6) #lower limit of radial ranges where M is used, given in table 4, Beck+19
mrange = (mrange_endps[1:] + mrange_endps[:-1])/2 #average of each of the intervals given in above array. contains 1 less point than above array

#scale height data
kpc_dat_r = np.array([2.96,4.18,5.34,6.56])*(8.5/7.6)
pc_dat_h = np.array([256.54,293.76,336.30,386.81])


