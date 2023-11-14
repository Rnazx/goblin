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
kms_sigmaLOS = vdisp_df["vdisp"].values
kms_sigmaLOS_warp = vdisp_df["vdisp warp"].values
arcsec_r = vdisp_df["r arcsec"].values
kpc_gal_dist = 780e0  # from Beck et al
kpc_radius = kpc_gal_dist*arcsec_r/(arcsec_deg*deg_rad)
#########################################################################################################################################################


# Magnetic field data from Beck et. al. 2019 Tables 3 and 4
G_dat_Bord = np.array([4.9, 5.2, 4.9, 4.6])
G_dat_Breg = np.array([1.8, 2.1, 2.6, 2.7])
G_dat_Btot = np.array([7.3, 7.5, 7.1, 6.3])
#radial ranges
mrange_endps = np.array([6.8, 9.0, 11.3, 13.6, 15.8]) #lower limit of radial ranges where M is used, given in table 4, Beck+19
range_pb_beck19 = (mrange_endps[1:] + mrange_endps[:-1])/2 #average of each of the intervals given in above array. contains 1 less point than above array

#ordered pitch angle data 
po_beck19 = np.array([30, 29, 26, 27, 27]) * np.pi/180 #pitch angle of ordered field
err_po_beck19 = np.array([5, 4, 3, 2, 3]) * np.pi/180 #error in po
rmrange_endps = np.array([7, 8, 9, 10, 11,12]) #given in table 4, Beck+19
range_po_beck19 = (rmrange_endps[1:] + rmrange_endps[:-1])/2 

#mean pitch angle data 
M_pb_beck19 = np.array([13, 19, 11, 8]) * np.pi/180 #pitch angle of regular field
#plot this against range_pb_beck19
err_M_pb_beck19 = np.array([4, 3, 3, 3]) * np.pi/180

RM_pb_beck19 = np.array([4, 9, 7, 7, 5]) * np.pi/180 #pitch angle of regular field at another radial range
#plot this against range_po_beck19
err_RM_pb_beck19 = np.array([5, 3, 3, 2, 3]) * np.pi/180

#scale height data
kpc_dat_r = np.array([7, 9, 11, 13])
pc_dat_h = np.array([316.4, 371.9, 437.1, 513.7])