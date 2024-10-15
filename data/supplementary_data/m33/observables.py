import numpy as np
import pandas as pd
import os

# conversion factors
pc_kpc     = 1e3  # number of pc in one kpc
cm_km      = 1e5  # number of cm in one km
cm_kpc     = 3.086e+21  # number of centimeters in one parsec
s_Myr      = 1e+6*(365*24*60*60)  # megayears to seconds
deg_rad    = 180e0/np.pi
arcmin_deg = 60e0
arcsec_deg = 3600e0
#####################################################################################################################################

galaxy_name = os.environ.get('galaxy_name')
base_path   = os.environ.get('MY_PATH')

os.chdir(os.path.join(base_path, 'data','supplementary_data', f'{galaxy_name}'))

# multiplication by root(3) happens in plot_generator.py
vdisp_df     = pd.read_csv(f'{galaxy_name}_veldisp.csv')
kms_sigmaLOS = vdisp_df["vel disp kms"].values
kpc_radius   = vdisp_df["r kpc"].values

#####################################################################################################################################

# Magnetic field data from Beck et. al. 2019 Tables 3 and 4
G_dat_Bord     = np.array([3.1,3.1])
G_dat_Breg     = np.array([1.3,2.4])
err_G_dat_Breg = 0.4*G_dat_Breg        # Beck+19 mentions 30-40% error in B_reg
G_dat_Btot     = np.array([8.7,7.6])

#ordered field pitch angle 
#Beck+19
po_beck19       = np.array([48,40,41,35]) * np.pi/180 #pitch angle of ordered field
err_po_beck19   = np.array([5,5,5,6]) * np.pi/180 #error in po
range_po        = np.array([1.0,3.0,5.0,7.0,9.0]) #for po
po_range_beck19 = (range_po[1:] + range_po[:-1])/2 #average of each of the intervals given in above array. contains 1 less point than above array

#mean field pitch angle
#Beck+19
pb_beck19       = np.array([51,41]) * np.pi/180 #pitch angle of regular field
err_pb_beck19   = np.array([2,2]) * np.pi/180
mrange_endps    = np.array([1.0,3.0,5.0]) #radial ranges (for B and pB) where M is used, given in table 4, Beck+19
mrange          = (mrange_endps[1:] + mrange_endps[:-1])/2 #average of each of the intervals given in above array. contains 1 less point than above array

#scale height data
kpc_dat_r = np.array([2.02,3.99])
pc_dat_h  = np.array([268.42,391.90])

#####################################################################################################################################

# RM_dat_po = np.array([48,40,41,35]) * np.pi/180 #pitch angle of ordered field
# err_RM_dat_po = np.array([5,5,5,6]) * np.pi/180 #error in po
# rmdat_tanpo = np.tan(RM_dat_po)
# rm_errdat_tanpo = 1/(np.cos(err_RM_dat_po))**2

# M_dat_pb = np.array([51,41]) * np.pi/180 #pitch angle of regular field
# err_M_dat_pb = np.array([2,2]) * np.pi/180
# RM_dat_pb = np.array([4, 9, 7, 7, 5]) * np.pi/180 #pitch angle of regular field at another radial range
# err_RM_dat_pb = np.array([5, 3, 3, 2, 3]) * np.pi/180
# rmdat_tanpb = np.tan(RM_dat_pb)
# rmdat_tanpb = np.tan(M_dat_pb) #need to change name from rm to m
# rm_errdat_tanpb = 1/(np.cos(err_RM_dat_pb))**2
# rm_errdat_tanpb = 1/(np.cos(err_M_dat_pb))**2 #need to change name from rm to m

#radial ranges
# range_po=np.array([1.0,3.0,5.0,7.0,9.0]) #for po
# po_mrange = (range_po[1:] + range_po[:-1])/2 #average of each of the intervals given in above array. contains 1 less point than above array

# mrange_endps = np.array([1.0,3.0,5.0]) #radial ranges (for B and pB) where M is used, given in table 4, Beck+19
# mrange = (mrange_endps[1:] + mrange_endps[:-1])/2 #average of each of the intervals given in above array. contains 1 less point than above array
# rmrange = np.arange(7.5, 12, 1) #radial ranges where RM is used, given in table 4, Beck+19

