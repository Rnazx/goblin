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
# Hitschfeld+09 data (d = 8.4 Mpc)
vdisp_df     = pd.read_csv(f'{galaxy_name}_veldisp.csv')
kms_sigmaLOS = vdisp_df["v disp"].values
kpc_radius   = vdisp_df["r"].values*(8.5/8.4)
#####################################################################################################################################

# Magnetic field data from Beck et. al. 2019 Tables 3 and 4
# *(8.5/7.6) is correction for distance to M51 chosen
# plotted against range_pb
G_dat_Bord     = np.array([8.6,7.6,7.6,7.8])
G_dat_Breg     = np.array([1.3,1.8,2.6,3.2]) # from Beck+19, slightly different from original values in Fletcher+11
err_G_dat_Breg = np.array([0.1,0.5,1,0.1]) # Fletcher+11 unmodified 
G_dat_Btot     = np.array([17,16,15,13])

#ordered field pitch angles
#Beck+19
po_endps_1      = np.array([1.2,2.4,3.6])*(8.5/7.6)
po_endps_2      = np.array([7.2,8.4])*(8.5/7.6)
range_po_beck19 = np.array([1.7,3.0,7.8])*(8.5/7.6) #for po
po_beck19       = np.array([20,27,19]) * np.pi/180 #pitch angle of ordered field
err_po          = np.array([2,2,5]) * np.pi/180 #error in po

#Surgent+23
#i= 22.5 d=8.58 Mpc
range_po_surgent23 = np.array([1.4393531,1.913746631,2.377358491,2.851752022,3.326145553,3.811320755,
                              4.285714286,4.749326146,5.223719677,5.687331536,6.161725067,6.64690027,7.121293801,])*(8.5/8.58) #distance correction done
po_surgent23       = np.array([30.75949367,22.78481013,19.36708861,23.92405063,27.72151899,28.86075949,
                       25.44303797,28.10126582,25.06329114,26.20253165,28.86075949,23.16455696,32.27848101])

#Borlaff+23
#i= 22.5 d=8.58 Mpc
range_po_borlaff23 = np.array([0.4532084,  0.89827749, 1.66880654, 2.3304253,  2.92022675, 3.66114695, 
                        4.32883351, 4.91799469 ,5.6388513,  6.34400257, 7.06176041, 7.79785886])*(8.5/8.58) #distance correction done
po_borlaff23       = np.array([80.37383178, 14.95327103, 30.8411215 , 32.71028037 ,28.03738318, 33.64485981,
                        45.79439252, 44.85981308, 30.8411215,  30.8411215  ,14.01869159, 33.64485981])

#Borlaff+21
#i= 22
range_po_borlaff21 = np.array([0.349693006,1.036105179,1.699332621,2.445568607,3.137987186,3.858756006,
                        4.578323545,5.283355579,6.152602776,6.873011212,7.59197811,8.310704752,9.017058195])*(8.5/8.58) #distance correction done
po_borlaff21       = np.array([70.76775227,36.02349172,24.64388681,22.46129204,23.11478911,30.84570208,31.49706353,26.48585158,
                        29.24933262,34.85638014,31.96796583,27.66364122,30.43993593])

#mean field pitch angles
dat_pb       = np.array([20,24,22,18]) * np.pi/180 #pitch angle of regular field
err_pb       = np.array([1,4,4,1]) * np.pi/180
mrange_endps = np.array([2.4,3.6,4.8,6.0,7.2])*(8.5/7.6) #table 4, Beck+19
range_pb     = (mrange_endps[1:] + mrange_endps[:-1])/2 #average of each of the intervals given in above array
mrange       = range_pb

#scale height data C16
kpc_dat_r = np.array([2.96,4.18,5.34,6.56])*(8.5/7.6)
pc_dat_h  = np.array([256.54,293.76,336.30,386.81]) # need to apply exponential correction if used


