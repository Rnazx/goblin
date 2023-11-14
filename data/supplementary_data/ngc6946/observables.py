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
#####################################################################################################################################

#no pb data for ngc 6946

# Magnetic field data from Beck et. al. 2019 Tables 3 and 4
# plotted against range_pb
G_dat_Bord = np.array([5.1,5.1])
G_dat_Breg = np.array([1.2,1.9])
G_dat_Btot = np.array([19,13])

#radial ranges #discontinuous ranges in Beck+19
mrange_endps = np.array([0,4.7,9.4]) #lower limit of radial ranges where M is used, given in table 4, Beck+19
mrange = ((mrange_endps[1:] + mrange_endps[:-1])/2)*(7.72/7) #average of each of the intervals given in above array. contains 1 less point than above array

#ordered field pitch angle data 
#Beck+19
po_range1_beck19 = np.array([27,21,10]) * np.pi/180 #pitch angle of ordered field
err_po_range1_beck19 = np.array([2,2,6]) * np.pi/180 #error in po
range1_endpoints=np.array([0,6,12,18])
range1_beck19=((range1_endpoints[1:] + range1_endpoints[:-1])/2)*(7.72/7)

po_range2_beck19 = np.array([30,32,10]) * np.pi/180 
err_po_range2_beck19 = np.array([2,4,5]) * np.pi/180 
range2_beck19=np.array([1.5,5.5,8.5])*(7.72/7)

#Borlaff+23
#i= 38.4 d=6.8 Mpc
range_po_borlaff23=np.array([0.353880432,0.790913335,1.418296894,1.982883949,2.471333709,3.080104765,3.694601205,4.184288508,4.784468282,5.418608,6.020405996,6.625304163,7.152697436,7.722072731,8.336771937,8.829699955,9.441738654,10.09620137,10.69315911])*(7.72/6.8) #distance correction done
po_borlaff23_signed=np.array([-76.74919675,-35.08977509,-13.38688339,-19.81666982,-29.86013986,-32.63088263,-19.93951994,-26.32772633,-21.80873181,-26.37875638,-32.76318276,-29.14571915,-26.43356643,-19.17595918,-27.37289737,-18.29332829,-19.21564922,-22.86524287,-13.78567379])
po_borlaff23 = -1*po_borlaff23_signed

#Surgent+23
#i= 38.4 d=6.8 Mpc
range_po_Surgent23=np.array([0.353880432,0.790913335,1.418296894,1.982883949,2.471333709,3.080104765,3.694601205,4.184288508,4.784468282,5.418608,6.020405996,6.625304163,7.152697436,7.722072731,8.336771937,8.829699955,9.441738654,10.09620137,10.69315911])*(7.72/6.8) #distance correction done
po_Surgent23_signed=np.array([-76.74919675,-35.08977509,-13.38688339,-19.81666982,-29.86013986,-32.63088263,-19.93951994,-26.32772633,-21.80873181,-26.37875638,-32.76318276,-29.14571915,-26.43356643,-19.17595918,-27.37289737,-18.29332829,-19.21564922,-22.86524287,-13.78567379])
po_Surgent23=-1*po_Surgent23_signed

#scale height data
kpc_dat_r = np.array([3.01,9.02,13.02])*(7.72/7)
pc_dat_h = np.array([259.2,564.92,923.81])