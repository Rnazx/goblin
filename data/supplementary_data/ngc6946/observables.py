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
# Boomsma+08 data (d = 6 Mpc)
vdisp_df     = pd.read_csv(f'{galaxy_name}_veldisp.csv')
kms_sigmaLOS = vdisp_df["v disp"].values
kpc_radius   = vdisp_df["r"].values*(7.72/6)

# Bacchini+19 data (d = 5.52 Mpc)
kms_sigmaLOS_b = vdisp_df["v disp_bacchini"].values
kpc_radius_b   = vdisp_df["r_bacchini"].values*(7.72/5.52)
#####################################################################################################################################

#no pb data for ngc 6946

# Magnetic field data from Beck et. al. 2019 Tables 3 and 4
# plotted against range_pb
G_dat_Bord     = np.array([5.1,5.1])
G_dat_Breg     = np.array([1.2,1.9])
err_G_dat_Breg = 0.4*G_dat_Breg        # Beck+19 mentions 30-40% error in B_reg
G_dat_Btot     = np.array([19,13])

# Basu + Roy data (d = 6.8 Mpc)
G_dat_Btot_Basu_Roy_ngc6946 = np.array([29.51143451, 19.34511435, 15.79002079, 14.91683992, 13.91891892, 13.04573805, 12.35966736, 11.92307692, 
    11.73596674, 11.73596674, 11.86070686, 11.92307692, 11.67359667, 11.36174636, 11.04989605, 10.8004158, 
    10.67567568, 10.61330561, 10.36382536, 10.23908524, 9.98960499, 9.615384615, 9.365904366, 9.116424116, 
    8.866943867])
kpc_Btot_Basu_Roy_ngc6946 = np.array([0.197132616, 0.591397849, 0.985663082, 1.379928315, 1.756272401, 2.186379928, 2.562724014, 2.974910394, 
    3.35125448, 3.745519713, 4.157706093, 4.551971326, 4.946236559, 5.340501792, 5.734767025, 6.129032258, 
    6.523297491, 6.917562724, 7.311827957, 7.70609319, 8.11827957, 8.512544803, 8.906810036, 9.301075269, 
    9.695340502])*(7.72/6.8)
err_G_dat_Btot_Basu_Roy_ngc6946 = np.array([4.490644, 3.430353, 0.841996, 0.966736, 1.216216, 1.091476, 0.842052, 0.966736, 0.966736, 0.748497, 
    0.997921, 1.434567, 1.372141, 1.278586, 1.091476, 1.060291, 0.904366, 1.153846, 1.278586, 1.247401, 
    1.029106, 0.842052, 0.935551, 0.935607, 0.810811])

#radial ranges #discontinuous ranges in Beck+19
mrange_endps = np.array([0,4.7,9.4])*(7.72/7) #lower limit of radial ranges where M is used, given in table 4, Beck+19
mrange       = ((mrange_endps[1:] + mrange_endps[:-1])/2) #average of each of the intervals given in above array. contains 1 less point than above array

#ordered field pitch angle data 
#Beck+19
po_range1_beck19     = np.array([27,21,10]) * np.pi/180 #pitch angle of ordered field
err_po_range1_beck19 = np.array([2,2,6]) * np.pi/180 #error in po
range1_endpoints     = np.array([0,6,12,18])
range1_beck19        = ((range1_endpoints[1:] + range1_endpoints[:-1])/2)*(7.72/7)

range1_endpoints     = np.array([0,6,12,18])*(7.72/7)
range2_endpoints     = np.array([1,2,5,6,8,9])*(7.72/7)

po_range2_beck19     = np.array([30,32,10]) * np.pi/180 
err_po_range2_beck19 = np.array([2,4,5]) * np.pi/180 
range2_beck19        = np.array([1.5,5.5,8.5])*(7.72/7)

#Borlaff+23
#i= 38.4 d=6.8 Mpc
range_po_borlaff23   = np.array([0.353880432,0.790913335,1.418296894,1.982883949,2.471333709,3.080104765,3.694601205,4.184288508,4.784468282,5.418608,6.020405996,6.625304163,7.152697436,7.722072731,8.336771937,8.829699955,9.441738654,10.09620137,10.69315911])*(7.72/6.8) #distance correction done
po_borlaff23_signed  = np.array([-76.74919675,-35.08977509,-13.38688339,-19.81666982,-29.86013986,-32.63088263,-19.93951994,-26.32772633,-21.80873181,-26.37875638,-32.76318276,-29.14571915,-26.43356643,-19.17595918,-27.37289737,-18.29332829,-19.21564922,-22.86524287,-13.78567379])
po_borlaff23         = -1*po_borlaff23_signed

#Surgent+23
#i= 38.4 d=6.8 Mpc
range_po_Surgent23   = np.array([0.827361564,1.087947883,1.361563518,1.622149837,1.908794788,2.182410423,2.456026059,2.716612378,2.990228013,3.276872964,3.550488599,3.811074919,4.084690554,4.371335505,4.631921824,4.905537459,5.179153094,5.45276873,5.726384365,5.986970684,6.260586319,6.534201954,7.342019544,7.628664495])
po_Surgent23_signed  = np.array([-23.3953843,-22.6543766,-22.68057384,-15.81190658,-17.75424492,-23.90810174,-23.93429898,-24.72520618,-21.30459491,-21.33203964,-19.82632199,-18.31935685,-17.96257537,-14.92619031,-13.41922517,-10.38159263,-5.812045187,-7.370157322,-10.07720563,-13.54896389,-9.362395176,-5.558805184,-64.61487283,-9.110402661])
po_Surgent23         = -1*po_Surgent23_signed

# tentative data from Vasanth pvt. comm.
range_po_V = np.arange(1.5, 10, 1)
po_V       = np.array([28.28,25.1,34.06,31.93,21.37,31.36,31.56,32.11,27.81])

#####################################################################################################################################

# C16 scale height data
# replaced with exponential profile now
kpc_dat_r = np.array([3.01,9.02,13.02])*(7.72/7)
pc_dat_h  = np.array([259.2,564.92,923.81])

# Bacchini+19 data (d = 5.52 Mpc)
os.chdir(os.path.join(base_path, 'data','supplementary_data', f'{galaxy_name}'))

h_df           = pd.read_csv(f'{galaxy_name}_h_bacchini.csv')
h_b            = h_df["h_bacchini"].values
kpc_radius_h_b = h_df["r_bacchini"].values*(7.72/5.52)

