import numpy as np

# ........................................ M31 ........................................
# Magnetic field data from Beck et. al. 2019 Tables 3 and 4
G_dat_Bord_m31 = np.array([4.9, 5.2, 4.9, 4.6])
G_dat_Breg_m31 = np.array([1.8, 2.1, 2.6, 2.7])
G_dat_Btot_m31 = np.array([7.3, 7.5, 7.1, 6.3])
#radial ranges
mrange_endps_m31 = np.array([6.8, 9.0, 11.3, 13.6, 15.8]) #lower limit of radial ranges where M is used, given in table 4, Beck+19
mrange_m31 = (mrange_endps_m31[1:] + mrange_endps_m31[:-1])/2 #average of each of the intervals given in above array. contains 1 less point than above array

#ordered pitch angle data 
po_beck19_m31 = np.array([30, 29, 26, 27, 27]) * np.pi/180 #pitch angle of ordered field
err_po_beck19_m31 = np.array([5, 4, 3, 2, 3]) * np.pi/180 #error in po
rmrange_endps_m31 = np.array([7, 8, 9, 10, 11,12]) #given in table 4, Beck+19
range_po_beck19_m31 = (rmrange_endps_m31[1:] + rmrange_endps_m31[:-1])/2 

#mean pitch angle data 
M_pb_beck19_m31 = np.array([13, 19, 11, 8]) * np.pi/180 #pitch angle of regular field
#plot this against range_pb_beck19
err_M_pb_beck19_m31 = np.array([4, 3, 3, 3]) * np.pi/180

RM_pb_beck19_m31 = np.array([4, 9, 7, 7, 5]) * np.pi/180 #pitch angle of regular field at another radial range
#plot this against range_po_beck19
err_RM_pb_beck19_m31 = np.array([5, 3, 3, 2, 3]) * np.pi/180

#scale height data
kpc_dat_r_m31 = np.array([7, 9, 11, 13])
pc_dat_h_m31 = np.array([316.4, 371.9, 437.1, 513.7])

# ........................................ M33 ........................................

# no RM data for m33

# Magnetic field data from Beck et. al. 2019 Tables 3 and 4
G_dat_Bord_m33 = np.array([3.1,3.1])
G_dat_Breg_m33 = np.array([1.3,2.4])
G_dat_Btot_m33 = np.array([8.7,7.6])

#ordered field pitch angle 
#Beck+19
po_beck19_m33 = np.array([48,40,41,35]) * np.pi/180 #pitch angle of ordered field
err_po_beck19_m33 = np.array([5,5,5,6]) * np.pi/180 #error in po
range_po_m33=np.array([1.0,3.0,5.0,7.0,9.0]) #for po
po_range_beck19_m33 = (range_po_m33[1:] + range_po_m33[:-1])/2 #average of each of the intervals given in above array. contains 1 less point than above array

#mean field pitch angle
#Beck+19
pb_beck19_m33 = np.array([51,41]) * np.pi/180 #pitch angle of regular field
err_pb_beck19_m33 = np.array([2,2]) * np.pi/180
mrange_endps_m33 = np.array([1.0,3.0,5.0]) #radial ranges (for B and pB) where M is used, given in table 4, Beck+19
mrange_m33 = (mrange_endps_m33[1:] + mrange_endps_m33[:-1])/2 #average of each of the intervals given in above array. contains 1 less point than above array

#scale height data
kpc_dat_r_m33 = np.array([2.02,3.99])
pc_dat_h_m33 = np.array([268.42,391.90])

# ........................................ M51 ........................................

# Magnetic field data from Beck et. al. 2019 Tables 3 and 4
# *(8.5/7.6) is correction for distance to M51 chosen
# plotted against range_pb
G_dat_Bord_m51 = np.array([8.6,7.6,7.6,7.8])
G_dat_Breg_m51 = np.array([1.3,1.8,2.6,3.2])
G_dat_Btot_m51 = np.array([17,16,15,13])

#ordered field pitch angles
#Beck+19
range_po_beck19_m51=np.array([1.7,3.0,7.8])*(8.5/7.6) #for po
po_beck19_m51 = np.array([20,27,19]) * np.pi/180 #pitch angle of ordered field
err_po_m51 = np.array([2,2,5]) * np.pi/180 #error in po

#Surgent+23
#i= 22.5 d=8.58 Mpc
range_po_surgent23_m51=np.array([1.4393531,1.913746631,2.377358491,2.851752022,3.326145553,3.811320755,
                              4.285714286,4.749326146,5.223719677,5.687331536,6.161725067,6.64690027,7.121293801,])*(8.5/8.58) #distance correction done
po_surgent23_m51=np.array([30.75949367,22.78481013,19.36708861,23.92405063,27.72151899,28.86075949,
                       25.44303797,28.10126582,25.06329114,26.20253165,28.86075949,23.16455696,32.27848101])

#Borlaff+23
#i= 22.5 d=8.58 Mpc
range_po_borlaff23_m51=np.array([0.4532084,  0.89827749, 1.66880654, 2.3304253,  2.92022675, 3.66114695, 
 4.32883351, 4.91799469 ,5.6388513,  6.34400257, 7.06176041, 7.79785886])*(8.5/8.58) #distance correction done
po_borlaff23_m51=np.array([80.37383178, 14.95327103, 30.8411215 , 32.71028037 ,28.03738318, 33.64485981,
 45.79439252, 44.85981308, 30.8411215,  30.8411215  ,14.01869159, 33.64485981])

#Borlaff+21
#i= 22
range_po_borlaff21_m51 = np.array([0.349693006,1.036105179,1.699332621,2.445568607,3.137987186,3.858756006,
                       4.578323545,5.283355579,6.152602776,6.873011212,7.59197811,8.310704752,9.017058195])*(8.5/8.58) #distance correction done
po_borlaff21_m51       = np.array([70.76775227,36.02349172,24.64388681,22.46129204,23.11478911,30.84570208,31.49706353,26.48585158,
                       29.24933262,34.85638014,31.96796583,27.66364122,30.43993593])

#mean field pitch angles
dat_pb_m51       = np.array([20,24,22,18]) * np.pi/180 #pitch angle of regular field
err_pb_m51       = np.array([1,4,4,1,]) * np.pi/180
mrange_endps_m51 = np.array([2.4,3.6,4.8,6.0,7.2])*(8.5/7.6) #table 4, Beck+19
range_pb_m51     = (mrange_endps_m51[1:] + mrange_endps_m51[:-1])/2 #average of each of the intervals given in above array
mrange_m51       = range_pb_m51

#scale height data
kpc_dat_r_m51 = np.array([2.96,4.18,5.34,6.56])*(8.5/7.6)
pc_dat_h_m51  = np.array([256.54,293.76,336.30,386.81])

# ........................................ NGC6946 ........................................

#no pb data for ngc 6946

# Magnetic field data from Beck et. al. 2019 Tables 3 and 4
# plotted against range_pb
G_dat_Bord_ngc6946          = np.array([5.1,5.1])
G_dat_Breg_ngc6946          = np.array([1.2,1.9])
G_dat_Btot_ngc6946          = np.array([19,13])
G_dat_Btot_Basu_Roy_ngc6946 = np.array([25.94158314, 18.01236385, 16.00077568, 14.82217009, 13.552339, 12.78313628, 12.48954207, 12.54311019, 12.33915784,
    12.30677476, 11.78905826, 11.32024453, 10.96905795, 10.88319039, 10.59432994, 10.22702533, 9.840617839, 9.354965301, 9.046228907])
kpc_Btot_Basu_Roy_ngc6946 = np.array([0.2472547511, 0.7417642533, 1.236273755, 1.730783258, 2.22529276, 2.719802262, 3.214311764, 3.708821266, 4.203330769,
    4.697840271, 5.192349773, 5.686859275, 6.181368777, 6.67587828, 7.170387782, 7.664897284, 8.159406786, 8.653916288, 9.14842579])
err_G_dat_Btot_Basu_Roy_ngc6946 = np.array([4.681717618, 0.7877719705, 0.313559657, 0.3610886715, 0.1835198111, 0.1369962248, 0.1256641352, 0.1768765286, 0.1847725005,
    0.3178479284, 0.1438453549, 0.1432692612, 0.119313422, 0.1240396, 0.1470014107, 0.1632999742, 0.1106762298, 0.1013907689, 0.08817978268])

#radial ranges #discontinuous ranges in Beck+19
mrange_endps_ngc6946 = np.array([0,4.7,9.4]) #lower limit of radial ranges where M is used, given in table 4, Beck+19
mrange_ngc6946       = ((mrange_endps_ngc6946[1:] + mrange_endps_ngc6946[:-1])/2)*(7.72/7) #average of each of the intervals given in above array. contains 1 less point than above array

#ordered field pitch angle data 
#Beck+19
po_range1_beck19_ngc6946     = np.array([27,21,10]) * np.pi/180 #pitch angle of ordered field
err_po_range1_beck19_ngc6946 = np.array([2,2,6]) * np.pi/180 #error in po
range1_endpoints_ngc6946     = np.array([0,6,12,18])
range1_beck19_ngc6946        = ((range1_endpoints_ngc6946[1:] + range1_endpoints_ngc6946[:-1])/2)*(7.72/7)

po_range2_beck19_ngc6946     = np.array([30,32,10]) * np.pi/180 
err_po_range2_beck19_ngc6946 = np.array([2,4,5]) * np.pi/180 
range2_beck19_ngc6946        = np.array([1.5,5.5,8.5])*(7.72/7)

#Borlaff+23
#i= 38.4 d=6.8 Mpc
range_po_borlaff23_ngc6946   = np.array([0.353880432,0.790913335,1.418296894,1.982883949,2.471333709,3.080104765,3.694601205,4.184288508,4.784468282,5.418608,6.020405996,6.625304163,7.152697436,7.722072731,8.336771937,8.829699955,9.441738654,10.09620137,10.69315911])*(7.72/6.8) #distance correction done
po_borlaff23_signed_ngc6946  = np.array([-76.74919675,-35.08977509,-13.38688339,-19.81666982,-29.86013986,-32.63088263,-19.93951994,-26.32772633,-21.80873181,-26.37875638,-32.76318276,-29.14571915,-26.43356643,-19.17595918,-27.37289737,-18.29332829,-19.21564922,-22.86524287,-13.78567379])
po_borlaff23_ngc6946         = -1*po_borlaff23_signed_ngc6946

#Surgent+23
#i= 38.4 d=6.8 Mpc
range_po_Surgent23_ngc6946   = np.array([0.827361564,1.087947883,1.361563518,1.622149837,1.908794788,2.182410423,2.456026059,2.716612378,2.990228013,3.276872964,3.550488599,3.811074919,4.084690554,4.371335505,4.631921824,4.905537459,5.179153094,5.45276873,5.726384365,5.986970684,6.260586319,6.534201954,7.342019544,7.628664495])
po_Surgent23_signed_ngc6946  = np.array([-23.3953843,-22.6543766,-22.68057384,-15.81190658,-17.75424492,-23.90810174,-23.93429898,-24.72520618,-21.30459491,-21.33203964,-19.82632199,-18.31935685,-17.96257537,-14.92619031,-13.41922517,-10.38159263,-5.812045187,-7.370157322,-10.07720563,-13.54896389,-9.362395176,-5.558805184,-64.61487283,-9.110402661])
po_Surgent23_ngc6946         = -1*po_Surgent23_signed_ngc6946

#scale height data
kpc_dat_r_ngc6946 = np.array([3.01,9.02,13.02])*(7.72/7)
pc_dat_h_ngc6946  = np.array([259.2,564.92,923.81])