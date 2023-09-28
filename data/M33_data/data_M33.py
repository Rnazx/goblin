import numpy as np
from sympy import *
from fractions import Fraction
import pickle
from scipy.interpolate import griddata
import pandas as pd
import os

current_directory = str(os.getcwd())

#converting from 2nd unit to 1st
pc_kpc = 1e3  # number of pc in one kpc
cm_km = 1e5  # number of cm in one km
s_day = 24*3600  # number of seconds in one day
s_min = 60  # number of seconds in one minute
s_hr = 3600  # number of seconds in one hour
cm_Rsun = 6.957e10  # solar radius in cm
g_Msun = 1.989e33  # solar mass in g
cgs_G = 6.674e-8
cms_c = 2.998e10
g_mH = 1.6736e-24
g_me = 9.10938e-28
cgs_h = 6.626e-27
deg_rad = 180e0/np.pi
arcmin_deg = 60e0
arcsec_deg = 3600e0
cm_kpc = 3.086e+21  # number of centimeters in one parsec
cm_pc = cm_kpc/1e+3
s_Myr = 1e+6*(365*24*60*60)  # megayears to seconds

#REQUIRED FUNCTIONS
###########################################################################################################################################
#importing data from csv file to np array
def file_reader(list_of_file_names):
    dataframe_list=[]
    for i in list_of_file_names:
        df = pd.read_csv(current_directory+r'\{}.dat'.format(i),delimiter=',')
        dataframe_list.append(df)
    return dataframe_list

#extrapolation
def interpolation(list1,list2,standard):
    extrapolated_data = griddata(list1, list2, standard, method='linear', fill_value=nan, rescale=False)
    return extrapolated_data
###########################################################################################################################################

#change first file name to sigmatot_up52_M33 to set upsilon to 72
file_names=['sigmatot_up52_M33','sigmaHI_Kam_M33','sigmaH2_M33','RC_M33_kam','RC_M33_kam',
            'sigmaSFR_M33','vdisp_M33']   # RC_M33_kam repeated twice to get error from the file

dataframe_list=file_reader(file_names)

#no radius correction needed
#chosen D=0.84 Mpc

#obtain arrays of quantities
col_names=['sigma tot','sigma HI','sigma H2','kms_vcirc_Kam','error_Kam',
           'sigma sfr','vel disp kms']
conv_factors=[(g_Msun/(cm_pc**2)), g_Msun/(cm_pc**2), g_Msun/(cm_pc**2), 1,1,
              g_Msun/((s_Myr*10**(3))*(cm_pc**2)), (3**0.5)] #last term to make 3D vdisp


# #to switch between Claude and Chemin data for sigmaHI and vcirc
# Claude_Chemin_switch=1 #Claude data chosen
# tbd=[col_names,dataframe_list,conv_factors,distances_Mpc,r_col_names,radius_conv_corr]
# if Claude_Chemin_switch==0: #choosing Chemin data
#     for i in range(len(tbd)):
#         del tbd[i][5] #remove vcirc_Claude
#         del tbd[i][2] #remove sigmaHI_Claude
# else: #chose Claude data
#     for i in range(len(tbd)):
#         del tbd[i][3] #remove vcirc_Chemin
#         del tbd[i][1] #remove sigmaHI_Chemin

#redefining the lists without Chemin/Claude data
# col_names        =tbd[0]
# dataframe_list   =tbd[1]
# conv_factors     =tbd[2]
# distances_Mpc    =tbd[3]
# r_col_names      =tbd[4]
# radius_conv_corr =tbd[5]

# get the radii
radius_list_kpc=[np.array(dataframe_list[i]['r kpc']) for i in range(len(dataframe_list))]

# get the quantities and convert to cgs units
quant_list=[np.array(dataframe_list[i][col_names[i]]) for i in range(len(dataframe_list))]
quant_list=[quant_list[i]*conv_factors[i] for i in range(len(quant_list))]  

#find array with least length
#if vel disp data is the smallest one, remove that and repeat the process
kpc_r = min(radius_list_kpc, key=len) 

#Temp data from Lin+17
# T_OIII=8398+(2243/7.203)*kpc_r
# T_NII=0*kpc_r
T_OIII=0*kpc_r
T_NII=7756+(3520/7.203)*kpc_r
T=(T_OIII+T_NII)/2
error_temp=np.std(T)

#calculate omega
kmskpc_Om = quant_list[3]/radius_list_kpc[3]
dat_omega = kmskpc_Om*cm_km/cm_kpc
#calculate q
dat_q = -1 * radius_list_kpc[3]/ kmskpc_Om * np.gradient(kmskpc_Om)/np.gradient(radius_list_kpc[3])
#add omega, q and radius
quant_list.insert(3,dat_q)
radius_list_kpc.insert(3,radius_list_kpc[3])
quant_list.insert(4,dat_omega)
radius_list_kpc.insert(4,radius_list_kpc[3])

#remove vcirc data #now sfr in 5th pos
del quant_list[5]
del radius_list_kpc[5]

#interpolating the data and appending the common radius list 
#interpolated arrays has an _ip ending
quant_list_ip=[interpolation(radius_list_kpc[i],quant_list[i],kpc_r) for i in range(len(quant_list))]

quant_list_ip.insert(len(quant_list_ip)-1,T) # -1 because i need v_disp to be at end to remove easily in next step
quant_list_ip.insert(0,kpc_r)

#removing nan values for points whr interpolation is impossible
nan_max = np.argmax([np.sum(np.isnan(d)) for d in quant_list_ip])
nan_max_data = quant_list_ip[nan_max]
nan_mask = ~np.isnan(nan_max_data)
nandeleted_data = []
for i,d in enumerate(quant_list_ip):
    nandeleted_data.append(d[nan_mask])

#separating vel disp data since to maintain uniformity with other galaxies. To be used while plotting
dat_u= nandeleted_data[-1]
del nandeleted_data[-1]


#separating vcirc error since to maintain uniformity with other galaxies 
RC_error= nandeleted_data[6]
del nandeleted_data[6] #removing the vcirc error from nandeleted data
# nandeleted_data = tuple(nandeleted_data)
col_names_f = ["Radius(kpc)", "sigmatot", "sigmaHI","sigmaH2", "q", "omega", "sigmasfr", "T"]
df = pd.DataFrame(nandeleted_data, index = col_names_f)
print(df)
os.chdir("..")
df.to_csv('data_interpolated.csv')

########################################################################################################################################
                                                      #ERROR DATA#
########################################################################################################################################

#to load errors if any given in the dat files
#this assumes that in all files, the error columns are named as 'quant_error'
error_list=[]

# temp_error=np.array([error_temp]*len(radius_list_kpc[4]))
temp_error=np.array([error_temp]*len(kpc_r))
error_list.append(temp_error)
error_list.append(RC_error)






