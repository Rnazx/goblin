import numpy as np
import pandas as pd
from data_helpers import incl_distance_correction, remove_data
import os 
current_directory = str(os.getcwd())

########################M31###################################
#M31
os.chdir(current_directory+'\M31_data')
raw_data = pd.read_csv('combined_data_m31.csv', skiprows=1)


#distance correction
n_dist_m31= 0.78 #Mpc
o_dist_m31=np.array([0.785,0.785,0.780,0.780])

#inclination correction
n_i_m31= 75 #deg
o_i_m31=np.array([n_i_m31,77,n_i_m31,77, n_i_m31,75,77.5]) #used i_m31 as no inclination correction is needed for Claude data

raw_data = incl_distance_correction(raw_data, distance_new=n_dist_m31, distance_old=o_dist_m31,\
                          i_new=np.radians(n_i_m31), i_old=np.radians(o_i_m31))

#data to be removed
data_rem = 'chemin'
remove_data(raw_data, data_rem)

os.chdir(current_directory)
raw_data.to_csv('formatted_data.csv', index=False)