import numpy as np
import pandas as pd
from data_helpers import incl_distance_correction, remove_data
import os 
current_directory = str(os.getcwd())

########################M31###################################
#M31
os.chdir(current_directory+'\M31_data')
raw_data = pd.read_csv('combined_data_m31.csv', skiprows=1)
#data to be removed
data_rem = 'chemin'
remove_data(raw_data, data_rem)
#distance correction
distance_m31= 0.78 #Mpc
distances_Mpc=np.array([0.785,0.785,0.780,0.780])

#inclination correction
i_m31= 75 #deg
inclinations=np.array([i_m31,i_m31,i_m31,75,77.5]) #used i_m31 as no inclination correction is needed for Claude data
raw_data = incl_distance_correction(raw_data, distance_new=distance_m31, distance_old=distances_Mpc,\
                          i_new=np.radians(i_m31), i_old=np.radians(inclinations))

print(raw_data)

os.chdir(current_directory)
raw_data.to_csv('formatted_data.csv', index=False)