#!/bin/bash
source config.sh

# Put the data in a usable format
cd data
<<<<<<< HEAD
# python data_common.py 
# python vel_disp.py
=======
python data_common.py 
python velocity_dispersion.py
>>>>>>> 4eee51b97cd7755aec848854d9c4f0e23e79c2cd

# #Solve all the expressions in terms of the scale height h
# cd ../expressions
# python turbulence_expressions.py
# python magnetic_expressions.py

cd ../src
#Change the data into the preffered format for Sympy
python zipped_data.py
#Find the root and solve for all quantities
python model_predictions.py

#If there is a change in the model run this to find new scaling relations
#python scale_exponents.py

#Error aprroximation
python error.py

#Plotting
python plot_generator.py


