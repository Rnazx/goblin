printf "Choose from the following list of galaxies \n"
printf "1. M31   ||   2. M33   ||   3. M51   ||   4. NGC6946   \n"
read galname
printf "You have selected $galname \n"  
printf "Running interpolation for galaxy $galname \n"
cd $galname"_data"
python "data_"$galname".py"