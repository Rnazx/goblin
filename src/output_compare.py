print('######  OUTPUT COMPARISON FILE BEGINS #####')

import numpy as np
import pickle
import os
import pandas as pd
from data_helpers import *
from datetime import date
import csv # for saving err data as csv file
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib
import astropy.units as u
import sys
from tqdm import tqdm
from icecream import ic # for debugging
from scipy import stats

today = date.today()

#converting from 2nd unit to 1st
pc_kpc     = 1e3  # number of pc in one kpc
cm_km      = 1e5  # number of cm in one km
s_day      = 24*3600  # number of seconds in one day
s_min      = 60  # number of seconds in one hour
s_hr       = 3600  # number of seconds in one hour
cm_Rsun    = 6.957e10  # solar radius in cm
g_Msun     = 1.989e33  # solar mass in g
cgs_G      = 6.674e-8
cms_c      = 2.998e10
g_mH       = 1.6736e-24
g_me       = 9.10938e-28
cgs_h      = 6.626e-27
deg_rad    = 180e0/np.pi
arcmin_deg = 60e0
arcsec_deg = 3600e0
cm_kpc     = 3.086e+21  # number of centimeters in one parsec
cm_pc      = cm_kpc/1e+3
s_Myr      = 1e+6*(365*24*60*60)  # megayears to seconds

# base_path   = os.environ.get('MY_PATH')
base_path = os.getcwd()
galaxy_name = os.environ.get('galaxy_name')

#########################################################################################
# available functions in the code 
# assigning func_number to each function
# 1 : observables against observables
# 2 : observable vs radius
# 3 : B_iso vs B_ord plot (including M31Alt model outputs)
# 4 : scale_length calculation for B strength plots
# 5 : outputs vs r and r/r25 (including M31Alt model outputs)
# 6 : outputs vs outputs
# 7 : miscellaneous magnetic outputs plots
#     magnetic field ratios vs radius and r/r25 (breg to btot)
#     magnetic field ratios vs radius and r/r25 (bord to btot)
#     magnetic field ratios vs radius and r/r25 (bord to breg)
# 8: plot taue/taur vs r and r/r25
# 9: plot omega*tau vs r and r/r25

func_number = 9 # change this to choose what to plot
print('Function number:', func_number)

# going to supplementary_data folder where files are stored
new_dir_supp_data = os.path.join(base_path, 'results')
plot_save_dir     = os.path.join(new_dir_supp_data,'comparison_plots')

# going to model_data folder where interpolated data is stored
new_dir_model_data = os.path.join(base_path, 'data')

# going to model_data folder where combined data is stored
combined_obs_data = os.path.join(base_path, 'data','model_data')

os.chdir(plot_save_dir)
# make a folder to save radial variation plots in new_dir_supp_data
os.makedirs('plots',              exist_ok=True) # if the folder already exists, it will not create a new one
os.makedirs('plots_r25',          exist_ok=True) 
# make a folder to save o/p vs i/p plots in new_dir_supp_data
os.makedirs('output_vs_obs',      exist_ok=True)
# make a folder to save log o/p vs log i/p plots in new_dir_supp_data
os.makedirs('output_vs_obs_log',  exist_ok=True)
# make a folder to save o/p vs o/p plots in new_dir_supp_data
os.makedirs('output_vs_output',   exist_ok=True)
# make a folder to save i/p vs i/p plots in new_dir_supp_data
os.makedirs('obs_vs_obs',         exist_ok=True)
# make a folder to store miscellaneous plots
os.makedirs('miscellaneous',      exist_ok=True)
# make a folder to save scale length plots and calculations
os.makedirs('scale_length_B',     exist_ok=True)
# make a folder to save observable vs r data (interpolated)
os.makedirs('obs_vs_radius_supp', exist_ok=True)

os.chdir(plot_save_dir)

def open_csv_file(string):
    for file in os.listdir():
        if string in file:
            # Read the CSV file into a DataFrame
            data = pd.read_csv(file)
            return data

ax = plt.gca()

# edits starting from June 19th 2024
def insert_averages(arr):
    result = []
    for i in range(len(arr) - 1):
        # Append current element
        result.append(arr[i])
        # Calculate and append average of current and next element
        average = (arr[i] + arr[i+1]) / 2
        result.append(average)
    # Append the last element
    result.append(arr[-1])
    return result

def fill_between_err(x, quan_f, quan_err, color = 'red', alpha = 0.2, error_exists = True):

    # convert df to list
    x = x.tolist()
    quan_f = quan_f.tolist()
    quan_err = quan_err.tolist()

    # convert list to np.array
    x = np.array(x)
    quan_f = np.array(quan_f)
    quan_err = np.array(quan_err)

    # x  = kpc_r
    y1 = quan_f + quan_err
    y2 = quan_f - quan_err

    # Insert averages
    x = insert_averages(x)
    
    # divide the x list into lists of 2 adjacent elements and append it to master_x
    master_x = []
    for i in range(len(x) - 1):
        master_x.append([x[i], x[i+1]])
    
    yo1 = []
    yo2 = []
    for i in range(len(y1)-1):
        yo1.append(y1[i])
        yo1.append(y1[i] if y1[i] > y1[i+1] else y1[i+1])
        yo2.append(y2[i])
        yo2.append(y2[i+1] if y2[i] > y2[i+1] else y2[i])
    yo1.append(y1[-1])
    yo2.append(y2[-1])

    # store first and last element of each list
    x_first   = x[0]
    x_last    = x[-1]
    yo1_first = yo1[0]
    yo1_last  = yo1[-1]
    yo2_first = yo2[0]
    yo2_last  = yo2[-1]

    # remove even indexed terms from all three lists
    x   = [x[i] for i in range(len(x)) if i%2 != 0]
    yo1 = [yo1[i] for i in range(len(yo1)) if i%2 != 0]
    yo2 = [yo2[i] for i in range(len(yo2)) if i%2 != 0]

    # add first and last elements to new list
    x.insert(0, x_first)
    x.append(x_last)
    yo1.insert(0, yo1_first)
    yo1.append(yo1_last)
    yo2.insert(0, yo2_first)
    yo2.append(yo2_last)

    # fill error 
    if error_exists:
        plt.fill_between(x, yo1, yo2, alpha=alpha, facecolor=color, where = None, interpolate=True, edgecolor=None)  
    else:
        return
    
def abline(slope, xmin, xmax, factor, legend_label, colour):
    """Plot a curve with exponent and intercept"""
    axes = plt.gca()
    # x_vals = np.array(axes.get_xlim())
    x_vals = np.linspace(xmin, xmax, 100)
    
    # intercept = ymin
    # y_vals = slope * x_vals
    y_vals = (x_vals**slope)*factor

    plt.plot(x_vals, y_vals, '--', color = colour, label=legend_label)

# loading the plotted data
galaxies = ['m31', 'm33', 'm51', 'ngc6946']
dist_new = [dist*u.Mpc for dist in [0.78, 0.84, 8.5, 7.72]]
dist_old = [dist*u.Mpc for dist in [0.785,0.84,8.2,6.8]]

log_d25_arcmin_paper2 = [3.25, 2.79, 2.14, 2.06] # from LEDA
d25_arcmin_paper2     = 0.1*(10**np.array(log_d25_arcmin_paper2))
r_25_arcmin_paper2    = d25_arcmin_paper2/2
dist_Mpc_paper2       = [0.78, 0.84, 8.5, 7.72]
# convert arcmin to radius in kpc using distance to these galaxies
r_25_new_kpc          = [r*dist_Mpc_paper2[i]*1000/(arcmin_deg*deg_rad) for i,r in enumerate(r_25_arcmin_paper2)]

# Initialize an empty dictionary
galaxy_data     = {}
galaxy_data_err = {}
galaxy_data_mag = {}
galaxy_obs      = {}

# loading the plotted data
galaxies = ['m31', 'm33', 'm51', 'ngc6946']

# color map for each galaxy
color_map = {'m31': 'blue', 'm33': 'green', 'm51': 'red', 'ngc6946': 'purple'}

params_dict = {'m31':     {'moldat': 'No', 'tau':'taue', 'ks':'N', 'u':'datamaker', 'h':'root_find', 'z': 15.0, 'psi': 1.0, 'ca': 4.0, 'beta': 8.0, 'A': 1.414},
               'm33':     {'moldat': 'No', 'tau':'taue', 'ks':'N', 'u':'datamaker', 'h':'root_find', 'z': 10.0, 'psi': 1.0, 'ca': 4.0, 'beta': 8.0, 'A': 1.414}, 
               'm51':     {'moldat': 'No', 'tau':'taue', 'ks':'N', 'u':'datamaker', 'h':'root_find', 'z': 15.0, 'psi': 1.0, 'ca': 4.0, 'beta': 8.0, 'A': 1.414}, 
               'ngc6946': {'moldat': 'No', 'tau':'taue', 'ks':'N', 'u':'datamaker', 'h':'root_find', 'z': 30.0, 'psi': 1.0, 'ca': 4.0, 'beta': 8.0, 'A': 1.414}}

# get today's date
today = str(date.today())

for galaxy in galaxies:
    # go to specific folder based on params of each galaxy and today's date
    # folder name format: 2024-06-30,moldat_No,taue,z_15.0,psi_2.0,ca_4.0,beta_8.0,A_1.414

    os.chdir(new_dir_supp_data+'\{}'.format(galaxy)+r'\2024-10-30,moldat_{},{},KS_{},u_{},h_{},z_{},psi_{},ca_{},beta_{},A_{}'.format(params_dict[galaxy]['moldat'], params_dict[galaxy]['tau'], params_dict[galaxy]['ks'], 
                                                                        params_dict[galaxy]['u'], params_dict[galaxy]['h'],
                                                                        params_dict[galaxy]['z'], params_dict[galaxy]['psi'], 
                                                                        params_dict[galaxy]['ca'], params_dict[galaxy]['beta'], 
                                                                        params_dict[galaxy]['A']))
    
    # loading the model outputs to df
    data     = open_csv_file('model_outputs')
    data_err = open_csv_file('error_output')
    data_mag = open_csv_file('magnetic_output')

    # Add the data to the dictionary
    galaxy_data[galaxy]     = data
    galaxy_data_err[galaxy] = data_err
    galaxy_data_mag[galaxy] = data_mag

    # do the same for M31Alt results
    if galaxy == 'm31':
        os.chdir(new_dir_supp_data+'\{}'.format(galaxy)+r'\2024-10-30,moldat_{},{},KS_{},u_{},h_{},z_{},psi_2.0,ca_{},beta_{},A_{}'.format(params_dict[galaxy]['moldat'], params_dict[galaxy]['tau'], params_dict[galaxy]['ks'], 
                                                                        params_dict[galaxy]['u'], params_dict[galaxy]['h'],
                                                                        params_dict[galaxy]['z'], params_dict[galaxy]['ca'], 
                                                                        params_dict[galaxy]['beta'], params_dict[galaxy]['A']))
    
        # loading the model outputs to df
        data_M31Alt     = open_csv_file('model_outputs')
        data_err_M31Alt = open_csv_file('error_output')
        data_mag_M31Alt = open_csv_file('magnetic_output')

        # Add the data to the dictionary
        galaxy_data['M31Alt']     = data_M31Alt
        galaxy_data_err['M31Alt'] = data_err_M31Alt
        galaxy_data_mag['M31Alt'] = data_mag_M31Alt

    # loading interpolated data for the galaxy
    os.chdir(new_dir_model_data)
    ip_data = open_csv_file('data_interpolated_{}'.format(galaxy))
    # Add interpolated data to the dictionary
    galaxy_obs[galaxy] = ip_data

    os.chdir(new_dir_supp_data) # go back to the parent directory: supplementary_data

# galaxy wise output 
m31_df     = galaxy_data['m31']
m33_df     = galaxy_data['m33']
m51_df     = galaxy_data['m51']
ngc6946_df = galaxy_data['ngc6946']
m31Alt_df  = galaxy_data['M31Alt']

# galaxy wise error
m31_df_err     = galaxy_data_err['m31']
m33_df_err     = galaxy_data_err['m33']
m51_df_err     = galaxy_data_err['m51']
ngc6946_df_err = galaxy_data_err['ngc6946']

# add inverse gamma error to each df
for gal in galaxies:
    galaxy_data_err[gal]['inv_gamma_err'] = np.sqrt(((1/galaxy_data[gal]['gamma']**2) * galaxy_data_err[gal]['gamma_err'])**2)
    if gal == 'm31': # do the same for M31Alt
        galaxy_data_err['M31Alt']['inv_gamma_err'] = np.sqrt(((1/galaxy_data['M31Alt']['gamma']**2) * galaxy_data_err['M31Alt']['gamma_err'])**2)

# galaxy wise observed data
m31_obs     = galaxy_obs['m31']
m33_obs     = galaxy_obs['m33']
m51_obs     = galaxy_obs['m51']
ngc6946_obs = galaxy_obs['ngc6946']

# list of headings in the data
quantities       = list(m31_df.columns)
quantities_err   = list(m31_df_err.columns)
ip_data_cols     = list(m31_obs.columns)

# units for the quantities
units_for_axis   = ['(kpc)', '(kpc)', '(kpc)', '(km/s)', '(km/s)', '(km/s)', r'($\mu$G)', r'($\mu$G)', r'($\mu$G)', 
                   '(degree)', '(degree)', '(km/s)', '(Myr)', '(Myr)', ' ', ' ', r'(Gyr$^{-1}$)', '(Gyr)']
quants_for_title = ['Radius', 'Scale height', 'Correlation length', 'Turbulent speed', 'Sound speed', 'Total speed', 'Total B', 'Regular B', 'Ordered B',
                   'Ordered pitch angle', 'Mean-field pitch angle', r'$\alpha_\mathrm{k}$', 'Eddy-turnover time', 'Renovation time', r'$\mathrm{D}/\mathrm{D}_{\mathrm{c}}$', 'Mach number', 'Local growth rate', 'Local e-folding time']
symbols_for_axis = [r'$r$', r'$h$', r'$l$', r'$u$', r'$c_\mathrm{s}$', r'$w$', r'$B_{\mathrm{tot}}$', r'$B_{\mathrm{reg}}$', r'$B_{\mathrm{ord}}$', 
                    r'$p_\mathrm{ord}$', r'$p_\mathrm{reg}$',  r'$\alpha_\mathrm{k}$', r'$\tau_\mathrm{e}$', r'$\tau_\mathrm{r}$', r'$\mathrm{D}/\mathrm{D}_{\mathrm{c}}$', r'$\mathcal{M} = u/c_\mathrm{s}$', r'$\gamma$', r'$\frac{1}{\gamma}$']

# units for observables
units_for_obs        = ['(kpc)', r'$(\mathrm{g}/\mathrm{cm}^2)$', r'$(\mathrm{g}/\mathrm{cm}^2)$', r'$(\mathrm{g}/\mathrm{cm}^2)$', ' ', '(1/s)', r'$(\mathrm{g}/\mathrm{s} \mathrm{ cm}^2)$', '(K)']
quants_for_title_obs = ['Radius', r'$\Sigma_{\mathrm{tot}}$', r'$\Sigma_{\mathrm{HI}}$', r'$\Sigma_{\mathrm{H}_\mathrm{2}}$', 'q', r'$Omega$', r'$\Sigma_{\mathrm{SFR}}$', 'Temperature']
symbols_for_axis_obs = [r'$r$', r'$\Sigma_{\mathrm{tot}}$', r'$\Sigma_{\mathrm{HI}}$', r'$\Sigma_{\mathrm{H}_\mathrm{2}}$', 'q', r'$Omega$', r'$\Sigma_{\mathrm{SFR}}$', 'T']


# add plot features
m                 = 9 #marker size
lw                = 3
dm                = 2.5
fs                = 20
lfs               = 10
leg_textsize      = 18
axis_textsize     = 30
title_textsize    = 25
tick_width        = 2
hd                = 1.6 #handlelength: changes length of line in legend
legend_labelspace = 0.17 #handletextpad: gap between label and symbol in legend

rc = {"font.family" : "serif", "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
matplotlib.rc('xtick', labelsize=fs)
matplotlib.rc('ytick', labelsize=fs)
matplotlib.ticker.AutoMinorLocator(n=None)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rcParams["xtick.minor.visible"] =  True
plt.rcParams["ytick.minor.visible"] =  True
plt.rcParams["legend.loc"] = 'upper right'
plt.rcParams["errorbar.capsize"] = 2
plt.rcParams['figure.max_open_warning'] = 500  # warning about open figures come up when 100+ images are open together
rc = {"font.family" : "serif", "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

# ..............................  PLOTTING BEGINS ............................ #

if func_number == 1:
    # plotting observables against observables
    for i in range(1,len(ip_data_cols)):
        for j in range(1,len(ip_data_cols)):
            plt.figure(figsize=(11, 6))
        # plt.figure(figsize=(11, 6))
            if i != j:
                for galaxy in galaxies:
                    
                    plt.plot(galaxy_obs[galaxy].iloc[:,j], galaxy_obs[galaxy].iloc[:,i],color_map[galaxy], marker='o', linewidth=lw, label=galaxy.upper())
                    title   = quants_for_title_obs[i] + ' vs '+ quants_for_title_obs[j] # y vs x
                    x_label = symbols_for_axis_obs[j] + ' ' + units_for_obs[j]
                    y_label = symbols_for_axis_obs[i] + ' ' + units_for_obs[i]

                    plt.xlabel(x_label, fontsize=fs)
                    plt.ylabel(y_label, fontsize=fs)

                    plt.tick_params(width=tick_width)
                    plt.minorticks_on()

                    plt.title(title, fontsize=title_textsize, weight='bold')
                    plt.legend(fontsize=lfs, frameon=False, handlelength=hd, ncol=1, prop={
                        'size': leg_textsize,  'family': 'Times New Roman'}, fancybox=True, framealpha=0.9, handletextpad=legend_labelspace, columnspacing=0.7)
                    os.chdir(plot_save_dir+'\obs_vs_obs')

                    # make a folder for each output quantity
                    os.makedirs('{}'.format(i), exist_ok=True)
                    # os.makedirs('{}'.format(j), exist_ok=True)
                    os.chdir('{}'.format(i))
                    plt.savefig('{}_vs_{}'.format(i,j))

                    os.chdir('..')
                    os.chdir('..')
                plt.close()

####################################################################################################################################

# observable vs r
if func_number == 2:
    m31_obs_list     = ['kpc_r','sigma_tot','sigma_HI_claude','sigma_H2','q','\Omega','sigma_sfr','T']
    m33_obs_list     = ['kpc_r','sigma_tot_up52','sigma_HI','sigma_H2','q','\Omega','sigma_sfr','T']
    m51_obs_list     = ['kpc_r','sigma_tot','sigma_HI','sigma_H2','q','\Omega','sigma_sfr','T']
    ngc6946_obs_list = ['kpc_r','sigma_tot','sigma_HI','sigma_H2','q','\Omega','sigma_sfr','T']

    obs_name_list = [m31_obs_list,m33_obs_list,m51_obs_list,ngc6946_obs_list]

    for i in range(1,len(ip_data_cols)):
        plt.figure(figsize=(11, 6))
        # one plot containing all galaxies per field strength     
        # change margin size
        plt.rcParams['axes.xmargin'] = 0
        plt.rcParams['axes.ymargin'] = 0
        plt.xticks(np.arange(0, 21, 4)) 

        plt.rcParams['axes.titley']   = 1.0    # y is in axes-relative coordinates.
        plt.rcParams['axes.titlepad'] = -25    # pad is in points...
        # Switch on major and minor ticks on y axis
        plt.tick_params(width=tick_width)

        plt.minorticks_on()
        plt.tick_params(axis='y', which='both', right=True, labelsize=fs, labelright=False, width = 2)
        # Enable ticks on the top of the plot without labels
        plt.tick_params(axis='x', which='both', top=True, labeltop=False, width = 3)

        # Adjust the top margin
        plt.subplots_adjust(top    = 0.95)  # Adjust this value as needed to reduce the upper margin
        plt.subplots_adjust(bottom = 0.15)  # Adjust this value as needed to reduce the upper margin

        plt.xlabel('Radius (kpc)', fontsize=fs, labelpad=1) # labelpad change gap between axis and axis title
        plt.ylabel('{} {}'.format(symbols_for_axis_obs[i],units_for_obs[i]), fontsize=fs, labelpad=1)
        for galaxy in galaxies:
            if galaxy == 'ngc6946':
                plt.plot(galaxy_obs[galaxy]['kpc_r'], galaxy_obs[galaxy][obs_name_list[galaxies.index(galaxy)][i]], color=color_map[galaxy],marker='o', linewidth=lw, label='NGC 6946')
            else:
                plt.plot(galaxy_obs[galaxy]['kpc_r'], galaxy_obs[galaxy][obs_name_list[galaxies.index(galaxy)][i]], color=color_map[galaxy],marker='o', linewidth=lw, label=galaxy.upper())
                    
            plt.legend(fontsize=lfs, frameon=False, handlelength=hd, ncol=1, loc = 'upper right', prop={
                    'size': leg_textsize-4,  'family': 'Times New Roman'}, fancybox=True, framealpha=0.9, handletextpad=legend_labelspace, columnspacing=0.7)

        os.chdir(plot_save_dir+'\obs_vs_radius_supp')
        plt.savefig('{}'.format(obs_name_list[galaxies.index(galaxy)][i]))
        os.chdir('..')
        plt.close()

    model_quants_sym   = ['$h$', '$l$', '$u$', r'$\tau_e$', r'$B_\mathrm{ord}$', r'$B_\mathrm{reg}$', r'$B_\mathrm{tot}$' , r'$\tan{p_B}$']
    model_quants_units = ['(pc)', '(pc)', '(km/s)', '(Myr)', '(µG)', '(µG)', '(µG)', '']
    model_quants_title = ['Scale Height', 'Correlation Length', 'Turbulent velocity', 'Eddy turnover time', 'Ordered Field', 'Regular Field', 'Total Field', 'Mean field pitch Angle']
    quant_compare_with_scal_exp_m31 = [m31_df['h'], m31_df['l'], m31_df['u'], m31_df['taue_e'], m31_df['Bord'], m31_df['Breg'], m31_df['Btot'], m31_df['pB']]
    quant_compare_with_scal_exp_m33 = [m33_df['h'], m33_df['l'], m33_df['u'], m33_df['taue_e'], m33_df['Bord'], m33_df['Breg'], m33_df['Btot'], m33_df['pB']]
    quant_compare_with_scal_exp_m51 = [m51_df['h'], m51_df['l'], m51_df['u'], m51_df['taue_e'], m51_df['Bord'], m51_df['Breg'], m51_df['Btot'], m51_df['pB']]
    quant_compare_with_scal_exp_ngc6946 = [ngc6946_df['h'], ngc6946_df['l'], ngc6946_df['u'], ngc6946_df['taue_e'], ngc6946_df['Bord'], ngc6946_df['Breg'], ngc6946_df['Btot'], ngc6946_df['pB']]
    quant_compare_with_scal_exp = [quant_compare_with_scal_exp_m31, quant_compare_with_scal_exp_m33, quant_compare_with_scal_exp_m51, quant_compare_with_scal_exp_ngc6946]
    
    # manually entering slopes of each qty from Chamandy+24 Table 3,4,5,6
    # slopes in this order: 'sigma_tot', 'sigma', 'q', '\\Omega', 'sigma_sfr', 'T' 
    exp_no_KS_sa = {'h': [-1, 0, 0, 0, 0, 1], 'l': [-0.37, -0.37, 0, 0, 0, 0.21], 'u': [-0.16, -0.5, 0, 0, 1/3, 0.27], 'taue_e': [-0.21, 0.12, 0, 0, -1/3, -0.07], 'Bord': [np.sqrt(0.23**2 + 0.13**2)/2, np.sqrt(0.07**2 + 0.13**2)/2, np.sqrt(0.5**2 + 0.5**2)/2, np.sqrt(0.5**2 + 1**2)/2, np.sqrt((1/6)**2 + 0**2)/2, -np.sqrt(0.26**2 + 0.29**2)/2], 'Breg': [0.13, 0.13, 0.5, 1, 0, -0.29], 'Btot': [np.sqrt(0.23**2 + 0.13**2 + 0.34**2)/3, np.sqrt(0.07**2 + 0.13**2 + 0.003**2)/3, np.sqrt(0.5**2 + 0.5**2 + 0**2)/3, np.sqrt(0.5**2 + 1**2 + 0**2)/3, np.sqrt((1/6)**2 + 0**2 + (1/3)**2)/3, -np.sqrt(0.26**2 + 0.29**2 + 0.23**2)/3], 'pB': [1.46, -0.87, -1, -1, 1/3, -1.52]}
    exp_no_KS_sb = {'h': [-1.49, -1.48, 0, 0, 0.99, 0.33], 'l': [-0.55, -0.92, 0, 0, 0.37, -0.04], 'u': [-0.24, -0.74, 0, 0, 0.5, 0.17], 'taue_e': [-0.31, -0.18, 0, 0, -0.13, -0.21], 'Bord': [np.sqrt(0.59**2 + 0.19**2)/2, np.sqrt(1.15**2 + 0.32**2)/2, np.sqrt(0.5**2 + 0.5**2)/2, np.sqrt(0.5**2 + 1**2)/2, -np.sqrt(0.56**2 + 0.13**2)/2, np.sqrt(0.23**2 - 0.21**2)/2], 'Breg': [0.19, 0.32, 0.5, 1, -0.13, -0.21], 'Btot': [np.sqrt(0.59**2 + 0.19**2 + 0.74**2)/3, np.sqrt(1.15**2 + 0.32**2 + 1.24**2)/3, np.sqrt(0.5**2 + 0.5**2)/3, np.sqrt(0.5**2 + 1**2)/3, -np.sqrt(0.56**2 + 0.13**2 + 0.5**2)/3, np.sqrt(0.23**2 - 0.21**2 + 0.33**2)/3], 'pB': [2.17, 1.29, -1, -1, -1.12, -0.54]}

    # get keys for both dictionaries
    exp_keys_sa = list(exp_no_KS_sa.keys())
    exp_keys_sb = list(exp_no_KS_sb.keys())

    # get values for both dictionaries
    exp_vals_sa = list(exp_no_KS_sa.values())
    exp_vals_sb = list(exp_no_KS_sb.values())

    # remove sigma_H2 from ip_data_cols
    ip_data_cols_short = [col for col in ip_data_cols if col != 'sigma_H2']
    ic(ip_data_cols_short)
    #remove 3rd element from symbols_for_axis_obs
    del symbols_for_axis_obs[3]
    # ic(symbols_for_axis_obs)
    del units_for_obs[3]
    del quants_for_title_obs[3]

    # find max and min of each observable by comparing the data for all galaxies
    max_obs = []
    min_obs = []
    for i in range(1, len(ip_data_cols_short)):
        obs = []
        for galaxy in galaxies:
            # remove the column containing sigma_H2 from galaxy_obs[galaxy]
            # ic(galaxy)
            # ic(galaxy_obs[galaxy].columns)
            if i == 1: # try removing the sigma_H2 column only once
                galaxy_obs[galaxy] = galaxy_obs[galaxy].drop('sigma_H2', axis=1)
            obs.append(galaxy_obs[galaxy].iloc[:,i].tolist())
        max_obs.append(max(max(sublist) for sublist in obs))
        min_obs.append(min(min(sublist) for sublist in obs))

    # find max and min of each output by comparing the data for all galaxies
    max_out = []
    min_out = []
    for i in range(len(quant_compare_with_scal_exp_m31)):
        out = []
        for galaxy in galaxies:
            out.append(quant_compare_with_scal_exp[galaxies.index(galaxy)][i].tolist())
        max_out.append(max(max(sublist) for sublist in out))
        min_out.append(min(min(sublist) for sublist in out))

    #manually entering factors to be multiplied 

    def factor_selector(i,j): # i denotes output (starts from 0 --> scale height), j denotes observables (starts from 1 --> sigma_tot)
    # in the order of outputs in model_quants_sym
        factors_tot    = [1e0, 1e1, 10**0.5, 1e0, 1e1, 1e1, 1e1, 10**3.5]
        factors_tot_b  = [1e0, 1e1, 10**0.5, 1e0, 1e1, 1e1, 1e1, 10**3.5]

        factors_HI     = [1e3, 1e0, 1e0, 1e1, 1e1, 1e1, 1e1, 1e-2]
        factors_HI_b   = [1e-2, 1e-1, 10**(-0.5), 1e0, 10**(1.5), 1e1, 1e3, 1e4]

        factors_q      = [1e2, 10**1.5, 1e1, 1e1, 1e1, 1e0, 1e1, 1e0]
        factors_q_b    = [1e3, 10**1.5, 1e1, 1e0, 10**(1.5), 1e1, 1e1, 1e1]

        factors_om     = [1e2, 1e2, 1e1, 1e0, 1e10, 1e15, 1e7, 1e-14]
        factors_om_b   = [1e2, 1e2, 1e1, 1e1, 1e8, 1e14, 1e6, 1e-17]
        
        factors_SFR    = [1e3, 1e2, 1e7, 1e-6, 1e2, 1e0, 1e4, 1e6] 
        factors_SFR_b  = [1e24, 1e10, 1e11, 1e-2, 1e-6, 1e-3, 1e-5, 1e-24]
        
        factors_T      = [10**-1.5, 10**0.5, 1e0, 1e1, 10**1.6, 1e2, 1e2, 1e7]
        factors_T_b    = [1e1, 1e2, 1e1, 1e2, 10**(0.5), 1e1, 1e1, 1e3]

        factors        = [factors_tot, factors_HI, factors_q, factors_om, factors_SFR, factors_T]
        factors_b      = [factors_tot_b, factors_HI_b, factors_q_b, factors_om_b, factors_SFR_b, factors_T_b]
        return factors[j-1][i], factors_b[j-1][i]

    def legend_loc_selectors(i,j):
        ll = 'lower left'
        lr = 'lower right'
        ul = 'upper left'
        ur = 'upper right'

        loc_tot = [ll,ur,ul,ur,ul,ul,ul,ul]
        loc_HI  = [ll,ur,ul,ll,ul,ul,ul,ll]
        loc_q   = [lr,ur,lr,ur,lr,ur,lr,ur]
        loc_om  = [ll,ur,lr,ur,lr,lr,lr,lr]
        loc_SFR = [ul,ur,lr,ur,ul,ul,lr,lr]
        loc_T   = [lr,ur,ll,ur,ll,ll,ll,ur]

        locs = [loc_tot, loc_HI, loc_q, loc_om, loc_SFR, loc_T]
        return locs[j-1][i]

    for i in tqdm(range(len(quant_compare_with_scal_exp_m31))): # for each output quantity
        
            for j in range(1,len(ip_data_cols_short)): # for each observable excluding kpc_r, after removing sigma_H2
            
                plt.figure(figsize=(11, 6)) # one plot for each output quantity vs observable
                
                # plot a line with slope from the dictionary
                # make a function similar to abline that takes the slope and the limits of x and y
                int_a, int_b = factor_selector(i,j)
                abline(exp_vals_sa[i][j-1], min_obs[j-1], max_obs[j-1], int_a, 'slope = {}'.format(round(exp_vals_sa[i][j-1],2)), 'k')
            
                for galaxy in galaxies:
                    plt.plot(galaxy_obs[galaxy].iloc[:,j], quant_compare_with_scal_exp[galaxies.index(galaxy)][i], color_map[galaxy], marker=' ', markersize=m, linestyle='-', linewidth = lw, label=galaxy.upper())
                    title   = '{} vs {}'.format(model_quants_title[i], symbols_for_axis_obs[j])
                    x_label = symbols_for_axis_obs[j] + ' ' + units_for_obs[j]
                    y_label = model_quants_sym[i] + ' ' + model_quants_units[i]

                    plt.xlabel(x_label, fontsize=fs)
                    plt.ylabel(y_label, fontsize=fs)
                    
                    # make x and y log-log
                    plt.xscale('log')
                    plt.yscale('log')

                    plt.tick_params(width=tick_width)
                    plt.minorticks_on()
                    plt.title(title, fontsize=title_textsize, weight='bold')

                    plt.legend(fontsize=lfs, frameon=False, handlelength=hd, ncol=1, prop={
                        'size': leg_textsize,  'family': 'Times New Roman'}, fancybox=True, framealpha=0.9, handletextpad=legend_labelspace, columnspacing=0.7, loc=legend_loc_selectors(i,j))
                    os.chdir('output_vs_obs_log')
                    os.makedirs('{}'.format(j), exist_ok=True)
                    os.chdir('{}'.format(j))
                    plt.savefig('{}_vs_{}'.format(model_quants_title[i], j))
                    
                    os.chdir('..')
                    os.chdir('..')
                plt.close()


if func_number == 3:
    # reproducing b_iso vs b_ord plot from Beck+19

    b_iso_dict     = {}
    b_ord_dict     = {}
    b_ord_iso      = {}
    b_ord_iso_err  = {}

    plt.figure(figsize=(9, 6))
    # one plot containing all galaxies per field strength     
    # change margin size
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0
    plt.xticks(np.arange(0, 21, 4)) 

    plt.rcParams['axes.titley']   = 1.0    # y is in axes-relative coordinates.
    plt.rcParams['axes.titlepad'] = -10  # pad is in points...
    # Switch on major and minor ticks on y axis
    plt.tick_params(width=tick_width)

    plt.minorticks_on()
    plt.tick_params(axis='y', which='both', right=True, labelsize=fs, labelright=False, width = 2)
    # Enable ticks on the top of the plot without labels
    plt.tick_params(axis='x', which='both', top=True, labeltop=False, width = 3)

    # Adjust the top margin
    plt.subplots_adjust(top    = 0.95)  # Adjust this value as needed to reduce the upper margin
    plt.subplots_adjust(bottom = 0.15)  # Adjust this value as needed to reduce the upper margin

    plt.xlabel('Radius (kpc)', fontsize=fs, labelpad=2) # labelpad change gap between axis and axis title
    plt.ylabel(r'$B_{\mathrm{ord}}/B_{\mathrm{iso}}$', fontsize=fs+2, labelpad=2) # labelpad change gap between axis and axis title

    for galaxy in galaxies:

        if galaxy == 'm31':
            b_iso_M31Alt = np.sqrt((galaxy_data['M31Alt']['Btot'])**2 - (galaxy_data['M31Alt']['Bord'])**2)
            b_iso_dict['M31Alt'] = b_iso_M31Alt
            b_ord_dict['M31Alt'] = galaxy_data[galaxy]['Bord']

            # ratio of b_ord to b_iso
            b_ord_iso['M31Alt'] = b_ord_dict['M31Alt']/b_iso_dict['M31Alt']

            # error in ratio
            b_ord_iso_err['M31Alt'] = b_ord_iso['M31Alt']*np.sqrt(((galaxy_data_mag['M31Alt']['err_b_iso'])/galaxy_data_mag['M31Alt']['b_iso'])**2 + (galaxy_data_err['M31Alt']['Bord_err']/galaxy_data['M31Alt']['Bord'])**2)
            
        b_iso = np.sqrt((galaxy_data[galaxy]['Btot'])**2 - (galaxy_data[galaxy]['Bord'])**2)
        b_iso_dict[galaxy] = b_iso
        b_ord_dict[galaxy] = galaxy_data[galaxy]['Bord']

        # ratio of b_ord to b_iso
        b_ord_iso[galaxy] = b_ord_dict[galaxy]/b_iso_dict[galaxy]

        # error in ratio
        b_ord_iso_err[galaxy] = b_ord_iso[galaxy]*np.sqrt(((galaxy_data_mag[galaxy]['err_b_iso'])/galaxy_data_mag[galaxy]['b_iso'])**2 + (galaxy_data_err[galaxy]['Bord_err']/galaxy_data[galaxy]['Bord'])**2)
        
        # plot b_ord_iso vs radius
        if galaxy == 'ngc6946':
            plt.plot(galaxy_data[galaxy]['radius'], b_ord_iso[galaxy], color=color_map[galaxy],marker='o', mfc='k', mec='k', markersize = m-7, linewidth=lw, label='NGC 6946')
        elif galaxy == 'm31':
            plt.plot(galaxy_data[galaxy]['radius'], b_ord_iso[galaxy], color=color_map[galaxy],marker='o', mfc='k', mec='k', markersize = m-7, linewidth=lw, label=galaxy.upper())
            plt.plot(galaxy_data['M31Alt']['radius'], b_ord_iso['M31Alt'], color='cyan' ,marker='o', mfc='k', mec='k', markersize = m-7, linewidth=lw, label='M31Alt')

        else:
        # M51 and M33
            plt.plot(galaxy_data[galaxy]['radius'], b_ord_iso[galaxy], color=color_map[galaxy],marker='o', mfc='k', mec='k', markersize = m-7, linewidth=lw, label=galaxy.upper())
        
        fill_between_err(galaxy_data[galaxy]['radius'], b_ord_iso[galaxy], b_ord_iso_err[galaxy], color = color_map[galaxy], alpha = 0.2, error_exists = True)
        if galaxy == 'm31':
            fill_between_err(galaxy_data['M31Alt']['radius'], b_ord_iso['M31Alt'], b_ord_iso_err['M31Alt'], color = 'cyan', alpha = 0.2, error_exists = True)

        if galaxy == 'ngc6946':
            os.chdir(base_path+'\data\supplementary_data'+r'\{}'.format(galaxy))

            # os.chdir(new_dir_supp_data+r'\{}'.format(galaxy))
            # loading Beck+19 data for comparison
            ngc6946_bord_biso = open_csv_file('bord_biso_Beck19')
            os.chdir(new_dir_supp_data)

            r_ngc6946         = ngc6946_bord_biso['r']
            bord_biso_ngc6946 = ngc6946_bord_biso['bord_biso']
            plt.plot(r_ngc6946, bord_biso_ngc6946, color=color_map[galaxy], marker='o', mfc=color_map[galaxy], mec=color_map[galaxy], markersize = m-5, linestyle=' ', linewidth=lw, label='NGC 6946 (B19)')

        elif galaxy == 'm51':
                # os.chdir(new_dir_supp_data+r'\{}'.format(galaxy))
                os.chdir(base_path+'\data\supplementary_data'+r'\{}'.format(galaxy))
                # loading Beck+19 data for comparison
                m51_bord_biso = open_csv_file('bord_biso_Beck19')

                os.chdir(new_dir_supp_data)
                r_m51         = m51_bord_biso['r']
                bord_biso_m51 = m51_bord_biso['bord_biso']

                plt.plot(r_m51, bord_biso_m51, color=color_map[galaxy], marker='o', mfc=color_map[galaxy], mec=color_map[galaxy], markersize = m-5, linestyle=' ', linewidth=lw, label='M51 (B19)')

        plt.xlim(left   = 0,   right = 19)
        plt.ylim(bottom = 0, top   = 1.6)
        plt.legend(fontsize=lfs, frameon=False, handlelength=hd, ncol=1, loc = 'upper right', prop={
            'size': leg_textsize-2,  'family': 'Times New Roman'}, fancybox=True, framealpha=0.9, handletextpad=legend_labelspace, columnspacing=0.7)

    os.chdir(plot_save_dir+'\miscellaneous')
    plt.savefig('bord_biso')
    os.chdir('..')
    plt.close()


if func_number == 4:
    # scale-length calculation for B strength plots
    B = ['B_tot', 'B_reg', 'B_ord']
    B_loc = [6,7,8] # indices of B strengths in quantities list
    slope_list_bord     = []
    err_slope_list_bord = []

    slope_list_breg     = []
    err_slope_list_breg = []

    slope_list_btot     = []
    err_slope_list_btot = []

    # Hyper-LEDA values
    log_d25_arcmin_paper2     = [3.25, 2.79, 2.14, 2.06]
    err_log_d25_arcmin_paper2 = [0.01, 0.01, 0.02, 0.01]
    d25_arcmin_paper2         = 0.1*(10**np.array(log_d25_arcmin_paper2))
    r25_arcmin_paper2         = d25_arcmin_paper2/2
    dist_Mpc_paper2           = [0.78, 0.84, 8.5, 7.72]
    # convert arcmin to radius in kpc using distance to these galaxies
    r25_galaxies              = [r*dist_Mpc_paper2[i]*1000/(arcmin_deg*deg_rad) for i,r in enumerate(r25_arcmin_paper2)]

    for i in range(1,len(quantities)): 
        if i in B_loc: # only for B-strengths
            plt.figure(figsize=(11, 6))
            # one plot containing all galaxies per field strength     
            # change margin size
            plt.rcParams['axes.xmargin'] = 0
            plt.rcParams['axes.ymargin'] = 0
            plt.xticks(np.arange(0, 21, 4)) 

            plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
            plt.rcParams['axes.titlepad'] = -10  # pad is in points...
            # Switch on major and minor ticks on y axis
            plt.tick_params(width=tick_width)

            plt.minorticks_on()
            plt.tick_params(axis='y', which='both', right=True, labelsize=fs, labelright=False, width = 2)
            # Enable ticks on the top of the plot without labels
            plt.tick_params(axis='x', which='both', top=True, labeltop=False, width = 3)
            
            # Adjust the top margin
            plt.subplots_adjust(top=0.95)  # Adjust this value as needed to reduce the upper margin
            plt.subplots_adjust(bottom=0.15)  # Adjust this value as needed to reduce the upper margin

            plt.xlabel('Radius (kpc)', fontsize=fs, labelpad=0.5) # labelpad change gap between axis and axis title
            
            for galaxy in galaxies:

                # perform straight-line fit for that quantity vs radius
                result = stats.linregress(galaxy_data[galaxy]['radius'], np.log10(galaxy_data[galaxy][quantities[i]]))

                if galaxy == 'ngc6946':
                    # slice the radius list to create a new list to include values greater than 6 kpc
                    ngc6946_data = galaxy_data[galaxy] 

                    if i == 6: # B_tot scale length in Beck+07 is calculated for r>3 kpc and r = 9 kpc is the extend of plots. so we use this range for B_tot
                        ngc6946_data = ngc6946_data[ngc6946_data['radius'] > 4.2]
                        ngc6946_data = ngc6946_data[ngc6946_data['radius'] < 21.0]
                    elif i  == 8: # B_ord scale length in Beck+07 is calculated for r>6 kpc and r = 9 kpc is the extend of plots. so we use this range for B_ord 
                        ngc6946_data = ngc6946_data[ngc6946_data['radius'] > 8.4]
                        ngc6946_data = ngc6946_data[ngc6946_data['radius'] < 21.0]
                    else:
                        ngc6946_data = ngc6946_data[ngc6946_data['radius'] > 4.2]
                        ngc6946_data = ngc6946_data[ngc6946_data['radius'] < 21.0]
 
                    # get the column with index i
                    ngc6946_mag = ngc6946_data[quantities[i]]
                    # take log base e 
                    ngc6946_mag = np.log10(ngc6946_mag)
                    # make it a list
                    ngc6946_mag_list = ngc6946_mag.tolist()

                    # get radius column and make it a list
                    ngc6946_r = ngc6946_data['radius'].tolist()
                    # perform straight-line fit for that quantity vs radius
                    result_ngc6946    = stats.linregress(ngc6946_r, ngc6946_mag_list)
                    slope_beck        = result_ngc6946.slope
                    intercept_beck    = result_ngc6946.intercept
                    slope_beck_stderr = result_ngc6946.stderr
                    r0_beck           = (-np.log10(np.e)/np.array(slope_beck))
                    err_r0_beck       = np.sqrt((r0_beck*slope_beck_stderr/slope_beck)**2)

                    print('slope, r0_beck, err_r0_beck:', i, ' ,', slope_beck, ' ,', r0_beck, ' ,', err_r0_beck)
            
                # Access the slope, intercept, and standard error of the slope
                slope        = result.slope
                intercept    = result.intercept
                slope_stderr = result.stderr

                # store slopes in respective lists
                if i == 6:
                    slope_list_btot.append(slope)
                    err_slope_list_btot.append(slope_stderr)
                    plt.ylim(top = 2, bottom = 0)
                    plt.ylabel(r'$\log{B_{\mathrm{tot}}}$', fontsize=fs, labelpad=0) # labelpad change gap between axis and axis title

                elif i == 7:
                    slope_list_breg.append(slope)
                    err_slope_list_breg.append(slope_stderr)
                    plt.ylim(top = 2, bottom = -0.5)
                    plt.ylabel(r'$\log{B_{\mathrm{reg}}}$', fontsize=fs, labelpad=0) # labelpad change gap between axis and axis title

                elif i == 8:
                    slope_list_bord.append(slope)
                    err_slope_list_bord.append(slope_stderr)
                    plt.ylim(top = 2, bottom = -0.5)
                    plt.ylabel(r'$\log{B_{\mathrm{ord}}}$', fontsize=fs, labelpad=0) # labelpad change gap between axis and axis title

                # use slope and intercept to create an array of field strengths and radii to plot
                log_B = slope*galaxy_data[galaxy]['radius'] + intercept

                if galaxy == 'ngc6946':
                    plt.plot(galaxy_data[galaxy]['radius'], np.log10(galaxy_data[galaxy][quantities[i]]), color=color_map[galaxy],marker='o', linewidth=lw, label='NGC 6946')
                else:
                    plt.plot(galaxy_data[galaxy]['radius'], np.log10(galaxy_data[galaxy][quantities[i]]), color=color_map[galaxy],marker='o', linewidth=lw, label=galaxy.upper())
                plt.plot(galaxy_data[galaxy]['radius'], log_B, color=color_map[galaxy], marker=' ', linestyle=':', linewidth=lw-1, label='m:{}, c:{}'.format(np.round(slope,3),np.round(intercept,3)))
                
                plt.legend(fontsize=lfs, frameon=False, handlelength=hd, ncol=2, loc = 'upper right', prop={
                'size': leg_textsize-4,  'family': 'Times New Roman'}, fancybox=True, framealpha=0.9, handletextpad=legend_labelspace, columnspacing=0.7)

            os.chdir(plot_save_dir+'\scale_length_B')
            plt.savefig('scale_length_{}'.format(B[B_loc.index(i)]))
            os.chdir('..')
        plt.close()

    # print a table of slope_lists
    # from tabulate import tabulate
    slope_lists       = [slope_list_btot, slope_list_breg, slope_list_bord]
    err_slope_lists   = [err_slope_list_btot, err_slope_list_breg, err_slope_list_btot]

    r0_lists           = []
    err_r0_lists       = []
    r0_r25_lists       = []
    err_r0_r25_lists   = []

    # scale length and error in r_0 calculation from slope
    for i in slope_lists:
        r0 = (-np.log10(np.e)/np.array(i))
        r0_lists.append(r0)
        err_r0 = np.sqrt((r0*np.array(err_slope_lists[slope_lists.index(i)])/np.array(i))**2)
        # err_r0 = np.sqrt(((np.log10(np.e)*np.array(err_slope_lists[slope_lists.index(i)]))/(np.array(i))**2)**2)
        err_r0_lists.append(err_r0)

    # calculating ratio of r0 to r25
    for i in r0_lists:
        r0_r25_list = []
        for j in range(len(i)):
            r0_r25 = i[j]/r25_galaxies[j]
            r0_r25_list.append(r0_r25)
        r0_r25_lists.append(r0_r25_list)

    # error r25
    err_r25_arcmin = np.full_like(r25_galaxies, np.log(10))*err_log_d25_arcmin_paper2*r25_arcmin_paper2 # array with 4 entries, corresponding to each galaxy
    # convert err_r25 to kpc distance
    err_r25_galaxies_kpc = np.array([r*dist_Mpc_paper2[i]*1000/(arcmin_deg*deg_rad) for i,r in enumerate(err_r25_arcmin)])

    # calculating error in ratio of r0 to r25

    for i in r0_r25_lists:
        # err_r0_r25 = np.sqrt((np.array(err_r0_lists[r0_r25_lists.index(i)])/np.array(r0_lists[r0_r25_lists.index(i)]))**2 + (err_r25_galaxies_kpc/np.array(r25_galaxies))**2)
        err_r0_r25 = i*np.sqrt((np.array(err_r0_lists[r0_r25_lists.index(i)])/np.array(r0_lists[r0_r25_lists.index(i)]))**2 + (err_r25_galaxies_kpc/np.array(r25_galaxies))**2)
            # print('---->',err_r0_r25) 
        # err_r0_r25_list.append(err_r0_r25)
        err_r0_r25_lists.append(err_r0_r25)

    column_names = [' ','M31','M33','M51','NGC 6946']

    B = ['B_tot', 'B_reg', 'B_ord']
    r_0_lists_n        = []
    slope_lists_n      = []
    err_slope_lists_n  = []
    err_r_0_lists_n    = []
    r0_r25_lists_n     = []
    err_r0_r25_lists_n = []

    for i in range(len(r0_lists)):
        x = list(r0_lists[i])
        x.insert(0,B[i])
        r_0_lists_n.append(x)

        w = list(err_r0_lists[i])
        w.insert(0,B[i])
        err_r_0_lists_n.append(w)

        v = list(r0_r25_lists[i])
        v.insert(0,B[i])
        r0_r25_lists_n.append(v)

        u = list(err_r0_r25_lists[i])
        u.insert(0,B[i])
        err_r0_r25_lists_n.append(u)

        y = list(slope_lists[i])
        y.insert(0,B[i])
        slope_lists_n.append(y)

        z = list(err_slope_lists[i])
        z.insert(0,B[i])
        err_slope_lists_n.append(z)
        
    with open(plot_save_dir+r'\scale_length_B'+r'\r0.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(column_names)
        csvwriter.writerows(r_0_lists_n)

    with open(plot_save_dir+r'\scale_length_B'+r'\r0_r25_ratio.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(column_names)
        csvwriter.writerows(r0_r25_lists_n)

    with open(plot_save_dir+r'\scale_length_B'+r'\error_r0.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(column_names)
        csvwriter.writerows(err_r_0_lists_n)

    with open(plot_save_dir+r'\scale_length_B'+r'\error_r0_r25_ratio.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(column_names)
        csvwriter.writerows(err_r0_r25_lists_n)

    with open(plot_save_dir+r'\scale_length_B'+r'\slopes.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(column_names)
        csvwriter.writerows(slope_lists_n)  

    with open(plot_save_dir+r'\scale_length_B'+r'\error_slopes.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(column_names)
        csvwriter.writerows(err_slope_lists_n)

if func_number == 5:
    # plot outputs vs r and r/r25 
    for i in range(1,len(quantities)): 
        plt.figure(figsize=(5, 5))

        # change margin size
        plt.rcParams['axes.xmargin'] = 0
        plt.rcParams['axes.ymargin'] = 0

        #plotting non-normalised radial variations
        for galaxy in galaxies:
            
            if i == 1: # add legend only for first panel
                if galaxy == 'ngc6946':
                    plt.plot(galaxy_data[galaxy]['radius'], galaxy_data[galaxy][quantities[i]], color=color_map[galaxy],marker='.', linewidth=lw, label='NGC 6946')
                elif galaxy == 'm31': # plot M31Alt
                    plt.plot(galaxy_data[galaxy]['radius'], galaxy_data[galaxy][quantities[i]], color=color_map[galaxy], marker='.', linewidth=lw, label=galaxy.upper())
                    plt.plot(galaxy_data['M31Alt']['radius'], galaxy_data['M31Alt'][quantities[i]], color='cyan', marker='.', linewidth=lw, label='M31Alt')

                else:
                    plt.plot(galaxy_data[galaxy]['radius'], galaxy_data[galaxy][quantities[i]], color=color_map[galaxy],marker='.', linewidth=lw, label=galaxy.upper())
                
                plt.legend(fontsize=lfs, frameon=False, handlelength=hd, ncol=1, loc = 'upper left', prop={
                'size': leg_textsize,  'family': 'Times New Roman'}, fancybox=True, framealpha=0.9, handletextpad=legend_labelspace, columnspacing=0.7)
            
            else:
                plt.plot(galaxy_data[galaxy]['radius'], galaxy_data[galaxy][quantities[i]], color=color_map[galaxy],marker='.', linewidth=lw)
                if galaxy == 'm31': # plot M31Alt
                    plt.plot(galaxy_data['M31Alt']['radius'], galaxy_data['M31Alt'][quantities[i]], color='cyan', marker='.', linewidth=lw)
                
            fill_between_err(galaxy_data[galaxy]['radius'], galaxy_data[galaxy][quantities[i]], galaxy_data_err[galaxy][quantities_err[i]], color_map[galaxy], 0.3)
            if galaxy == 'm31':
                fill_between_err(galaxy_data['M31Alt']['radius'], galaxy_data['M31Alt'][quantities[i]], galaxy_data_err['M31Alt'][quantities_err[i]], 'cyan', 0.3)

            # title   = quants_for_title[i] + ' vs '+ quants_for_title[0] # y vs x and x= r here
            title = symbols_for_axis[i] + ' ' + units_for_axis[i]

            # plt.ylabel(y_label, fontsize=fs, labelpad=1)
            plt.xticks(np.arange(0, 21, 4)) #for ngc

            # log axis for some plots
            if i in [6,7,8,11,12,13,14,16]:
                plt.yscale('log')

            plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
            plt.rcParams['axes.titlepad'] = -25  # pad is in points...
            # Switch on major and minor ticks on y axis
            plt.tick_params(width=tick_width)

            plt.minorticks_on()
            # plt.tick_params(axis='y', which='major', labelsize=fs)  # Customize major ticks
            # plt.tick_params(axis='y', which='minor', labelleft=False)  # Turn off labels for minor ticks
            plt.tick_params(axis='y', which='both', right=True, labelsize=fs, labelright=False, width = 2)
            # Enable ticks on the top of the plot without labels
            plt.tick_params(axis='x', which='both', top=True, labeltop=False, width = 3)
            
            # Adjust the top margin
            plt.subplots_adjust(top    = 0.95)  # Adjust this value as needed to reduce the upper margin
            plt.subplots_adjust(bottom = 0.15)  # Adjust this value as needed to reduce the upper margin

            if i in [9,10,14,16,17]:
                plt.xlabel('Radius (kpc)', fontsize = fs+2, labelpad = 1) # labelpad change gap between axis and axis title

            # e-folding time
            if i == 17:
                plt.ylim(top=10)

            # gamma
            if i == 16:
                # set y limit at 40
                plt.ylim(top=40, bottom = 0.1)
                ticks = [1, 5, 10, 20, 30, 45]
                plt.yticks(ticks)
                plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y,_:f'{int(y)}'))
            
            # Mach no
            if i == 15:
                plt.ylim(bottom = 0, top=4)

            # dkdc
            if i == 14:
                plt.ylim(top = 100, bottom=1)

            # Field strengths
            if i in [6,7,8]:
                # Set custom ticks
                plt.ylim(top=60)
                plt.ylim(bottom=0.25)

                ticks = [1, 2, 3, 5, 10, 20, 30, 40, 60]
                plt.yticks(ticks)
                plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y,_:f'{int(y)}'))
            
            # speeds
            if i in [3,4,5]:
                plt.ylim(bottom=0,top=28)

            # pitch angles
            if i in [9]:
                plt.ylim(bottom=0,top=40)
            if i in [10]:
                plt.ylim(bottom=0,top=20)

            # tau_e and tau_r
            if i in [12,13]:
                plt.ylim(bottom=0.05, top=400)
            
            if i in [1, 3, 4, 5, 7, 8, 10, 11, 14, 15, 16]:
                plt.text(x=0.7, y=1.1, s=title, fontsize=title_textsize, fontweight='normal',ha='right', va='top', transform=ax.transAxes, 
                        bbox=dict(facecolor='none', edgecolor='black', boxstyle='square,pad=0.1'))
            elif i in [6, 9, 13]:
                plt.text(x=0.7, y=0.18, s=title, fontsize=title_textsize, fontweight='normal',ha='right', va='top', transform=ax.transAxes, 
                        bbox=dict(facecolor='none', edgecolor='black', boxstyle='square,pad=0.1'))
            else:
                plt.text(x=0, y=1.1, s=title, fontsize=title_textsize, fontweight='normal', ha='left', va='top', transform=ax.transAxes, 
                        bbox=dict(facecolor='none', edgecolor='black', boxstyle='square,pad=0.1'))

            os.chdir(os.path.join(plot_save_dir+'\plots'))
            plt.savefig('{}'.format(quantities[i]))
            # os.chdir('..')

            # os.chdir(base_path)
        plt.close()

        # after normalising with r_25
        plt.figure(figsize=(5, 5))

        # change margin size
        plt.rcParams['axes.xmargin'] = 0
        plt.rcParams['axes.ymargin'] = 0


        for galaxy in galaxies:
            r_r25 = galaxy_data[galaxy]['radius']/r_25_new_kpc[galaxies.index(galaxy)]

            if i == 1: # add legend for only first panel
                if galaxy == 'ngc6946':
                    plt.plot(r_r25, galaxy_data[galaxy][quantities[i]],color_map[galaxy], marker='o', linewidth=lw, label='NGC 6946')
                elif galaxy == 'm31': # plot M31Alt
                    plt.plot(r_r25, galaxy_data[galaxy][quantities[i]], color=color_map[galaxy], marker='.', linewidth=lw, label=galaxy.upper())
                    plt.plot(r_r25, galaxy_data['M31Alt'][quantities[i]], color='cyan', marker='.', linewidth=lw, label='M31Alt')
                else:
                    plt.plot(r_r25, galaxy_data[galaxy][quantities[i]],color_map[galaxy], marker='o', linewidth=lw, label=galaxy.upper())
                
                plt.legend(fontsize=lfs, frameon=False, handlelength=hd, ncol=1, loc = 'upper left', prop={
                'size': leg_textsize,  'family': 'Times New Roman'}, fancybox=True, framealpha=0.9, handletextpad=legend_labelspace, columnspacing=0.7)
            
            else:
                plt.plot(r_r25, galaxy_data[galaxy][quantities[i]],color_map[galaxy], marker='o', linewidth=lw)
                if galaxy == 'm31': # plot M31Alt
                    plt.plot(r_r25, galaxy_data['M31Alt'][quantities[i]], color='cyan', marker='.', linewidth=lw)
                

            fill_between_err(r_r25, galaxy_data[galaxy][quantities[i]], galaxy_data_err[galaxy][quantities_err[i]], color_map[galaxy], 0.3)
            if galaxy == 'm31':
                fill_between_err(r_r25, galaxy_data['M31Alt'][quantities[i]], galaxy_data_err['M31Alt'][quantities_err[i]], 'cyan', 0.3)

            # title = quants_for_title[i] + ' vs '+ quants_for_title[0] # y vs x and x= r/r_25 here
            title = symbols_for_axis[i] + ' ' + units_for_axis[i]

            # plt.ylabel(y_label, fontsize=fs)

            plt.xticks(np.arange(0, 2, 0.5)) 

            # log axis for some plots
            if i in [6,7,8,11,12,13,14,16]:
                plt.yscale('log')

            plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
            plt.rcParams['axes.titlepad'] = -25  # pad is in points...
            plt.tick_params(width=tick_width)
            # Switch on major and minor ticks on y axis
            
            plt.minorticks_on()
            plt.tick_params(axis='y', which='both', right=True, labelsize=fs, labelright=False, width = 2)
            # Enable ticks on the top and right of the plot without labels
            plt.tick_params(axis='x', which='both', top=True, labeltop=False, width = 4)

            # Adjust the top margin
            plt.subplots_adjust(top    = 0.95)  # Adjust this value as needed to reduce the upper margin
            plt.subplots_adjust(bottom = 0.15)  # Adjust this value as needed to reduce the upper margin

            if i in [9,10,14,16,17]:
                plt.xlabel(r'$r/r_{25}$', fontsize=fs+2, labelpad=1)

            plt.tick_params(axis='y', which='both', right=True, labelright=False)

            # e-folding time
            if i == 17:
                plt.ylim(top=7)
                
            # gamma
            if i == 16:
                # set y limit at 40
                plt.ylim(top=40 , bottom = 0.1)

            # Mach no
            if i == 15:
                plt.ylim(bottom = 0, top=4)

            # dkdc
            if i == 14:
                plt.ylim(top = 65, bottom=1)

            # Field strengths
            if i in [6,7,8]:
                # Set custom ticks
                plt.ylim(top=45)
                plt.ylim(bottom=0.25)

                ticks = [1, 2, 3, 4, 5, 10, 20, 30]
                plt.yticks(ticks)
                plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y,_:f'{int(y)}'))
            
            # speeds
            if i in [3,4,5]:
                plt.ylim(bottom=0,top=28)

            # pitch angles
            if i in [9]:
                plt.ylim(bottom=0,top=40)
            if i in [10]:
                plt.ylim(bottom=0,top=20)

            # tau_e and tau_r
            if i in [12,13]:
                plt.ylim(bottom=0.1, top=400)

            if i in [1, 3, 4, 5, 7, 8, 10, 11, 14, 15, 16]:
                plt.text(x=0.7, y=1.1, s=title, fontsize=title_textsize, fontweight='normal',ha='right', va='top', transform=ax.transAxes, 
                        bbox=dict(facecolor='none', edgecolor='black', boxstyle='square,pad=0.1'))
            elif i in [6, 9, 13]:
                plt.text(x=0.7, y=0.18, s=title, fontsize=title_textsize, fontweight='normal',ha='right', va='top', transform=ax.transAxes, 
                        bbox=dict(facecolor='none', edgecolor='black', boxstyle='square,pad=0.1'))
            else:
                plt.text(x=0, y=1.1, s=title, fontsize=title_textsize, fontweight='normal', ha='left', va='top', transform=ax.transAxes, 
                        bbox=dict(facecolor='none', edgecolor='black', boxstyle='square,pad=0.1'))

            os.chdir(plot_save_dir+'\plots_r25')
            plt.savefig('{}'.format(quantities[i]))

            # os.chdir('..')
        plt.close()

if func_number == 6:
    # outputs plotted against each other
    for j in range(1,len(quantities)-1):
        for i in range(1,len(quantities)-1):
            plt.figure(figsize=(11, 6))
            if i != j:
                for galaxy in galaxies:
                    plt.plot(galaxy_data[galaxy][quantities[j]], galaxy_data[galaxy][quantities[i]],color_map[galaxy], marker='o', linewidth=lw, label=galaxy.upper())
                    # fill_between_err(galaxy_obs[galaxy].iloc[:,j], galaxy_data[galaxy][quantities[i]], galaxy_data_err[galaxy][quantities_err[i-1]], color_map[galaxy], 0.3)
                    title   = quants_for_title[i] + ' vs '+ quants_for_title[j] # y vs x
                    x_label = symbols_for_axis[j] + ' ' + units_for_axis[j]
                    y_label = symbols_for_axis[i] + ' ' + units_for_axis[i]

                    plt.xlabel(x_label, fontsize=fs)
                    plt.ylabel(y_label, fontsize=fs)

                    plt.tick_params(width=tick_width)
                    plt.minorticks_on()
                    
                    plt.title(title, fontsize=title_textsize, weight='bold')
                    plt.legend(fontsize=lfs, frameon=False, handlelength=hd, ncol=1, prop={
                        'size': leg_textsize,  'family': 'Times New Roman'}, fancybox=True, framealpha=0.9, handletextpad=legend_labelspace, columnspacing=0.7)
                    os.chdir(plot_save_dir+'\output_vs_output')

                    # make a folder for each output quantity
                    os.makedirs('{}'.format(quantities[j]), exist_ok=True)
                    os.chdir('{}'.format(quantities[j]))
                    plt.savefig('{}_vs_{}'.format(quantities[i],quantities[j]))
                    
                    os.chdir('..')
                    os.chdir('..')
                plt.close()

if func_number == 7:
    # get all variables from mag_data_gal_combined.py
    from mag_data_gal_combined import *

    # plotting the pB vs Breg and saving in miscellaneous folder
    plt.figure(figsize=(11, 6))
    plt.plot(m31_df['Breg'], m31_df['pB'], color_map['m31'], linewidth=lw, label='M31')
    plt.plot(m33_df['Breg'], m33_df['pB'], color_map['m33'], linewidth=lw, label='M33')
    plt.plot(m51_df['Breg'], m51_df['pB'], color_map['m51'], linewidth=lw, label='M51')
    # plt.plot(ngc6946_df['Breg'], ngc6946_df['pB'], color_map['ngc6946'], linewidth=lw, label='NGC6946')

    # plotting pB vs Breg data from Beck+19
    plt.plot(G_dat_Breg_m31, M_pb_beck19_m31/(np.pi/180), color_map['m31'], marker='*', markersize = m, linestyle=' ')#, label='M31 (Beck+19)')
    plt.plot(G_dat_Breg_m33, pb_beck19_m33/(np.pi/180), color_map['m33'], marker='D', markersize = m, linestyle=' ')#, label='M33 (Beck+19)')
    plt.plot(G_dat_Breg_m51, dat_pb_m51/(np.pi/180), color_map['m51'], marker='p', markersize = m, linestyle=' ')#, label='M51 (Beck+19)')
    # plt.plot(G_dat_Btot_ngc6946, dat_pb_ngc6946, color_map['ngc6946'], marker='s', markersize = m, linestyle=' ')#, label='NGC6946 (Beck+19)')

    plt.xlabel(r'$B_{\mathrm{reg}}$ ($\mu$G)', fontsize=fs)
    plt.ylabel(r'$p_{\mathrm{B}}$ (deg)', fontsize=fs)

    # x limit 0 to 15
    plt.xlim(0, 15)
    plt.ylim(0, 55)

    plt.tick_params(width=tick_width)
    plt.minorticks_on()
    plt.title(r'$p_{\mathrm{B}}$ vs $B_{\mathrm{reg}}$', fontsize=title_textsize, weight='bold')
    plt.legend(fontsize=lfs, frameon=False, handlelength=hd, ncol=1, prop={
        'size': leg_textsize,  'family': 'Times New Roman'}, fancybox=True, framealpha=0.9, handletextpad=legend_labelspace, columnspacing=0.7)
    os.chdir(plot_save_dir+'\miscellaneous')
    plt.savefig('pB_vs_Breg')
    plt.close()
    os.chdir('..')

    # plotting the pB vs Btot and saving in miscellaneous folder
    plt.figure(figsize=(11, 6))
    plt.plot(m31_df['Btot'], m31_df['pB'], color_map['m31'], linewidth=lw, label='M31')
    plt.plot(m33_df['Btot'], m33_df['pB'], color_map['m33'], linewidth=lw, label='M33')
    plt.plot(m51_df['Btot'], m51_df['pB'], color_map['m51'], linewidth=lw, label='M51')
    plt.plot(ngc6946_df['Btot'], ngc6946_df['pB'], color_map['ngc6946'], linewidth=lw, label='NGC6946')

    # plotting pB vs Btot data from Beck+19
    plt.plot(G_dat_Btot_m31, M_pb_beck19_m31/(np.pi/180), color_map['m31'], marker='*', markersize = m, linestyle=' ')#, label='M31 (Beck+19)')
    plt.plot(G_dat_Btot_m33, pb_beck19_m33/(np.pi/180), color_map['m33'], marker='D', markersize = m, linestyle=' ')#, label='M33 (Beck+19)')
    plt.plot(G_dat_Btot_m51, dat_pb_m51/(np.pi/180), color_map['m51'], marker='p', markersize = m, linestyle=' ')#, label='M51 (Beck+19)')
    # plt.plot(G_dat_Btot_ngc6946, dat_pb_ngc6946, color_map['ngc6946'], marker='s', markersize = m, linestyle=' ')#, label='NGC6946 (Beck+19)')

    plt.xlabel(r'$B_{\mathrm{tot}}$ ($\mu$G)', fontsize=fs)
    plt.ylabel(r'$p_{\mathrm{B}}$ (deg)', fontsize=fs)

    # x limit 0 to 15
    plt.xlim(2, 40)
    plt.ylim(0, 55)

    plt.tick_params(width=tick_width)
    plt.minorticks_on()
    plt.title(r'$p_{\mathrm{B}}$ vs $B_{\mathrm{tot}}$', fontsize=title_textsize, weight='bold')
    plt.legend(fontsize=lfs, frameon=False, handlelength=hd, ncol=1, prop={
        'size': leg_textsize,  'family': 'Times New Roman'}, fancybox=True, framealpha=0.9, handletextpad=legend_labelspace, columnspacing=0.7)
    os.chdir(plot_save_dir+'\miscellaneous')
    plt.savefig('pB_vs_Btot')
    plt.close()
    os.chdir('..')

    # for all galaxies, get ratio of breg to btot using imported data
    breg_btot_m31_beck = G_dat_Breg_m31/G_dat_Btot_m31
    breg_btot_m33_beck = G_dat_Breg_m33/G_dat_Btot_m33
    breg_btot_m51_beck = G_dat_Breg_m51/G_dat_Btot_m51
    breg_btot_ngc6946_beck = G_dat_Breg_ngc6946/G_dat_Btot_ngc6946

    # for all galaxies, get ratio of bord to btot using imported data
    bord_btot_m31_beck = G_dat_Bord_m31/G_dat_Btot_m31
    bord_btot_m33_beck = G_dat_Bord_m33/G_dat_Btot_m33
    bord_btot_m51_beck = G_dat_Bord_m51/G_dat_Btot_m51
    bord_btot_ngc6946_beck = G_dat_Bord_ngc6946/G_dat_Btot_ngc6946

    # for all galaxies, get ratio of bord to breg using imported data
    bord_breg_m31_beck = G_dat_Bord_m31/G_dat_Breg_m31
    bord_breg_m33_beck = G_dat_Bord_m33/G_dat_Breg_m33
    bord_breg_m51_beck = G_dat_Bord_m51/G_dat_Breg_m51
    bord_breg_ngc6946_beck = G_dat_Bord_ngc6946/G_dat_Breg_ngc6946

    # for all galaxies, get ratio of breg to btot
    breg_btot_m31 = m31_df['Breg']/m31_df['Btot']
    breg_btot_m33 = m33_df['Breg']/m33_df['Btot']
    breg_btot_m51 = m51_df['Breg']/m51_df['Btot']
    breg_btot_ngc6946 = ngc6946_df['Breg']/ngc6946_df['Btot']

    # plot these ratios in a single plot and save in the miscellaneous folder
    plt.figure(figsize=(11, 6))
    plt.plot(m31_df['radius'], breg_btot_m31, color_map['m31'], marker='o', linewidth=lw, label='M31')
    plt.plot(m33_df['radius'], breg_btot_m33, color_map['m33'], marker='o', linewidth=lw, label='M33')
    plt.plot(m51_df['radius'], breg_btot_m51, color_map['m51'], marker='o', linewidth=lw, label='M51')
    plt.plot(ngc6946_df['radius'], breg_btot_ngc6946, color_map['ngc6946'], marker='o', linewidth=lw, label='NGC6946')

    # plot ratios against radius
    plt.plot(mrange_m31, breg_btot_m31_beck, color_map['m31'], marker='*', markersize = m, linestyle=' ')#, label='M31 (Beck+19)')
    plt.plot(mrange_m33, breg_btot_m33_beck, color_map['m33'], marker='D', markersize = m, linestyle=' ')#, label='M33 (Beck+19)')
    plt.plot(mrange_m51, breg_btot_m51_beck, color_map['m51'], marker='p', markersize = m, linestyle=' ')#, label='M51 (Beck+19)')
    plt.plot(mrange_ngc6946, breg_btot_ngc6946_beck, color_map['ngc6946'], marker='s', markersize = m, linestyle=' ')#, label='NGC6946 (Beck+19)')

    plt.xlabel('Radius (kpc)', fontsize=fs)
    plt.ylabel(r'$B_{\mathrm{reg}}/B_{\mathrm{tot}}$', fontsize=fs)
    plt.tick_params(width=tick_width)
    plt.minorticks_on()
    plt.title(r'$B_{\mathrm{reg}}/B_{\mathrm{tot}}$ vs Radius', fontsize=title_textsize, weight='bold')
    plt.legend(fontsize=lfs, frameon=False, handlelength=hd, ncol=1, prop={
        'size': leg_textsize,  'family': 'Times New Roman'}, fancybox=True, framealpha=0.9, handletextpad=legend_labelspace, columnspacing=0.7)
    os.chdir('miscellaneous')
    plt.savefig('breg_btot')
    plt.close()
    os.chdir('..')

    # plot against r/25
    plt.figure(figsize=(11, 6))
    plt.plot(m31_df['radius']/r_25_new_kpc[0], breg_btot_m31, color_map['m31'], marker='o', linewidth=lw, label='M31')
    plt.plot(m33_df['radius']/r_25_new_kpc[1], breg_btot_m33, color_map['m33'], marker='o', linewidth=lw, label='M33')
    plt.plot(m51_df['radius']/r_25_new_kpc[2], breg_btot_m51, color_map['m51'], marker='o', linewidth=lw, label='M51')
    plt.plot(ngc6946_df['radius']/r_25_new_kpc[3], breg_btot_ngc6946, color_map['ngc6946'], marker='o', linewidth=lw, label='NGC6946')

    plt.xlabel(r'$r/r_{25}$', fontsize=fs)
    plt.ylabel(r'$B_{\mathrm{reg}}/B_{\mathrm{tot}}$', fontsize=fs)
    plt.tick_params(width=tick_width)
    plt.minorticks_on()
    plt.title(r'$B_{\mathrm{reg}}/B_{\mathrm{tot}}$ vs $r/r_{25}$', fontsize=title_textsize, weight='bold')
    plt.legend(fontsize=lfs, frameon=False, handlelength=hd, ncol=1, prop={
        'size': leg_textsize,  'family': 'Times New Roman'}, fancybox=True, framealpha=0.9, handletextpad=legend_labelspace, columnspacing=0.7)
    os.chdir('miscellaneous')
    plt.savefig('breg_btot_r25')
    plt.close()
    os.chdir('..')

    # for all galaxies, get ratio of bord to btot
    bord_btot_m31 = m31_df['Bord']/m31_df['Btot']
    bord_btot_m33 = m33_df['Bord']/m33_df['Btot']
    bord_btot_m51 = m51_df['Bord']/m51_df['Btot']
    bord_btot_ngc6946 = ngc6946_df['Bord']/ngc6946_df['Btot']

    # plot these ratios in a single plot and save in the miscellaneous folder
    plt.figure(figsize=(11, 6))
    plt.plot(m31_df['radius'], bord_btot_m31, color_map['m31'], marker='o', linewidth=lw, label='M31')
    plt.plot(m33_df['radius'], bord_btot_m33, color_map['m33'], marker='o', linewidth=lw, label='M33')
    plt.plot(m51_df['radius'], bord_btot_m51, color_map['m51'], marker='o', linewidth=lw, label='M51')
    plt.plot(ngc6946_df['radius'], bord_btot_ngc6946, color_map['ngc6946'], marker='o', linewidth=lw, label='NGC6946')

    # plot ratio data against radius
    plt.plot(mrange_m31, bord_btot_m31_beck, color_map['m31'], marker='*', markersize = m,linestyle=' ')#, label='M31 (Beck+19)')
    plt.plot(mrange_m33, bord_btot_m33_beck, color_map['m33'], marker='D', markersize = m, linestyle=' ')#, label='M33 (Beck+19)')
    plt.plot(mrange_m51, bord_btot_m51_beck, color_map['m51'], marker='P', markersize = m, linestyle=' ')#, label='M51 (Beck+19)')
    plt.plot(mrange_ngc6946, bord_btot_ngc6946_beck, color_map['ngc6946'], marker='s', markersize = m, linestyle=' ')#, label='NGC6946 (Beck+19)')

    plt.xlabel('Radius (kpc)', fontsize=fs)
    plt.ylabel(r'$B_{\mathrm{ord}}/B_{\mathrm{tot}}$', fontsize=fs)
    plt.tick_params(width=tick_width)
    plt.minorticks_on()
    plt.title(r'$B_{\mathrm{ord}}/B_{\mathrm{tot}}$ vs Radius', fontsize=title_textsize, weight='bold')
    plt.legend(fontsize=lfs, frameon=False, handlelength=hd, ncol=1, prop={
        'size': leg_textsize,  'family': 'Times New Roman'}, fancybox=True, framealpha=0.9, handletextpad=legend_labelspace, columnspacing=0.7)
    os.chdir('miscellaneous')
    plt.savefig('bord_btot')
    plt.close()
    os.chdir('..')

    # plot against r/25
    plt.figure(figsize=(11, 6))
    plt.plot(m31_df['radius']/r_25_new_kpc[0], bord_btot_m31, color_map['m31'], marker='o', linewidth=lw, label='M31')
    plt.plot(m33_df['radius']/r_25_new_kpc[1], bord_btot_m33, color_map['m33'], marker='o', linewidth=lw, label='M33')
    plt.plot(m51_df['radius']/r_25_new_kpc[2], bord_btot_m51, color_map['m51'], marker='o', linewidth=lw, label='M51')
    plt.plot(ngc6946_df['radius']/r_25_new_kpc[3], bord_btot_ngc6946, color_map['ngc6946'], marker='o', linewidth=lw, label='NGC6946')

    plt.xlabel(r'$r/r_{25}$', fontsize=fs)
    plt.ylabel(r'$B_{\mathrm{ord}}/B_{\mathrm{tot}}$', fontsize=fs)
    plt.tick_params(width=tick_width)
    plt.minorticks_on()
    plt.title(r'$B_{\mathrm{ord}}/B_{\mathrm{tot}}$ vs $r/r_{25}$', fontsize=title_textsize, weight='bold')
    plt.legend(fontsize=lfs, frameon=False, handlelength=hd, ncol=1, prop={
        'size': leg_textsize,  'family': 'Times New Roman'}, fancybox=True, framealpha=0.9, handletextpad=legend_labelspace, columnspacing=0.7)
    os.chdir(plot_save_dir+'\miscellaneous')
    plt.savefig('bord_btot_r25')
    plt.close()
    os.chdir('..')

    # for all galaxies, get ratio of bord to breg
    bord_breg_m31 = m31_df['Bord']/m31_df['Breg']
    bord_breg_m33 = m33_df['Bord']/m33_df['Breg']
    bord_breg_m51 = m51_df['Bord']/m51_df['Breg']
    bord_breg_ngc6946 = ngc6946_df['Bord']/ngc6946_df['Breg']

    # plot these ratios in a single plot and save in the miscellaneous folder
    plt.figure(figsize=(11, 6))

    plt.plot(m31_df['radius'], bord_breg_m31, color_map['m31'], marker='o', linewidth=lw, label='M31')
    plt.plot(m33_df['radius'], bord_breg_m33, color_map['m33'], marker='o', linewidth=lw, label='M33')
    plt.plot(m51_df['radius'], bord_breg_m51, color_map['m51'], marker='o', linewidth=lw, label='M51')
    plt.plot(ngc6946_df['radius'], bord_breg_ngc6946, color_map['ngc6946'], marker='o', linewidth=lw, label='NGC6946')

    # plot ratio data against radius
    plt.plot(mrange_m31, bord_breg_m31_beck, color_map['m31'], marker='*', markersize = m, linestyle=' ')#, label='M31 (Beck+19)')
    plt.plot(mrange_m33, bord_breg_m33_beck, color_map['m33'], marker='D', markersize = m, linestyle=' ')#, label='M33 (Beck+19)')
    plt.plot(mrange_m51, bord_breg_m51_beck, color_map['m51'], marker='P', markersize = m, linestyle=' ')#, label='M51 (Beck+19)')
    plt.plot(mrange_ngc6946, bord_breg_ngc6946_beck, color_map['ngc6946'], marker='s', markersize = m, linestyle=' ')#, label='NGC6946 (Beck+19)')
    plt.xlabel('Radius (kpc)', fontsize=fs)
    plt.ylabel(r'$B_{\mathrm{ord}}/B_{\mathrm{reg}}$', fontsize=fs)
    plt.tick_params(width=tick_width)
    plt.minorticks_on()
    plt.title(r'$B_{\mathrm{ord}}/B_{\mathrm{reg}}$ vs Radius', fontsize=title_textsize, weight='bold')
    plt.legend(fontsize=lfs, frameon=False, handlelength=hd, ncol=1, prop={
        'size': leg_textsize,  'family': 'Times New Roman'}, fancybox=True, framealpha=0.9, handletextpad=legend_labelspace, columnspacing=0.7)
    os.chdir(plot_save_dir+'\miscellaneous')
    plt.savefig('bord_breg')
    plt.close()
    os.chdir('..')

    # plot against r/25
    plt.figure(figsize=(11, 6))
    plt.plot(m31_df['radius']/r_25_new_kpc[0], bord_breg_m31, color_map['m31'], marker='o', linewidth=lw, label='M31')
    plt.plot(m33_df['radius']/r_25_new_kpc[1], bord_breg_m33, color_map['m33'], marker='o', linewidth=lw, label='M33')
    plt.plot(m51_df['radius']/r_25_new_kpc[2], bord_breg_m51, color_map['m51'], marker='o', linewidth=lw, label='M51')
    plt.plot(ngc6946_df['radius']/r_25_new_kpc[3], bord_breg_ngc6946, color_map['ngc6946'], marker='o', linewidth=lw, label='NGC6946')

    plt.xlabel(r'$r/r_{25}$', fontsize=fs)
    plt.ylabel(r'$B_{\mathrm{ord}}/B_{\mathrm{reg}}$', fontsize=fs)
    plt.tick_params(width=tick_width)
    plt.minorticks_on()
    plt.title(r'$B_{\mathrm{ord}}/B_{\mathrm{reg}}$ vs $r/r_{25}$', fontsize=title_textsize, weight='bold')
    plt.legend(fontsize=lfs, frameon=False, handlelength=hd, ncol=1, prop={
        'size': leg_textsize,  'family': 'Times New Roman'}, fancybox=True, framealpha=0.9, handletextpad=legend_labelspace, columnspacing=0.7)
    os.chdir(plot_save_dir+'\miscellaneous')
    plt.savefig('bord_breg_r25')
    plt.close()
    os.chdir('..')

# make a plot of tau_e/tau_r vs radius for all galaxies
if func_number == 8:
    ratio_taue_tau_r_m31     = m31_df['taue_e']/m31_df['tau_r']
    ratio_taue_tau_r_m33     = m33_df['taue_e']/m33_df['tau_r']
    ratio_taue_tau_r_m51     = m51_df['taue_e']/m51_df['tau_r']
    ratio_taue_tau_r_ngc6946 = ngc6946_df['taue_e']/ngc6946_df['tau_r']

    # Include the M31Alt data
    ratio_taue_tau_r_m31_alt = m31Alt_df['taue_e']/m31Alt_df['tau_r']

    # plot these ratios in a single plot and save in the miscellaneous folder
    plt.figure(figsize=(9, 6))
    # x-axis limit 0 to 20
    plt.xlim(0, 20)
    # y-axis limit 0 to 14
    plt.ylim(0, 14)

    plt.plot(m31_df['radius']    , ratio_taue_tau_r_m31, color_map['m31'], marker='o', linewidth=lw, label='M31')
    plt.plot(m33_df['radius']    , ratio_taue_tau_r_m33, color_map['m33'], marker='o', linewidth=lw, label='M33')
    plt.plot(m51_df['radius']    , ratio_taue_tau_r_m51, color_map['m51'], marker='o', linewidth=lw, label='M51')
    plt.plot(ngc6946_df['radius'], ratio_taue_tau_r_ngc6946, color_map['ngc6946'], marker='o', linewidth=lw, label='NGC6946')
    # Include the M31Alt data
    plt.plot(m31Alt_df['radius'] , ratio_taue_tau_r_m31_alt, color='cyan', marker='o', linewidth=lw, label='M31Alt')

    plt.xlabel('Radius (kpc)', fontsize=fs)
    plt.ylabel(r'$\tau_{\mathrm{e}}/\tau_{\mathrm{r}}$', fontsize=fs)
    plt.tick_params(width=tick_width)
    plt.minorticks_on()
    plt.title(r'$\tau_{\mathrm{e}}/\tau_{\mathrm{r}}$ vs Radius', fontsize=title_textsize, weight='bold')
    plt.legend(fontsize=lfs, frameon=False, handlelength=hd, ncol=1, prop={
        'size': leg_textsize,  'family': 'Times New Roman'}, fancybox=True, framealpha=0.9, handletextpad=legend_labelspace, columnspacing=0.7)
    os.chdir(plot_save_dir+'\miscellaneous')
    plt.savefig('tau_e_tau_r')
    plt.close()
    os.chdir('..')

    # plot against r/25
    plt.figure(figsize=(9, 6))
    # x-axis limit 0 to 1.5
    plt.xlim(0, 1.5)
    # y-axis limit 0 to 14
    plt.ylim(0, 14)

    plt.plot(m31_df['radius']/r_25_new_kpc[0]    , ratio_taue_tau_r_m31, color_map['m31'], marker='o', linewidth=lw, label='M31')
    plt.plot(m33_df['radius']/r_25_new_kpc[1]    , ratio_taue_tau_r_m33, color_map['m33'], marker='o', linewidth=lw, label='M33')
    plt.plot(m51_df['radius']/r_25_new_kpc[2]    , ratio_taue_tau_r_m51, color_map['m51'], marker='o', linewidth=lw, label='M51')
    plt.plot(ngc6946_df['radius']/r_25_new_kpc[3], ratio_taue_tau_r_ngc6946, color_map['ngc6946'], marker='o', linewidth=lw, label='NGC6946')
    # Include the M31Alt data
    plt.plot(m31Alt_df['radius']/r_25_new_kpc[0] , ratio_taue_tau_r_m31_alt, color='cyan', marker='o', linewidth=lw, label='M31Alt')

    plt.xlabel(r'$r/r_{25}$', fontsize=fs)
    plt.ylabel(r'$\tau_{\mathrm{e}}/\tau_{\mathrm{r}}$', fontsize=fs)
    plt.tick_params(width=tick_width)
    plt.minorticks_on()
    plt.title(r'$\tau_{\mathrm{e}}/\tau_{\mathrm{r}}$ vs $r/r_{25}$', fontsize=title_textsize, weight='bold')
    plt.legend(fontsize=lfs, frameon=False, handlelength=hd, ncol=1, prop={
        'size': leg_textsize,  'family': 'Times New Roman'}, fancybox=True, framealpha=0.9, handletextpad=legend_labelspace, columnspacing=0.7)
    os.chdir(plot_save_dir+'\miscellaneous')
    plt.savefig('tau_e_tau_r_r25')
    plt.close()
    os.chdir('..')

# if func_number == 9:
#     # plot omega times t
print('######  OUTPUT COMPARISON FILE ENDS #####')
