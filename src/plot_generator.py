# This file is used to plot the data from the output files

print('#####  Plotting starts #####')

import matplotlib
from helper_functions import datamaker, parameter_read, analytical_pitch_angle_integrator, plot_rectangle, fill_error, new_pitch_angle_integrator
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import pickle
import os
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import griddata
import sys
from datetime import date
import csv 
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)
import matplotlib.patches as patches
import pandas as pd
from icecream import ic

today = date.today() # to set the date for the folder name

# Defining the Observables
q        = Symbol('q')
omega    = Symbol('\Omega')
sigma    = Symbol('\Sigma')
sigmatot = Symbol('Sigma_tot')
sigmasfr = Symbol('Sigma_SFR')
T        = Symbol('T')

# Defining the Constants
gamma = Symbol('gamma')
boltz = Symbol('k_B')
mu    = Symbol('mu')
mh    = Symbol('m_H')

# Defining the general parameters
u   = Symbol('u')
tau = Symbol('tau')
l   = Symbol('l')
h   = Symbol('h')

# conversion factors
pc_kpc     = 1e3  # number of pc in one kpc
cm_km      = 1e5  # number of cm in one km
cm_kpc     = 3.086e+21  # number of centimeters in one parsec
s_Myr      = 1e+6*(365*24*60*60)  # megayears to seconds
deg_rad    = 180e0/np.pi
arcmin_deg = 60e0
arcsec_deg = 3600e0

# reading the parameters
base_path   = os.environ.get('MY_PATH')
galaxy_name = os.environ.get('galaxy_name')

params      = parameter_read(os.path.join(base_path,'inputs','parameter_file.in'))
switch      = parameter_read(os.path.join(base_path,'inputs','switches.in'))

sys.path.append(os.path.join(base_path,'data','supplementary_data', galaxy_name))
# importing literature data for field strength, pitch angle, velocity dispersion and scale height 
from observables import * 

current_directory = str(os.getcwd())

os.chdir(os.path.join(base_path,'outputs'))

with open(f'{galaxy_name}output_ca_'+str(params[r'C_\alpha'])+'K_'+str(params[r'K'])+'z_'+str(params[r'\zeta'])+'psi_'+str(params[r'\psi'])+'b_'+str(params[r'\beta'])+'.out', 'rb') as f:
    kpc_r, h_f, l_f, u_f, cs_f, alphak_f, taue_f, taur_f, biso_f, bani_f, Bbar_f, tanpB_f, tanpb_f , dkdc_f = pickle.load(
        f)
with open('errors_subsonic.out', 'rb') as f:
        subsonic_errors= pickle.load(f)
with open('errors_supersonic.out', 'rb') as f:
        supersonic_errors= pickle.load(f)
h_err, l_err, u_err, cs_err, alphak_err, tau_err, \
        taur_err, biso_err, bani_err, Bbar_err, \
                tanpB_err, tanpb_err, dkdc_err = [np.maximum(sub, sup) for sub,sup in zip(subsonic_errors, supersonic_errors)]

# delete this file from the outputs folder
os.remove(f'{galaxy_name}output_ca_'+str(params[r'C_\alpha'])+'K_'+str(params[r'K'])+'z_'+str(params[r'\zeta'])+'psi_'+str(params[r'\psi'])+'b_'+str(params[r'\beta'])+'.out')

# compare errors in sub/super-sonic regimes and save the maximum of the two
err_quant_list = [np.maximum(sub, sup) for sub,sup in zip(subsonic_errors, supersonic_errors)]

os.chdir(os.path.join(base_path,'inputs'))

with open('zip_data.in', 'rb') as f:
    kpc_r, data_pass = pickle.load(f)

# interpolate velocity and scale height data as per availability
dat_u = griddata(kpc_radius, np.sqrt(3)*kms_sigmaLOS, kpc_r, method='linear',
                 fill_value=nan, rescale=False)*1e+5
try: # only for M31
    dat_u_warp = griddata(kpc_radius, np.sqrt(3)*kms_sigmaLOS_warp, kpc_r, method='linear',
                    fill_value=nan, rescale=False)*1e+5
except NameError:
    pass

try: # Bacchini+19 data for NGC 6946
    dat_u_bacchini = griddata(kpc_radius_b, np.sqrt(3)*kms_sigmaLOS_b, kpc_r, method='linear',
                    fill_value=nan, rescale=False)*1e+5
except NameError:
    pass

try: # Bacchini+19 data for NGC 6946 h
    dat_h_bacchini = griddata(kpc_radius_h_b, h_b, kpc_r, method='linear',
                    fill_value=nan, rescale=False)
except NameError:
    pass

os.chdir(current_directory)

# # calculate pitch angles and errors
# pB, po, pb, pB_err, po_err, pb_err = analytical_pitch_angle_integrator(kpc_r, tanpB_f,tanpb_f, \
#                                    Bbar_f, bani_f, tanpB_err,tanpb_err, Bbar_err, bani_err)

# calculate pitch angles and errors
pB, po, pb, pB_err, po_err, pb_err = new_pitch_angle_integrator(kpc_r, tanpB_f,tanpb_f, \
                                   Bbar_f, bani_f, tanpB_err,tanpb_err, Bbar_err, bani_err, taue_f, data_pass)

# calculate observational analogues of field strengths and errors
G_scal_Bbartot = np.sqrt(biso_f**2 + bani_f**2 + Bbar_f**2)
G_scal_Bbarreg = Bbar_f
G_scal_Bbarord = np.sqrt(bani_f**2 + Bbar_f**2)

G_scal_Bbartot_err = np.sqrt((biso_err*biso_f )**2+ (bani_err*bani_f)**2 + (Bbar_err*Bbar_f)**2)/G_scal_Bbartot
G_scal_Bbarreg_err = Bbar_err
G_scal_Bbarord_err = np.sqrt((bani_err*bani_f)**2 + (Bbar_err*Bbar_f)**2)/G_scal_Bbarord

############################################################################################################################

# set the plot parameters
m                 = 9 #marker size
lw                = 3
dm                = 2.5
fs                = 20
lfs               = 10
leg_textsize      = 18
title_textsize    = 25
axis_textsize     = 20
hd                = 1.6  # handlelength: changes length of line in legend
legend_labelspace = 0.17 # handletextpad: gap between label and symbol in legend
frameon_param     = True # true when frame is needed in legend
frame_alpha_param = 0.5  # opacity of legend box
rc                = {"font.family" : "serif", "mathtext.fontset" : "stix"}

plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
matplotlib.rc('xtick', labelsize=fs)
matplotlib.rc('ytick', labelsize=fs)
matplotlib.ticker.AutoMinorLocator(n=None)
plt.rcParams["xtick.minor.visible"] =  True
plt.rcParams["ytick.minor.visible"] =  True
plt.rcParams["legend.loc"] = 'upper right'
plt.rcParams["errorbar.capsize"] = 2

def axis_pars(ax):
    # add tick marks without labels at the top and right of the plot also
    if galaxy_name == 'm31':
        # add the plot title 
        ax.set_title(r'M31', fontsize = title_textsize)
        # set bottom of x axis to be 6
        ax.set_xlim(left=6)
        ax.xaxis.set_ticks(np.arange(6, 17, 1)) 

    elif galaxy_name == 'm33':
        ax.set_title(r'M33', fontsize=title_textsize)
        ax.set_xlim(left=0.5, right=8.5)
        ax.xaxis.set_ticks(np.arange(1, 9, 1))

    elif galaxy_name == 'm51':
        ax.set_title(r'M51', fontsize=title_textsize)
        ax.set_xlim(left=1)
        ax.xaxis.set_ticks(np.arange(1, 9, 1)) 

    else:
        ax.set_title(r'NGC 6946', fontsize=title_textsize)
        ax.set_xlim(left=0.3, right=19)
        ax.xaxis.set_ticks(np.arange(0, 20.5, 2)) 

    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.tick_params(axis='both', which='minor',
                   labelsize=axis_textsize, colors='k', length=3, width=1)
    ax.tick_params(axis='both', which='major',
                   labelsize=axis_textsize, colors='k', length=5, width=1.25)

# saving the errors as csv file
# save_files_dir_err = r'D:\Documents\Gayathri_college\MSc project\codes\GOBLIN\goblin\data\supplementary_data\{}'.format(galaxy_name)
save_files_dir_err = os.path.join(base_path,'data','supplementary_data',galaxy_name)
os.chdir(save_files_dir_err)

filename = r'{}_quant_err_moldat_{},taue,z_{},psi_{},ca_{},beta_{},A_{}.csv'.format(galaxy_name,switch['incl_moldat'],params[r'\zeta'],params[r'\psi'],
                params[r'C_\alpha'],params[r'\beta'],params['A'])

rel_err_transpose = list(zip(*err_quant_list))
column_names = ['h_err', 'l_err', 'u_err', 'cs_err', 'alphak_err', 'tau_err', \
                'taur_err', 'biso_err', 'bani_err', 'Bbar_err', \
                'tanpB_err', 'tanpb_err', 'dkdc_err']

# Writing to the file
with open(os.path.join(save_files_dir_err,filename), 'w', newline = '') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(column_names)
    csvwriter.writerows(rel_err_transpose)

switches_info = r'{},moldat_{},{},KS_{},u_{},h_{},z_{},psi_{},ca_{},beta_{},A_{}'.format(str(today),
                                                                                          switch['incl_moldat'],switch['tau'],
                                                                                          switch['force_kennicut_scmidt'][0],switch['u'],
                                                                                          switch['h'],params[r'\zeta'],params[r'\psi'],
                                                                                            params[r'C_\alpha'],params[r'\beta'],params['A'])
# define folder names for outputs 
save_files_dir = os.path.join(base_path,'results',galaxy_name,switches_info)

# new directories wont be made when the directory name is imported from this file
if __name__ == '__main__': 
    try:
        os.makedirs(save_files_dir)
        #os.chdir(save_files_dir)
    except FileExistsError:
        # Handle the case where the directory already exists
        print(f"The directory '{save_files_dir}' already exists, going there.")
        #os.chdir(save_files_dir)
        #anything in this folder before will be re-written
    except OSError as e:
        # Handle other OSError exceptions if they occur
        print(f"An error occurred while creating the directory: {e}")


######################## PLOTTING LENGTH SCALES ##################################################################
fig,ax = plt.subplots(nrows = 1, ncols = 1, figsize = (7, 5), constrained_layout = True)

ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')

h = h_f/cm_kpc #in kpc
l = l_f/cm_kpc #in kpc
# l = l_f/cm_kpc #in kpc

# defining h data
if galaxy_name == 'ngc6946':
    # defining data from Patra+20 for NGC 6946
    h_NGC6946_Patra_kpc = (38.9 + 23.9*kpc_r)/1000 #in kpc
    ax.plot(kpc_r, h_NGC6946_Patra_kpc, linestyle=':', linewidth=lw, label=r' Patra (2020)')
elif galaxy_name == 'm33':
    # defining the data from Braun+91 for M31
    h_M31_Braun_kpc = (187 + 16*kpc_r)/1000 #in kpc
    ax.plot(kpc_r, h_M31_Braun_kpc, linestyle=':', linewidth=lw, label=r' Braun et al. (1991)')

# Hyper-LEDA values
log_d25_arcmin_paper2 = [3.25, 2.79, 2.14, 2.06]
d25_arcmin_paper2     = 0.1*(10**np.array(log_d25_arcmin_paper2))
r_25_arcmin_paper2    = d25_arcmin_paper2/2
dist_Mpc_paper2       = [0.78, 0.84, 8.5, 7.72]
# convert arcmin to radius in kpc using distance to these galaxies
r_25_kpc_paper2       = [r*dist_Mpc_paper2[i]*1000/(arcmin_deg*deg_rad) for i,r in enumerate(r_25_arcmin_paper2)]

# plotting data
if galaxy_name == 'm31':
    # plotting model output
    ax.plot(kpc_r, h, c='b', marker='o', markersize=4, mfc='k',mec='k',linestyle='-', linewidth=lw, label=r' Scale height')
    ax.plot(kpc_r, l, c='g', marker='o', markersize=4, mfc='k',mec='k', linestyle='-', linewidth=4, label=r' Turbulent correlation length')

    # MW scaling with exponental relation from C16
    h_scaled = 0.18*np.exp(kpc_r/((10/16)*r_25_kpc_paper2[0])) # in kpc
    ax.plot(kpc_r, h_scaled, zorder=2,linestyle='-',marker=' ',c='r',mfc='b',mec='k',mew=1, linewidth=1, markersize = 13, label=r' Milky Way scaling for $h$')

else: # since no legend needed for m33, m51 and ngc6946
    # plotting model output
    ax.plot(kpc_r, h, c='b', marker='o', markersize=4, mfc='k',mec='k',linestyle='-', linewidth=lw)
    # ax.plot(kpc_r, l, c='g',linestyle='-.', mfc='k', mec='k', markersize=m, marker='o', label=r'$l$ (pc)')
    ax.plot(kpc_r, l, c='g',marker='o', markersize=4, mfc='k',mec='k',linestyle='-', linewidth=lw)

    # MW scaling with exponental relation from C16
    if galaxy_name == 'm33':
        h_scaled = 0.18*np.exp(kpc_r/((10/16)*r_25_kpc_paper2[1])) # in kpc
    elif galaxy_name == 'm51':
        h_scaled = 0.18*np.exp(kpc_r/((10/16)*r_25_kpc_paper2[2]))
    else: # NGC 6946
        h_scaled = 0.18*np.exp(kpc_r/((10/16)*r_25_kpc_paper2[3]))
        # plot Bacchini+19 data
        ax.plot(kpc_r, dat_h_bacchini, zorder=2,linestyle=':',marker=' ',c='k',mfc='k',mec='k',mew=1, linewidth=2, markersize = 13, label = ' Bacchini et al. (2019)')
    ax.plot(kpc_r, h_scaled, zorder=2,linestyle='-',marker=' ',c='r',mfc='b',mec='k',mew=1, linewidth=1, markersize = 13)

l_err_corr_units = l_err/cm_kpc
percent_err_l    = (l_err_corr_units/l)*100
try:
    fill_error(ax, kpc_r, l, l_err_corr_units,'g', 0.2)
except NameError:
    pass

if galaxy_name == 'm31':
    bbox_to_anchor = (0.08, 0.63, 0.3, 0.3)  # Example values: (x0, y0, width, height)
    axins = inset_axes(ax, width="100%", height="100%", loc='upper left', 
                    bbox_to_anchor=bbox_to_anchor, bbox_transform=ax.transAxes)
    # change transparency of inset plot
    axins.patch.set_alpha(0.5)
elif galaxy_name == 'm33':
    bbox_to_anchor = (0.1, 0.60, 0.4, 0.4)  
    axins = inset_axes(ax, width="100%", height="100%", loc='upper left', 
                    bbox_to_anchor=bbox_to_anchor, bbox_transform=ax.transAxes)
elif galaxy_name == 'ngc6946':
    bbox_to_anchor = (0.08, 0.60, 0.4, 0.4)  
    axins = inset_axes(ax, width="100%", height="100%", loc='upper left', 
                    bbox_to_anchor=bbox_to_anchor, bbox_transform=ax.transAxes)
else:
    bbox_to_anchor = (0, 0.7, 0.4, 0.4)  
    axins = inset_axes(ax, width="100%", height="100%", loc='upper left', 
                    bbox_to_anchor=bbox_to_anchor, borderpad=4.5, bbox_transform=ax.transAxes)

axins.xaxis.set_ticks_position('both')
axins.yaxis.set_ticks_position('both')
axins.tick_params(axis='both', which='major', labelsize=17)
axins.plot(kpc_r, l, c='g',marker='o', markersize=4, mfc='k',mec='k',linestyle='-', linewidth=lw)

try:
    fill_error(axins, kpc_r, l, l_err_corr_units,'g', 0.2)
except NameError:
    pass

axis_pars(ax)

# scale height error
h_err_corr_units = h_err/cm_kpc #in kpc
percent_err_h    = (h_err_corr_units/h)*100
try:
    fill_error(ax, kpc_r, h, h_err_corr_units,'b', 0.2)
except NameError:
    pass

# log scale if moldata is included
if switch['incl_moldat'] == 'Yes':
    #ax.set_yscale('log')
    pass
else:
    if galaxy_name == 'ngc6946':
        ax.set_ylim(bottom = 0)
        ax.yaxis.set_ticks(np.arange(0,2,0.2)) # for kpc data
    else:
        ax.set_ylim(bottom = 0)
        ax.yaxis.set_ticks(np.arange(0,max(h_err_corr_units+h)+0.1,0.2)) # for kpc data

ax.set_xlabel(r'Radius (kpc)', fontsize   = fs)
ax.set_ylabel(r'$h$, $l$ (kpc)', fontsize = fs)

# legend specs 
if galaxy_name == 'm31':
    ax.legend(fontsize = lfs, frameon = frameon_param, handlelength = hd, ncol = 1, bbox_to_anchor = (1, 1), prop = {
            'size': leg_textsize-1,  'family': 'Times New Roman'}, fancybox = True, framealpha = frame_alpha_param, handletextpad = legend_labelspace, columnspacing = 0.7)
elif galaxy_name == 'm51':
    ax.legend(fontsize = lfs, frameon = False, handlelength = hd, ncol = 1, bbox_to_anchor = (0.65, 0.8), prop = {
            'size': leg_textsize,  'family': 'Times New Roman'}, fancybox = True, framealpha = frame_alpha_param, handletextpad = legend_labelspace, columnspacing = 0.7)
elif galaxy_name == 'm33':
    ax.legend(fontsize = lfs, frameon = frameon_param, handlelength = hd, ncol = 1, bbox_to_anchor = (1, 1),prop={
            'size': leg_textsize, 'family': 'Times New Roman'}, fancybox = True, framealpha = frame_alpha_param, handletextpad = legend_labelspace, columnspacing = 0.7)
elif galaxy_name == 'ngc6946':
    ax.legend(fontsize = lfs, frameon = frameon_param, handlelength = hd, ncol = 1, bbox_to_anchor = (1, 1),prop = {
            'size': leg_textsize-1, 'family': 'Times New Roman'}, fancybox = True, framealpha = frame_alpha_param, handletextpad = legend_labelspace, columnspacing = 0.7)

plt.savefig(save_files_dir+r'\1 h,l')

#################### PLOTTING VELOCITY DISPERSIONS ###############################################################
fig,ax = plt.subplots(nrows = 1, ncols = 1, figsize = (7, 5), tight_layout = True)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')

# converting from cgs units
u         = u_f/cm_km
cs        = cs_f/cm_km # speed of diffuse component, common to both switch ON and OFF of moldata
sig       = (np.sqrt(u_f**2 + (cs_f)**2))/cm_km
dat_u     = dat_u/cm_km

# to consider cs in molecular gas if moldata is included
# define weighted average of cs
# if switch['incl_moldat'] == 'Yes':
#     cs_moldat = cs/10 # in km/s

#     # obtain sigma_HI and sigma_H2 from interpolated_data files
#     os.chdir(os.path.join(base_path,'data'))

#     # open the file data_interpolated_galaxyname.csv and take columns named sigma_HI and sigma_H2
#     df_obs  = pd.read_csv('data_interpolated_{}.csv'.format(galaxy_name))
#     if galaxy_name == 'm31':
#         sigma_HI  = np.array(df_obs['sigma_HI_claude'])
#     else: 
#         sigma_HI  = np.array(df_obs['sigma_HI'])
#     sigma_H2      = np.array(df_obs['sigma_H2'])
#     sigma_gas     = (3*params['mu']/(4-params['mu']))*sigma_HI+ (params['mu_prime']/(4-params['mu_prime']))*sigma_H2 
#     cs = np.sqrt(((sigma_H2*(cs_moldat**2) + sigma_HI*(cs**2)))/sigma_gas) # in km/s

# legend details for sound speed and velocity dispersion data
if galaxy_name == 'm31':
    ref_cs    = ' (Tabatabaei et al. 2013)'
    ref_data  = ' C. Carignan (no warp)'
elif galaxy_name == 'm33':
    ref_cs    = ' (Lin et al. 2017)'
    ref_data  = ' Kam et al. (2017)'
elif galaxy_name == 'm51':
    ref_cs    = ' (Bresolin et al. 2004)'
    ref_data  = ' Hitschfeld et al. (2009)'
else:
    ref_cs    = ' (Gusev et al. 2013)'
    ref_data  = ' Boomsma et al. (2008)' 

# extra data for M31 and NGC 6946
if galaxy_name == 'm31':
    dat_u_warp = dat_u_warp/cm_km
    ax.plot(kpc_r, dat_u_warp, c='g',  linestyle=' ', label=r' C. Carignan (with warp)', alpha = 1,marker='D',mfc='b',mec='k',mew=1, markersize = 8)
elif galaxy_name == 'ngc6946':
    dat_u_bacchini = dat_u_bacchini/cm_km
    ax.plot(kpc_r, dat_u_bacchini, c='g',  linestyle=' ', label=r' Bacchini et al. (2019)', alpha = 1,marker='D',mfc='b',mec='k',mew=1, markersize = 8)
# data for u all galaxies
ax.plot(kpc_r, dat_u, c='g', label=r'{}'.format(ref_data),linestyle=' ',alpha = 1,marker='*',mfc='b',mec='k',mew=1, markersize = 13)

if galaxy_name == 'm31':
    ax.plot(kpc_r, cs, color='g', zorder=3, linewidth=lw, marker='o',markersize=4,mfc='k',mec='k', label=r' $c_\mathrm{s}$'+ '{}'.format(ref_cs))
    ax.plot(kpc_r, u, color='red',marker='o',markersize=4,mfc='k',mec='k',linewidth=lw, label=r' $u$')
    ax.plot(kpc_r, sig, color='b', linewidth=lw, marker='o',markersize=4,mfc='k',mec='k', label=r' $w$')
else: # since legend needed only for M31
    ax.plot(kpc_r, cs, c='g',zorder=3,marker='o', markersize=4, mfc='k',mec='k',linestyle='-', linewidth=lw, label=r' $c_\mathrm{s}$'+ '{}'.format(ref_cs))
    ax.plot(kpc_r, u, color='r', linewidth=lw, marker='o', markersize=4, mfc='k', mec='k')
    ax.plot(kpc_r, sig, color='b', linewidth=lw, marker='o', markersize=4, mfc='k', mec='k')

# error bars for velocity dispersion
err_u         = u_err/cm_km
percent_err_u = (u_err/u_f)*100
try:
    fill_error(ax, kpc_r, u, err_u, 'red', 0.2)
except NameError:
    pass

# modify cs_error to also include a 20 percent error in gamma
# get relative error
rel_err_cs = cs_err/cs_f # equal to del_T/T
# add 20 percent error in gamma = 1.5
rel_err_cs = np.sqrt(rel_err_cs**2 + (0.2)**2)
# get absolute error
cs_err = rel_err_cs*cs_f

sig_err = np.sqrt((u_f*u_err)**2 + (cs_f*cs_err)**2)/(sig*cm_km**2)
try:
    fill_error(ax, kpc_r, sig, sig_err, 'b', 0.2)
except NameError:
    pass

err_cs         = cs_err/cm_km
percent_err_cs = (err_cs/cs)*100
try:
    fill_error(ax, kpc_r, cs, err_cs, 'g', 0.2)
except NameError:
    pass

# calculating Mach number and corresponding error
Mach             = u/cs
Mach_err         = Mach*np.sqrt((err_u/u)**2 + (err_cs/cs)**2) # in SI units
percent_err_Mach = (Mach_err/Mach)*100

err_tot         = sig_err
percent_err_sig = (sig_err/sig)*100

# legend specs
axis_pars(ax)
# log scale if moldata is included
if switch['incl_moldat'] == 'Yes':
    # ax.set_yscale('log')
    if galaxy_name == 'm31':
        ax.set_ylim(bottom=4)
        ax.yaxis.set_ticks(np.arange(4,max(sig+sig_err)+16,4))
        ax.legend(fontsize=lfs, frameon=frameon_param, handlelength=hd, ncol=2, bbox_to_anchor=(1, 1),prop={
                'size': leg_textsize, 'family': 'Times New Roman'}, fancybox=True, framealpha=frame_alpha_param, handletextpad=legend_labelspace, columnspacing=0.7)
    elif galaxy_name == 'm33':
        ax.set_ylim(bottom=2)
        ax.yaxis.set_ticks(np.arange(2,max(sig+sig_err)+4,2))
        ax.legend(fontsize=lfs, frameon=frameon_param, handlelength=hd, ncol=1, bbox_to_anchor=(1, 1),prop={
                'size': leg_textsize, 'family': 'Times New Roman'}, fancybox=True, framealpha=frame_alpha_param, handletextpad=legend_labelspace, columnspacing=0.7)
    elif galaxy_name == 'm51':
        if params['\zeta'] == 10 and params['\psi'] == 1:
            ax.set_ylim(bottom=6)
            ax.yaxis.set_ticks(np.arange(6,max(sig+sig_err)+3,2))
        elif params['\zeta'] == 10 and params['\psi'] == 1.5:
            ax.set_ylim(bottom=6)
            ax.yaxis.set_ticks(np.arange(6,max(sig+sig_err)+3,4))
        elif params['\zeta'] == 15 and params['\psi'] == 1:
            ax.set_ylim(bottom=0)
            ax.yaxis.set_ticks(np.arange(0,max(dat_u)+2,4))
        else:
            ax.set_ylim(bottom=0)
            ax.yaxis.set_ticks(np.arange(0,max(sig+sig_err)+8,5))
        ax.legend(fontsize=lfs, frameon=frameon_param, handlelength=hd, ncol=1, bbox_to_anchor=(0.6, 1),prop={
                'size': leg_textsize, 'family': 'Times New Roman'}, fancybox=True, framealpha=frame_alpha_param, handletextpad=legend_labelspace, columnspacing=0.7)
    else:
        ax.set_ylim(bottom=6)
        ax.yaxis.set_ticks(np.arange(0,max(sig+sig_err)+8,4))#changed from 5 to 0
        ax.legend(fontsize=lfs, frameon=frameon_param, handlelength=hd, ncol=1, bbox_to_anchor=(1, 1),prop={
                'size': leg_textsize, 'family': 'Times New Roman'}, fancybox=True, framealpha=frame_alpha_param, handletextpad=legend_labelspace, columnspacing=0.7)

else:
    if galaxy_name == 'm31':
        ax.set_ylim(bottom=0)#changed to 0
        ax.yaxis.set_ticks(np.arange(0,max(dat_u)+16,4))
        ax.legend(fontsize=lfs, frameon=frameon_param, handlelength=hd, ncol=2, bbox_to_anchor=(1, 1),prop={
                'size': leg_textsize, 'family': 'Times New Roman'}, fancybox=True, framealpha=frame_alpha_param, handletextpad=legend_labelspace, columnspacing=0.7)
    elif galaxy_name == 'm33':
        ax.set_ylim(bottom=2)
        ax.yaxis.set_ticks(np.arange(2,max(dat_u)+4,2))
        ax.legend(fontsize=lfs, frameon=frameon_param, handlelength=hd, ncol=1, bbox_to_anchor=(1, 1),prop={
                'size': leg_textsize, 'family': 'Times New Roman'}, fancybox=True, framealpha=frame_alpha_param, handletextpad=legend_labelspace, columnspacing=0.7)
    elif galaxy_name == 'm51':
        if params['\zeta'] == 10 and params['\psi'] == 1:
            ax.set_ylim(bottom=6)
            ax.yaxis.set_ticks(np.arange(6,max(dat_u)+3,2))
        elif params['\zeta'] == 10 and params['\psi'] == 1.5:
            ax.set_ylim(bottom=6)
            ax.yaxis.set_ticks(np.arange(6,max(sig+sig_err)+3,4))
        elif params['\zeta'] == 15 and params['\psi'] == 1:
            ax.set_ylim(bottom=6)
            ax.yaxis.set_ticks(np.arange(5,max(sig+sig_err)+2,4))
        else:
            ax.set_ylim(bottom=6)
            ax.yaxis.set_ticks(np.arange(6,max(dat_u)+8,2))
        ax.legend(fontsize=lfs, frameon=frameon_param, handlelength=hd, ncol=1, bbox_to_anchor=(1, 1),prop={
                'size': leg_textsize, 'family': 'Times New Roman'}, fancybox=True, framealpha=frame_alpha_param, handletextpad=legend_labelspace, columnspacing=0.7)
    else:
        ax.set_ylim(bottom=0)
        ax.yaxis.set_ticks(np.arange(0,max(sig)+8,4))
        ax.legend(fontsize=lfs, frameon=frameon_param, handlelength=hd, ncol=1, bbox_to_anchor=(1, 1),prop={
                'size': leg_textsize, 'family': 'Times New Roman'}, fancybox=True, framealpha=frame_alpha_param, handletextpad=legend_labelspace, columnspacing=0.7)

ax.set_xlabel(r'Radius (kpc)', fontsize=fs)
ax.set_ylabel(r'Speed (km s$^{-1}$)',  fontsize=fs)

plt.savefig(save_files_dir+r'\2 speeds')

############################ Plotting field strengths ############################

# Converting to micro Gauss
Btot = G_scal_Bbartot*1e+6
Breg = G_scal_Bbarreg*1e+6
Bord = G_scal_Bbarord*1e+6

fig,ax = plt.subplots(nrows = 1, ncols = 1, figsize = (7, 5), tight_layout = True)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')

# shading radial region for Beck+19 data
for i in range(len(mrange_endps)-1):
    try:
        plot_rectangle(ax, mrange_endps[i], G_dat_Btot[i]-err_G_dat_Btot[i], mrange_endps[i+1], G_dat_Btot[i]+err_G_dat_Btot[i], color='gray')
    except NameError:
        pass
    try:
        plot_rectangle(ax, mrange_endps[i], G_dat_Breg[i]-err_G_dat_Breg[i], mrange_endps[i+1], G_dat_Breg[i]+err_G_dat_Breg[i], color='gray')
    except NameError:
        pass
    try:
        plot_rectangle(ax, mrange_endps[i], G_dat_Bord[i]-err_G_dat_Bord[i], mrange_endps[i+1], G_dat_Bord[i]+err_G_dat_Bord[i], color='gray')
    except NameError:
        pass

# plot model outputs and Beck+19 data points
if galaxy_name == 'm31':
    ax.plot(kpc_r, Btot , c='b', linewidth=lw, marker='o',markersize=4,mfc='k',mec='k', label=r' $B_{\mathrm{tot}}$')
    ax.plot(kpc_r, Bord , c='g', linewidth=lw, marker='o',markersize=4,mfc='k',mec='k', label=r' $B_{\mathrm{ord}}$')
    ax.plot(kpc_r, Breg , c='r', linewidth=lw, marker='o',markersize=4,mfc='k',mec='k', label=r' $B_{\mathrm{reg}}$')

    ax.errorbar(mrange, G_dat_Btot, zorder=2, elinewidth=1, yerr=err_G_dat_Btot, mew=1, capsize=7,
                c='b', linestyle=' ', marker='*', mfc='b', mec='k',label=r' Fletcher et al. (2004)',ecolor='k',markersize= 13)
    ax.errorbar(mrange, G_dat_Bord, zorder=2, elinewidth=1, yerr=err_G_dat_Bord, mew=1, capsize=7,
                c='g', linestyle=' ', marker='D', mfc='g', mec='k',label=r' Fletcher et al. (2004)',ecolor='k',markersize= 8.75)
    ax.errorbar(mrange, G_dat_Breg, zorder=2, elinewidth=1, yerr=err_G_dat_Breg, mew=1, capsize=7,
                c='r', linestyle=' ', marker='s', mfc='r', mec='k',label=r' Beck et al. (2019)',ecolor='k',markersize= 8.75)

elif galaxy_name == 'm33':
    ax.plot(kpc_r, Btot , c='b', linewidth=lw, marker='o',markersize=4,mfc='k',mec='k')
    ax.plot(kpc_r, Breg , c='r', linewidth=lw, marker='o',markersize=4,mfc='k',mec='k')
    ax.plot(kpc_r, Bord , c='g', linewidth=lw, marker='o',markersize=4,mfc='k',mec='k')

    ax.plot(mrange, G_dat_Btot, zorder=2,c='b', linestyle=' ', marker='*',mfc='b',mec='k',mew=1,markersize = 13, label=r' Tabatabaei et al. (2008)')#, label='Average Binned data $B_{tot}$ ($\mu G$)')
    ax.errorbar(mrange, G_dat_Breg, zorder=2, elinewidth=1, yerr=err_G_dat_Breg, mew=1, capsize=7,
                c='r', linestyle=' ', marker='s', mfc='r', mec='k',label=r' Beck et al. (2019)',ecolor='k',markersize= 8.75)
    ax.plot(mrange, G_dat_Bord, zorder=2,c='g', linestyle=' ', marker='D',mfc='g',mec='k',mew=1,markersize = 8.75, label=r' Tabatabaei et al. (2008)')#, label='Average Binned data $B_{ord}$ ($\mu G$)')

elif galaxy_name == 'm51':
    ax.plot(kpc_r, Btot , c='b', linewidth=lw, marker='o',markersize=4,mfc='k',mec='k')
    ax.plot(kpc_r, Breg , c='r', linewidth=lw, marker='o',markersize=4,mfc='k',mec='k')
    ax.plot(kpc_r, Bord , c='g', linewidth=lw, marker='o',markersize=4,mfc='k',mec='k')

    ax.plot(mrange, G_dat_Btot, zorder=2,c='b', linestyle=' ', marker='*',mfc='b',mec='k',mew=1,markersize = 13, label=r' Fletcher et al. (2011)')#, label='Average Binned data $B_{tot}$ ($\mu G$)')
    ax.plot(mrange, G_dat_Bord, zorder=2,c='g', linestyle=' ', marker='D',mfc='g',mec='k',mew=1,markersize = 8.75, label=r' Fletcher et al. (2011)')#, label='Average Binned data $B_{ord}$ ($\mu G$)')
    ax.errorbar(mrange, G_dat_Breg, zorder=2, elinewidth=1, yerr=err_G_dat_Breg, mew=1, capsize=7,
                c='r', linestyle=' ', marker='s', mfc='r', mec='k',label=r' Beck et al. (2019)',ecolor='k',markersize= 8.75)
else:
    ax.plot(kpc_r, Btot , c='b', linewidth=lw, marker='o',markersize=4,mfc='k',mec='k')
    ax.plot(kpc_r, Breg , c='r', linewidth=lw, marker='o',markersize=4,mfc='k',mec='k')
    ax.plot(kpc_r, Bord , c='g', linewidth=lw, marker='o',markersize=4,mfc='k',mec='k')

    ax.plot(mrange, G_dat_Btot, zorder=2,c='b', linestyle=' ', marker='*',mfc='b',mec='k',mew=1,markersize = 13, label=r' Ehle & Beck (1993)')#, label='Average Binned data $B_{tot}$ ($\mu G$)')
    ax.errorbar(mrange, G_dat_Breg, zorder=2, elinewidth=1, yerr=err_G_dat_Breg, mew=1, capsize=7,
                c='r', linestyle=' ', marker='s', mfc='r', mec='k',label=r' Beck et al. (2019)',ecolor='k',markersize= 8.75)
    ax.plot(mrange, G_dat_Bord, zorder=2,c='g', linestyle=' ', marker='D',mfc='g',mec='k',mew=1,markersize = 8.75, label=r' Ehle & Beck (1993)')#, label='Average Binned data $B_{ord}$ ($\mu G$)')
    ax.errorbar(kpc_Btot_Basu_Roy_ngc6946, G_dat_Btot_Basu_Roy_ngc6946, zorder=2, elinewidth=1, yerr=err_G_dat_Btot_Basu_Roy_ngc6946, mew=1, capsize=7,
                c='b', linestyle=' ', marker='p', mfc='b', mec='k',label=r' Basu & Roy (2013)',ecolor='k',markersize= 8.75)
    
# plotting error
try:
    fill_error(ax, kpc_r, G_scal_Bbartot*1e+6,G_scal_Bbartot_err*1e+6, 'b', 0.2)
    fill_error(ax, kpc_r, G_scal_Bbarreg*1e+6,G_scal_Bbarreg_err*1e+6, 'r', 0.2)
    fill_error(ax, kpc_r, G_scal_Bbarord*1e+6,G_scal_Bbarord_err*1e+6, 'g', 0.2)
except NameError:
    pass

percent_err_Btot = (G_scal_Bbartot_err/G_scal_Bbartot)*100
percent_err_Breg = (G_scal_Bbarreg_err/G_scal_Bbarreg)*100
percent_err_Bord = (G_scal_Bbarord_err/G_scal_Bbarord)*100

ax.set_xlabel(r'Radius (kpc)', fontsize = fs)

# log scale if moldata is included
if switch['incl_moldat'] == 'Yes':
    pass
    #ax.set_yscale('log')
else:
    ax.set_ylim(bottom=0)
    ax.yaxis.set_ticks(np.arange(0,max(Btot)+max(G_scal_Bbartot_err*1e+6)+4,5))


# ax.set_ylim(bottom=0)
# ax.yaxis.set_ticks(np.arange(0,max(Btot)+max(G_scal_Bbartot_err*1e+6)+4,10))
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.set_ylabel('Magnetic field strength ($\mathrm{\mu G}$)', fontsize=fs)

# legend specs
axis_pars(ax)
if galaxy_name == 'm31':
    ax.legend(fontsize=lfs, frameon=frameon_param, handlelength=hd, ncol=2, bbox_to_anchor=(1,1),prop={
            'size': leg_textsize, 'family': 'Times New Roman'}, fancybox=True, framealpha=frame_alpha_param, handletextpad=legend_labelspace, columnspacing=0.7)
else:
    ax.legend(fontsize=lfs, frameon=frameon_param, handlelength=hd, ncol=1, bbox_to_anchor=(1,1),prop={
            'size': leg_textsize, 'family': 'Times New Roman'}, fancybox=True, framealpha=frame_alpha_param, handletextpad=legend_labelspace, columnspacing=0.7)

plt.savefig(save_files_dir+r'\3 B')

######################## Pitch angles ##################################

fig,ax = plt.subplots(nrows = 1, ncols = 1, figsize = (7, 5), tight_layout = True)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
pb_bool = False
# plotting model outputs
if galaxy_name == 'm31':
    ax.plot(kpc_r, -180*pB/np.pi , c='r', linestyle='-', linewidth=lw, marker='o',markersize=4,mfc='k',mec='k', label=r' $-p_{\mathrm{reg}}$')
    ax.plot(kpc_r, -180*po/np.pi , c='g', linestyle='-', linewidth=lw, marker='o',markersize=4,mfc='k',mec='k', label=r' $-p_{\mathrm{ord}}$')
    if pb_bool: ax.plot(kpc_r, -180*pb/np.pi , c='b', linestyle='-', linewidth=lw, marker='o',markersize=4,mfc='k',mec='k', label=r' $-p_{\mathrm{b}}$')
else:
    ax.plot(kpc_r, -180*pB/np.pi , c='r', linestyle='-', linewidth=lw, marker='o',markersize=4,mfc='k',mec='k')
    ax.plot(kpc_r,-180*po/np.pi , c='g', linestyle='-', linewidth=lw, marker='o',markersize=4,mfc='k',mec='k')
    if pb_bool: ax.plot(kpc_r,-180*pb/np.pi , c='b', linestyle='-', linewidth=lw, marker='o',markersize=4,mfc='k',mec='k')
# print(-180*pB/np.pi,-180*po/np.pi,-180*pb/np.pi)
# plotting error
try:
    fill_error(ax, kpc_r, -180*pB/np.pi,180*pB_err/np.pi, 'r')
except NameError:
    pass
try:
    fill_error(ax, kpc_r, -180*po/np.pi,180*po_err/np.pi, 'g')
except NameError:
    pass
if pb_bool: 
    try:
        fill_error(ax, kpc_r, -180*pb/np.pi,180*pb_err/np.pi, 'b')
    except NameError:
        pass

percent_err_pB = (pB_err/pB)*100
percent_err_po = (po_err/po)*100

# plotting observations
if galaxy_name == 'm31':

    ax.errorbar(mrange, 180*M_pb_beck19/np.pi,zorder=2,elinewidth=1, yerr=180*err_M_pb_beck19/np.pi,ecolor='k', mew=1, capsize=2,
                c='k', linestyle=' ', mfc='red', mec='k',label=r' $-p_{\mathrm{reg}}$ (Fletcher et al. 2004: M)',barsabove=True,marker='D',markersize=11)
    for i in range(len(mrange_endps)-1):
        plot_rectangle(ax, mrange_endps[i], 180*M_pb_beck19[i]/np.pi-180*err_M_pb_beck19[i]/np.pi, mrange_endps[i+1], 180*M_pb_beck19[i]/np.pi+180*err_M_pb_beck19[i]/np.pi, color='gray')
    
    ax.errorbar(range_po_beck19, 180*RM_pb_beck19/np.pi,zorder=2,elinewidth=1, yerr=180*err_RM_pb_beck19/np.pi,ecolor='k',  mew=1, capsize=2,
                c='k', linestyle=' ', mfc='red', mec='k',label=r' $-p_{\mathrm{reg}}$ (Beck et al. 2020: RM)',barsabove=True,marker='o',markersize=11)
    ax.errorbar(range_po_beck19, 180*po_beck19/np.pi,zorder=2,elinewidth=1, yerr=180*err_po_beck19/np.pi, mew=1, capsize=2,
                c='k', linestyle=' ', marker='P', mfc='g', mec='k',label=r' $-p_{\mathrm{ord}}$ (Beck et al. 2019)',ecolor='k',markersize=11)
    for i in range(len(rmrange_endps)-1):
        plot_rectangle(ax, rmrange_endps[i], 180*po_beck19[i]/np.pi-180*err_po_beck19[i]/np.pi, rmrange_endps[i+1], 180*po_beck19[i]/np.pi+180*err_po_beck19[i]/np.pi, color='gray'   )
        plot_rectangle(ax, rmrange_endps[i], 180*RM_pb_beck19[i]/np.pi-180*err_RM_pb_beck19[i]/np.pi, rmrange_endps[i+1], 180*RM_pb_beck19[i]/np.pi+180*err_RM_pb_beck19[i]/np.pi, color='gray'   )

elif galaxy_name == 'm33':

    ax.errorbar(mrange, 180*pb_beck19/np.pi,zorder=2,elinewidth=1, yerr=180*err_pb_beck19/np.pi,ecolor='k', ms=11, mew=1, capsize=2,
                  c='k', linestyle=' ', mfc='red', mec='k',marker='D',barsabove=True, label=r' $-p_{\mathrm{reg}}$ (Tabatabaei et al. 2008: M)')
    for i in range(len(mrange_endps)-1):
        plot_rectangle(ax, mrange_endps[i], 180*pb_beck19[i]/np.pi-180*err_pb_beck19[i]/np.pi, mrange_endps[i+1], 180*pb_beck19[i]/np.pi+180*err_pb_beck19[i]/np.pi, color='gray')
    
    ax.errorbar(po_range_beck19, 180*po_beck19/np.pi,zorder=2,elinewidth=1, yerr=180*err_po_beck19/np.pi, ms=11, mew=1, capsize=2,
                  c='k', linestyle=' ', marker='P', mfc='g', mec='k',ecolor='k') #,label=r' $p_{\mathrm{ord}}$ (Beck et al. (2019))')
    for i in range(len(range_po)-1):
        plot_rectangle(ax, range_po[i], 180*po_beck19[i]/np.pi-180*err_po_beck19[i]/np.pi, range_po[i+1], 180*po_beck19[i]/np.pi+180*err_po_beck19[i]/np.pi, color='gray'   )
        
elif galaxy_name == 'm51':

# #Beck+19 data
    ax.errorbar(range_po_beck19, po_beck19/(np.pi/180),zorder=2,elinewidth=1, yerr=err_po/(np.pi/180), markersize=11, mew=1, capsize=2,
                  c='k', linestyle=' ', marker='P', mfc='g', mec='k',ecolor='k')
    for i in range(len(po_endps_1)-1):
        plot_rectangle(ax, po_endps_1[i], po_beck19[i]/(np.pi/180)-err_po[i]/(np.pi/180), po_endps_1[i+1], po_beck19[i]/(np.pi/180)+err_po[i]/(np.pi/180), color='gray')
    for i in range(len(po_endps_2)-1):
        plot_rectangle(ax, po_endps_2[i], po_beck19[-1]/(np.pi/180)-err_po[-1]/(np.pi/180), po_endps_2[i+1], po_beck19[-1]/(np.pi/180)+err_po[-1]/(np.pi/180), color='gray')
    
    ax.errorbar(range_pb, dat_pb/(np.pi/180),zorder=2,elinewidth=1, yerr=180*err_pb/np.pi,ecolor='k', markersize=11, mew=1, capsize=2,
                  c='k', linestyle=' ', mfc='r', mec='k',marker='D',label=r' $-p_{\mathrm{reg}}$ (Fletcher et al. 2011: M)')
    for i in range(len(mrange_endps)-1):
        plot_rectangle(ax, mrange_endps[i], dat_pb[i]/(np.pi/180)-180*err_pb[i]/np.pi, mrange_endps[i+1], dat_pb[i]/(np.pi/180)+180*err_pb[i]/np.pi, color='gray')
#Borlaff+23 data
    ax.plot(range_po_borlaff23,po_borlaff23, zorder=2,c='k', linestyle=' ', marker='*',
              markersize=13, mfc='b', mec='k', label=r' $-p_{\mathrm{ord}}$ (Borlaff et al. 2023)')

#Borlaff+21 data
    ax.plot(range_po_borlaff21,po_borlaff21,zorder=2, c='k', linestyle=' ',
              markersize=m, mfc='orange', mec='k', label=r' $-p_{\mathrm{ord}}$ (Borlaff et al. 2021)', marker='s')

#Surgent+23 data
    ax.plot(range_po_surgent23,po_surgent23,zorder=2, c='k', linestyle=' ', marker='o',
              markersize=11, mfc='yellow', mec='k', label=r' $-p_{\mathrm{ord}}$ (Surgent et al. 2023)')

else:

    print('no p_reg data for 6946 currently')

#Beck+19
    ax.errorbar(range1_beck19, 180*po_range1_beck19/np.pi, yerr=180*err_po_range1_beck19/np.pi, zorder=2,ms=9, mew=1, capsize=2,linestyle=' ',
                   marker='D', mfc='g',label=r' $-p_{\mathrm{ord}}$ (Beck et al. 2019: range 1)', mec='k',ecolor='k')#, label=r'Data $p_{o}$ (ordered field)(RM)')
    for i in range(len(range1_endpoints)-1):
        plot_rectangle(ax, range1_endpoints[i], 180*po_range1_beck19[i]/np.pi-180*err_po_range1_beck19[i]/np.pi, range1_endpoints[i+1], 180*po_range1_beck19[i]/np.pi+180*err_po_range1_beck19[i]/np.pi, color='gray'   )
        
    
    ax.errorbar(range2_beck19, 180*po_range2_beck19/np.pi, yerr=180*err_po_range2_beck19/np.pi, zorder=2,ms=9, mew=1, capsize=2,linestyle=' ',
                   marker='D', mfc='violet',label=r' $-p_{\mathrm{ord}}$ (Beck et al. 2019: range 2)', mec='k',ecolor='k')#, label=r'Data $p_{o}$ (ordered field)(RM)')
    
    for i in range(len(range2_endpoints)-1):
        if i%2 == 0:
            plot_rectangle(ax, range2_endpoints[i], 180*po_range2_beck19[int(i/2)]/np.pi-180*err_po_range2_beck19[int(i/2)]/np.pi, range2_endpoints[i+1], 180*po_range2_beck19[int(i/2)]/np.pi+180*err_po_range2_beck19[int(i/2)]/np.pi, color='gray'   )

#Borlaff+23
    ax.plot(range_po_borlaff23, po_borlaff23, zorder=2, marker='*',linestyle=' ',
              markersize=13, mfc='b', mec='k', label=r' $-p_{\mathrm{ord}}$ (Borlaff et al. 2023)')

#Surgent+23
    ax.plot(range_po_Surgent23, po_Surgent23, zorder=2, marker='o',linestyle=' ',
              markersize=9, mfc='yellow', mec='k', label=r' $-p_{\mathrm{ord}}$ (Surgent et al. 2023)')

# # Vasanth's results
#     ax.plot(range_po_V, po_V, zorder=2, marker='s',linestyle=' ',
#               markersize=9, mfc='k', mec='k', label=r" $p_{\mathrm{ord}}$ (R. Vasanth pvt. comm.)")

ax.set_xlabel(r'Radius (kpc)', fontsize=fs)
ax.set_ylabel(r'Pitch angle (degrees)', fontsize=fs)


ax.set_ylim(bottom=0)
ax.yaxis.set_ticks(np.arange(0, 100, 10))

# legend specs
axis_pars(ax)
if galaxy_name == 'm51':
    ax.legend(fontsize=lfs, frameon=frameon_param, handlelength=hd, ncol=1, bbox_to_anchor=(1,1),prop={
            'size': leg_textsize, 'family': 'Times New Roman'}, fancybox=True, framealpha=frame_alpha_param, handletextpad=legend_labelspace, columnspacing=0.7)
elif galaxy_name == 'ngc6946':
    ax.legend(fontsize=lfs, frameon=frameon_param, handlelength=hd, ncol=1, bbox_to_anchor=(1,1),prop={
            'size': leg_textsize, 'family': 'Times New Roman'}, fancybox=True, framealpha=frame_alpha_param, handletextpad=legend_labelspace, columnspacing=0.7)
elif galaxy_name == 'm33':
    ax.legend(fontsize=lfs, frameon=frameon_param, handlelength=hd, ncol=1, bbox_to_anchor=(1,1),prop={
            'size': leg_textsize, 'family': 'Times New Roman'}, fancybox=True, framealpha=frame_alpha_param, handletextpad=legend_labelspace, columnspacing=0.7)
else:
    ax.legend(fontsize=lfs, frameon=frameon_param, handlelength=hd, ncol=1, bbox_to_anchor=(1,1),prop={
            'size': leg_textsize, 'family': 'Times New Roman'}, fancybox=True, framealpha=frame_alpha_param, handletextpad=legend_labelspace, columnspacing=0.7)

plt.savefig(save_files_dir+r'\4 pitch angles')

############################# alpha ########################

# alpha plotting
fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5), tight_layout=True)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')

alpham_f   = alphak_f*((1/dkdc_f)-1)
alphasat_f = alphak_f + alpham_f

# plotting model outputs
if galaxy_name == 'm31':
    ax.plot(kpc_r, alphak_f/cm_km,   linewidth=lw, color='b', marker='o',markersize=4,mfc='k',mec='k', label=r' $\mathrm{\alpha_k}$')
    ax.plot(kpc_r, alpham_f/cm_km,   linewidth=lw, color='r', marker='o',markersize=4,mfc='k',mec='k', label=r' $\mathrm{\alpha_m}$')
    ax.plot(kpc_r, alphasat_f/cm_km, linewidth=lw, color='g', marker='o',markersize=4,mfc='k',mec='k', label=r' $\mathrm{\alpha_{\mathrm{sat}}}$')
else:
    ax.plot(kpc_r, alphak_f/cm_km,   linewidth=lw, color='b', marker='o',markersize=4,mfc='k',mec='k')
    ax.plot(kpc_r, alpham_f/cm_km,   linewidth=lw, color='r', marker='o',markersize=4,mfc='k',mec='k')
    ax.plot(kpc_r, alphasat_f/cm_km, linewidth=lw, color='g', marker='o',markersize=4,mfc='k',mec='k')

# plotting errors
try:
  fill_error(ax, kpc_r, alphak_f/cm_km,alphak_err/cm_km, 'blue')
except NameError:
  pass
    
a                  = alpham_f/cm_km
b                  = alphak_f/cm_km
percent_err_alphak = (alphak_err/alphak_f)*100

# manually setting y-axis limits
if switch['incl_moldat'] == 'No':
    if galaxy_name == 'm31':
        ax.yaxis.set_ticks(np.arange(-(round(min(abs(a)),1)+1.2),round(max(b),2)+0.2,0.8))
    elif galaxy_name == 'm33':
        ax.yaxis.set_ticks(np.arange(-0.5,0.7,0.2))
    elif galaxy_name == 'm51':
        if params['\zeta'] == 10:
            ax.yaxis.set_ticks(np.arange(-(round(min(abs(a)),1)+1.5),round(max(b),1)+1.5,0.8))
        else:
            ax.yaxis.set_ticks(np.arange(-(round(min(abs(a)),1)+3),round(max(b),1)+1.5,0.8))
    else:
        ax.yaxis.set_ticks(np.arange(-7,11,2))

ax.axhline(y=0, color='black', linestyle=':', alpha = 1)

ax.set_xlabel(r'Radius (kpc)', fontsize=fs)
ax.set_ylabel(r'$\alpha$ (km s$^{-1}$)', fontsize=fs)

# legend specs
axis_pars(ax)
if galaxy_name == 'm31':
    ax.legend(fontsize=lfs, frameon=frameon_param, handlelength=hd, ncol=1, bbox_to_anchor=(1,1),prop={
            'size': leg_textsize+5, 'family': 'Times New Roman'}, fancybox=True, framealpha=frame_alpha_param, handletextpad=legend_labelspace, columnspacing=0.7)

plt.savefig(save_files_dir+r'\5 alphas')

########################## Krause condition #####################
fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5), tight_layout=True)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')

omega  = Symbol('\Omega')
kalpha = Symbol('K_alpha')
calpha = Symbol('C_alpha')
tau_f  = taue_f

# calculate for plotting
omt = datamaker(omega, data_pass, h_f, tau_f)*tau_f
kah = datamaker(kalpha/calpha, data_pass, h_f, tau_f)*(h_f/(tau_f*u_f))

ax.axhline(y=0, color='k',linestyle=':')
ax.axhline(y=1, color='g', linewidth=lw, alpha=1)

if galaxy_name == 'm31':
    ax.plot(kpc_r, omt, c='r', marker='o',markersize=4, mfc='k',mec='k', linewidth=lw, label=r' $\Omega\tau$')
    ax.plot(kpc_r, kah, c='b', marker='o',markersize=4, mfc='k',mec='k', linewidth=lw, label=r' $\frac{C_\mathrm{\alpha}^{\prime} h}{C_\mathrm{\alpha} \tau u}$')
else:
    ax.plot(kpc_r, omt, c='r', marker='o',markersize=4,mfc='k',mec='k', linewidth=lw)
    ax.plot(kpc_r, kah, c='b', marker='o',markersize=4,mfc='k',mec='k', linewidth=lw)

if galaxy_name == 'ngc6946':
    ax.set_ylim(bottom=0)
    ax.yaxis.set_ticks(np.arange(0,max(kah)+2,1))
elif galaxy_name == 'm31':
    ax.set_ylim(bottom=0)
    ax.yaxis.set_ticks(np.arange(0,max(kah)+0.5,0.5))

# legend specs
axis_pars(ax)
if galaxy_name == 'm31':
    ax.legend(fontsize=lfs, frameon=frameon_param, handlelength=hd, ncol=2, bbox_to_anchor=(0.5,1),prop={
            'size': leg_textsize+5, 'family': 'Times New Roman'}, fancybox=True, framealpha=frame_alpha_param, handletextpad=legend_labelspace, columnspacing=0.7)

ax.set_xlabel('Radius (kpc)', fontsize=fs)
ax.set_ylabel(r'Condition for $\mathrm{\alpha_k}$ (Myr)', fontsize=fs)

plt.savefig(save_files_dir+r'\6 omega')

##################### correlation time ########################

fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5), tight_layout=True)
ax.set_yscale('log')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')

# plot model outputs
if galaxy_name == 'm31':
    ax.plot(kpc_r, taue_f/s_Myr, marker='o',markersize=4,mfc='k',mec='k', linewidth=lw, c='b',label=r' $\tau_\mathrm{e}$')
    ax.plot(kpc_r, taur_f/s_Myr, marker='o',markersize=4,mfc='k',mec='k', linewidth=lw,c='g',label=r' $\tau_\mathrm{r}$')
else:
    ax.plot(kpc_r, taue_f/s_Myr, marker='o',markersize=4,mfc='k',mec='k', linewidth=lw, c='b')
    ax.plot(kpc_r, taur_f/s_Myr, marker='o',markersize=4,mfc='k',mec='k', linewidth=lw,c='g')

# plot the errors
try:
  fill_error(ax, kpc_r, taue_f/s_Myr,tau_err/s_Myr, 'blue')
except NameError:
  pass
try:
  fill_error(ax, kpc_r, taur_f/s_Myr,taur_err/s_Myr, 'green')
except NameError:
  pass

percent_err_taue = (tau_err/taue_f)*100
percent_err_taur = (taur_err/taur_f)*100
t                = h_f/(u_f*s_Myr)

#taue in Myr
taue       = taue_f/s_Myr
taue_err_c = tau_err/s_Myr

#taur in Myr
taur       = taur_f/s_Myr
taur_err_c = taur_err/s_Myr

# other timescales, not plotted
t_hu    = h_f/(u_f*s_Myr)
omt_inv = 1/omt

# legend specs
if galaxy_name == 'm31':
    ax.legend(fontsize=lfs, frameon=frameon_param, handlelength=hd, ncol=2, bbox_to_anchor=(0.5,1),prop={
            'size': leg_textsize+5, 'family': 'Times New Roman'}, fancybox=True, framealpha=frame_alpha_param, handletextpad=legend_labelspace, columnspacing=0.7)

axis_pars(ax)
ax.set_xlabel('Radius (kpc)', fontsize=fs)
ax.set_ylabel(r'Correlation time $\tau$ (Myr)', fontsize=fs)
ax.axhline(y=0, color='k', linestyle=':')

plt.savefig(save_files_dir+r'\7 times')

################## Dynamo number ########################

# plotting dynamo number
fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5), tight_layout=True)
ax.xaxis.set_ticks_position('both')

if galaxy_name == 'm31':
    ax.xaxis.set_ticks(np.arange(6, 17, 1)) #for m31
elif galaxy_name == 'm33':
    ax.xaxis.set_ticks(np.arange(0, 8, 1)) #for m33
elif galaxy_name == 'm51':
    ax.xaxis.set_ticks(np.arange(1, 9, 1)) #for m51
else:
    ax.xaxis.set_ticks(np.arange(1, 19, 2)) #for ngc

# critical dynamo number
Dc = (np.pi**5)/32

# gamma in Gyr^-1
gamma             = ((((np.pi**2)*(tau_f*(u_f**2))/3*(np.sqrt(dkdc_f)-1)/(4*h_f**2))**(-1))/
                        (s_Myr*1e+3))**(-1)
err_gamma         = gamma*(np.sqrt((2*h_err/h_f)**2 + (tau_err/tau_f)**2 + (2*u_err/u_f)**2 + (dkdc_err/(2*Dc*(dkdc_f-np.sqrt(dkdc_f))))**2))
percent_err_gamma = (err_gamma/gamma)*100

# plot gamma 
if galaxy_name == 'm31':
    p1 = ax.plot(kpc_r, gamma , c='b', linewidth=lw,linestyle='-', label=r' $\gamma$')
else:
    p1 = ax.plot(kpc_r, gamma , c='b', linewidth=lw,linestyle='-')

# plot error in gamma
try:
    fill_error(ax, kpc_r, gamma, err_gamma, 'b', 0.5)
except NameError:
    pass

ax.set_ylabel(r'Local growth rate $\gamma$ ($\mathrm{Gyr^{-1}}$)', fontsize=fs)

# starting right side axis for D/Dc
ax2 = ax.twinx()

# plot dynamo number
if galaxy_name == 'm31':
    p2 = ax2.plot(kpc_r, dkdc_f,c='r',linestyle='-',linewidth=lw,label=r' $D/D_\mathrm{c}$')
else:
    p2 = ax2.plot(kpc_r, dkdc_f,c='r',linestyle='-',linewidth=lw)

# plot error in dynamo number
try:
    fill_error(ax2, kpc_r, dkdc_f,dkdc_err, 'r', 0.5)
except NameError:
    pass

axis_pars(ax)
ax2.set_ylabel(r'$D/D_\mathrm{c}$',  fontsize=fs)

# set limit of ax2 
if switch['incl_moldat'] == 'No':
    if galaxy_name == 'm51':
        ax2.set_ylim(bottom = -1, top = max(dkdc_f+dkdc_err)+1)
    else:
        ax2.set_ylim(bottom = 0, top = max(dkdc_f+dkdc_err)+1)
ax.set_xlabel('Radius (kpc)', fontsize=fs)
ax2.axhline(y=1, color='black', linestyle=':', alpha = 1)

percent_error_dkdc = (dkdc_err/dkdc_f)*100

# specs for y axis of gamma
if switch['incl_moldat'] == 'No':
    if galaxy_name == 'm33':
        ax.set_ylim(bottom=-1.5)
        ax.yaxis.set_ticks(np.arange(-1.5,max(gamma+err_gamma)+1,2))
    elif galaxy_name == 'm51':
        ax.set_ylim(bottom=-10)
        ax.yaxis.set_ticks(np.arange(-10,max(gamma+err_gamma)+2,10))
    elif galaxy_name == 'ngc6946':
        ax.set_ylim(bottom=-2)
        ax.yaxis.set_ticks(np.arange(-2,max(gamma+err_gamma)+1,20))
    else:
        ax.set_ylim(bottom=-6)
        ax.yaxis.set_ticks(np.arange(-6,max(gamma+err_gamma)+2,4))

# legend specs
leg  = p1 + p2
labs = [l.get_label() for l in leg]
if galaxy_name == 'm31':
    ax.legend(leg, labs,fontsize=lfs, frameon=frameon_param, handlelength=hd, ncol=1, bbox_to_anchor=(0.8,1),prop={
                'size': leg_textsize+5, 'family': 'Times New Roman'}, fancybox=True, framealpha=frame_alpha_param, handletextpad=legend_labelspace, columnspacing=0.7)

plt.savefig(save_files_dir+r'\8 dynamo number_efolding')

################### e-folding time ###########################################################
# e-folding time in Gyr
e_folding      = 1/gamma
e_folding_err  = np.sqrt((err_gamma/gamma**2)**2)

fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5), tight_layout=True)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')

# plot e-folding time
if galaxy_name == 'm31':
    ax.plot(kpc_r, e_folding, marker='o',markersize=4,mfc='k',mec='k', linewidth=lw, c='b',label=r' $1/\gamma$ (Gyr)')
else:
    ax.plot(kpc_r, e_folding, marker='o',markersize=4,mfc='k',mec='k', linewidth=lw, c='b')

# plot error in e-folding time
try:
    fill_error(ax, kpc_r, e_folding, e_folding_err, 'b', 0.5)
except NameError:
    pass

# legend specs
if galaxy_name == 'm31':
    ax.legend(fontsize=lfs, frameon=frameon_param, handlelength=hd, ncol=2, bbox_to_anchor=(0.5,1),prop={
            'size': leg_textsize+5, 'family': 'Times New Roman'}, fancybox=True, framealpha=frame_alpha_param, handletextpad=legend_labelspace, columnspacing=0.7)

axis_pars(ax)
ax.set_xlabel('Radius (kpc)', fontsize=fs)
ax.set_ylabel(r'Local e-folding time (Gyr)', fontsize=fs)

plt.savefig(save_files_dir+r'\9 e folding')

##############################################################################################33

# saving all relative errors to a csv file
filename = r'\error_output.csv'

Btot_err             = G_scal_Bbartot_err*1e+6
Breg_err             = G_scal_Bbarreg_err*1e+6
Bord_err             = G_scal_Bbarord_err*1e+6
err_po               = 180*po_err/np.pi
err_pb               = 180*pB_err/np.pi
alpha_err_corr_units = (alphak_err/cm_km)

plotted_quant_err    = [kpc_r,h_err_corr_units,l_err_corr_units,err_u,err_cs,err_tot,Btot_err,Breg_err,Bord_err,err_po,err_pb,alpha_err_corr_units,taue_err_c,taur_err_c,dkdc_err,Mach_err,err_gamma]
rel_err_transpose    = list(zip(*plotted_quant_err))
column_names         = ['radius','h_err', 'l_err', 'u_err', 'cs_err', 'sig_err','Btot_err','Breg_err','Bord_err','po_err','pB_err','alphak_err','taue_err','taur_err','dkdc_err','mach_err','gamma_err']

# Writing to the file
# with open(save_files_dir_err+filename, 'w', newline='') as csvfile:
with open(save_files_dir+filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(column_names)
    csvwriter.writerows(rel_err_transpose)

##################################################################################################################
# saving percent errors to a csv file
filename = r'\percent_error_output.csv'

# quants=[h,l,u,cs,sig,Btot,Breg,Bord,pB_deg,po_deg,b,dkdc_f]
percent_err           = [kpc_r,percent_err_h,percent_err_l,percent_err_u,percent_err_cs,percent_err_sig,percent_err_Btot,percent_err_Breg,percent_err_Bord,percent_err_po,percent_err_pB,percent_err_alphak,percent_err_taue,percent_err_taur,percent_error_dkdc,percent_err_Mach,percent_err_gamma]
percent_err_transpose = list(zip(*percent_err))
column_names          = ['radius','h_err', 'l_err','u_err','cs_err','sig_err','Btot_err','Breg_err','Bord_err','po_err','pB_err','alphak_err','taue_err','taur_err','dkdc_err','mach_err','gamma_err']

# Writing to the file
with open(save_files_dir+filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(column_names)
    csvwriter.writerows(percent_err_transpose)  
##################################################################################################################

# saving quantities to a csv file
filename = r'\model_outputs.csv'

pB_deg           = 180*pB/np.pi
po_deg           = 180*po/np.pi
quants           = [kpc_r,h,l,u,cs,sig,Btot,Breg,Bord,po_deg,pB_deg,b,taue,taur,dkdc_f,Mach,gamma,e_folding]
quants_transpose = list(zip(*quants))
column_names     = ['radius','h', 'l','u','cs','sig','Btot','Breg','Bord','po','pB','alphak','taue_e','tau_r','dkdc','mach no','gamma','e_folding']

# Writing to the file
with open(save_files_dir+filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(column_names)
    csvwriter.writerows(quants_transpose) 

##################################################################################################################

filename = r'\magnetic_output.csv'
# find tan inverse pb_f
pb_f             = np.arctan(pb)
# convert to degrees
pb_deg           = 180*pb_f/np.pi
pb_err           = 180*pb_err/np.pi
pB_err           = 180*pB_err/np.pi
po_err           = 180*po_err/np.pi

# convert to microGauss
biso_f             = biso_f*1e+6
biso_err           = biso_err*1e+6
bani_f             = bani_f*1e+6
bani_err           = bani_err*1e+6
Bbar_f             = Bbar_f*1e+6
Bbar_err           = Bbar_err*1e+6
mag_quants         = [kpc_r, biso_f, biso_err, bani_f, bani_err, Bbar_f, Bbar_err, pb_deg, pb_err, po_deg, po_err, pB_deg, pB_err]
mag_column_names   = ['radius','b_iso', 'err_b_iso', 'b_ani', 'err_b_ani', 'B_bar', 'err_B_bar', 'pb', 'err_pb', 'po', 'err_po', 'pB', 'err_pB']

# Writing to the file
with open(save_files_dir+filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(mag_column_names)
    csvwriter.writerows(list(zip(*mag_quants)))
##################################################################################################################
