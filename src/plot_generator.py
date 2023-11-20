import matplotlib
from helper_functions import datamaker, parameter_read
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import pickle
import os
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import griddata
import sys

# Defining the Observables
q = Symbol('q')
omega = Symbol('\Omega')
sigma = Symbol('\Sigma')
sigmatot = Symbol('Sigma_tot')
sigmasfr = Symbol('Sigma_SFR')
T = Symbol('T')


# Defining the Constants
gamma = Symbol('gamma')
boltz = Symbol('k_B')
mu = Symbol('mu')
mh = Symbol('m_H')


# Defining the general parameters
u = Symbol('u')
tau = Symbol('tau')
l = Symbol('l')
h = Symbol('h')
##############################################################################################################################
#reading the parameters
base_path = os.environ.get('MY_PATH')
galaxy_name = os.environ.get('galaxy_name')

params = parameter_read(os.path.join(base_path,'inputs','parameter_file.in'))
sys.path.append(os.path.join(base_path,'data','supplementary_data', galaxy_name))
from observables import *

########################################################################################################################
# subprocess.run(["python", "zipped_data.py"])
# subprocess.run(["python", "get_magnetic_observables.py"])

# conversion factors
pc_kpc = 1e3  # number of pc in one kpc
cm_km = 1e5  # number of cm in one km
cm_kpc = 3.086e+21  # number of centimeters in one parsec
s_Myr = 1e+6*(365*24*60*60)  # megayears to seconds
deg_rad = 180e0/np.pi
arcmin_deg = 60e0
arcsec_deg = 3600e0

########################################################################################################
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

os.chdir(os.path.join(base_path,'inputs'))

with open('zip_data.in', 'rb') as f:
    kpc_r, data_pass = pickle.load(f)

#######################################################################################################################################

dat_u = griddata(kpc_radius, np.sqrt(3)*kms_sigmaLOS, kpc_r, method='linear',
                 fill_value=nan, rescale=False)*1e+5
try:
    dat_u_warp = griddata(kpc_radius, np.sqrt(3)*kms_sigmaLOS_warp, kpc_r, method='linear',
                    fill_value=nan, rescale=False)*1e+5
except NameError:
    pass
os.chdir(current_directory)

from helper_functions import pitch_angle_integrator

pB, po, pb, pB_err, po_err, pb_err = pitch_angle_integrator(kpc_r, tanpB_f,tanpb_f, \
                                   Bbar_f, bani_f, tanpB_err,tanpb_err, Bbar_err, bani_err)

G_scal_Bbartot = np.sqrt(biso_f**2 + bani_f**2 + Bbar_f**2)
G_scal_Bbarreg = Bbar_f
G_scal_Bbarord = np.sqrt(bani_f**2 + Bbar_f**2)


G_scal_Bbartot_err = np.sqrt((biso_err*biso_f )**2+ (bani_err*bani_f)**2 + (Bbar_err*Bbar_f)**2)/G_scal_Bbartot
G_scal_Bbarreg_err = Bbar_err
G_scal_Bbarord_err = np.sqrt((bani_err*bani_f)**2 + (Bbar_err*Bbar_f)**2)/G_scal_Bbarord

m = 2
dm = 2.5
fs = 15
lfs = 10
leg_textsize = 10
axis_textsize = 10
rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
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
    #ax.xaxis.set_ticks(np.arange(6, 20, 2))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.tick_params(axis='both', which='minor',
                   labelsize=axis_textsize, colors='k', length=3, width=1)
    ax.tick_params(axis='both', which='major',
                   labelsize=axis_textsize, colors='k', length=5, width=1.25)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_linewidth(2)
    # ax.spines['left'].set_linewidth(2)
    ax.legend(fontsize=lfs, frameon=False, handlelength=4, ncol=1, prop={
            'size': leg_textsize, 'family': 'Times New Roman'}, fancybox=True, framealpha=0.9, handletextpad=0.7, columnspacing=0.7)
    
def fill_error(ax, quan_f, quan_err, color = 'red', alpha = 0.2, error_exists = True):
    if error_exists:
        ax.fill_between(kpc_r, (quan_f+quan_err), (quan_f-quan_err)
                        , alpha=alpha, edgecolor='k', facecolor=color, where = None, interpolate=True)
    else:
        return


os.chdir(os.path.join(base_path,'plots'))

from matplotlib.backends.backend_pdf import PdfPages
PDF = PdfPages(f'{galaxy_name}_ca_'+str(params[r'C_\alpha'])+'K_'+str(params[r'K'])+'z_'+str(
      params[r'\zeta'])+'psi_'+str(params[r'\psi'])+'b_'+str(params[r'\beta'])+'.pdf')#('plots_model'+str(model_no)+let+'t_vary_'+'ca_'+str(ca)+'rk_'+str(rk)+'z_'+str(z.mean())+'.pdf')

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(7, 10), tight_layout=True)
fig.suptitle(r'$C_\alpha$ = '+str(params[r'C_\alpha'])+r'    $K$ = '+str(params[r'K'])+
             r'    $\zeta$ = '+str(params[r'\zeta'])+r'    $\psi$ = '+str(params[r'\psi'])+r'    $\beta$ = '+str(params[r'\beta']), weight = 15)
i = 0
ax[i].plot(kpc_r, h_f*pc_kpc/cm_kpc, c='r', linestyle='-', mfc='k',
              mec='k', markersize=m, marker='o', label=r' $h$(pc)')
try:
    ax[i].plot(kpc_dat_r, pc_dat_h, c='b', linestyle='dotted', 
                marker='*',mfc='y',mec='b',mew=1, markersize = 7, label=r'Fiducial values from Chamandy et.al.(2016) $h(pc)$')
except NameError:
    pass
ax[i].plot(kpc_r, l_f*pc_kpc/cm_kpc, c='g',
              linestyle='-', mfc='k', mec='k', markersize=m, marker='o', label=r'Correlation length l(pc)')
# ax[i].plot(kpc_r, datamaker(lsn , data_pass, h_f, tau_f)*pc_kpc/cm_kpc,c = 'y',linestyle='--',mfc='k',mec='k', marker='o')
ax[i].axhline(y=100, color='black', linestyle='--', alpha = 0.2)
#ax[i].set_yticks(list(plt.yticks()[0])+[100])
axis_pars(ax[i])
try:
    fill_error(ax[i], h_f*pc_kpc/cm_kpc, h_err*pc_kpc/cm_kpc)
except NameError:
    pass
    
#ax[i].set_xlabel(r'Radius (kpc)', fontsize=fs)
ax[i].set_ylabel(r'Length scale (pc)', fontsize=fs)


i = 1
ax[i].plot(kpc_r, u_f/cm_km, color='tab:orange', marker='o', mfc='k',
              mec='k', markersize=m, label=r'$u$')
try:
    fill_error(ax[i], u_f/cm_km,u_err/cm_km, 'tab:orange', 0.5)
except NameError:
    pass
    

# ax[i].plot(kpc_r, alphak_f/cm_km, color='b', marker='o',
#               mfc='k', mec='k', markersize=m, label=r'$\alpha_k$')
# try:
#   fill_error(ax[i], alphak_f/cm_km,alphak_err/cm_km, 'blue')
# except NameError:
#   pass
    # 

# ax[i].plot(kpc_r, alpham_f/cm_km, color='r', marker='o',
#               mfc='k', mec='k', markersize=m, label=r'$\alpha_m$')
# ax[i].plot(kpc_r, alphasat_f/cm_km, color='m', marker='o',
#               mfc='k', mec='k', markersize=m, label=r'$\alpha_{sat}$')

sig = np.sqrt(u_f**2 + cs_f**2) 
ax[i].plot(kpc_r, sig /
              cm_km, color='r', marker='o', mfc='k', mec='k', markersize=m, label=r'$\sqrt{u^2+c_s^2}$')
try:
    fill_error(ax[i], sig /cm_km,np.sqrt((u_f*u_err)**2 + (cs_f*cs_err)**2)/(sig*cm_km), 'r')
except NameError:
    pass
    

ax[i].plot(kpc_r, cs_f /
              cm_km, color='g', linestyle='--', label=r'$c_s$', alpha = 0.5)
try:
    fill_error(ax[i], cs_f/cm_km,cs_err/cm_km, 'green')
except NameError:
    pass
    
try:
    ax[i].plot(kpc_r, dat_u/cm_km, 
                c='y', linestyle='--', label='vel disp',alpha = 1,marker='*',mfc='y'
    ,mec='k',mew=1, markersize = 7)
    ax[i].plot(kpc_r, dat_u_warp/cm_km, 
                c='tab:cyan',  linestyle='dashdot', label='With warp', alpha = 0.3,marker='*',mfc='tab:cyan'
    ,mec='k',mew=1, markersize = 7)
except NameError:
    pass

axis_pars(ax[i])

ax[i].set_ylim(0)
#ax[i].set_xlabel(r'Radius ($kpc$)', fontsize=fs)
ax[i].set_ylabel(r'Speed (km/s)',  fontsize=fs)

###############################################################################################################################################################
PDF.savefig(fig)
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(7, 10), tight_layout=True)

i = 1
try:
    ax[i].errorbar(rmrange, 180*RM_dat_pb/np.pi,elinewidth=1, yerr=180*err_RM_dat_pb/np.pi,ecolor='k', ms=7, mew=1, capsize=2,
                    c='r', linestyle='--', mfc='r', mec='k',barsabove=True,marker='*')#, label=r'Data $p_{B}$ (mean field)(RM)')
    ax[i].errorbar(rmrange, 180*RM_dat_po/np.pi,elinewidth=1, yerr=180*err_RM_dat_po/np.pi, ms=7, mew=1, capsize=2,
                    c='g', linestyle='--', marker='*', mfc='g', mec='k',ecolor='k')#, label=r'Data $p_{o}$ (ordered field)(RM)')
except NameError:
    pass
ax[i].plot(kpc_r, 180*pB/np.pi, c='r', linestyle='-', marker='o',
              markersize=m, mfc='k', mec='k', label=r' $p_{B}$ (mean field)')
try:
    fill_error(ax[i], 180*pB/np.pi,180*pB_err/np.pi, 'r')
except NameError:
    pass
    
# ax[i].plot(kpc_r, 180*pbb/np.pi,c = 'r',linestyle='--', marker='o',label = r' $p_{b}$ (anisotropic field)')
ax[i].plot(kpc_r, 180*po/np.pi, c='g', linestyle='-', mfc='k', markersize=m,
        
              mec='k', marker='o', label=r' $p_{o}$ (ordered field)')
try:
    fill_error(ax[i], 180*po/np.pi,180*po_err/np.pi, 'g')
except NameError:
    pass
    

axis_pars(ax[i])
ax[i].set_xlabel(r'Radius ($kpc$)', fontsize=fs)
ax[i].set_ylabel(r'Pitch angle (deg)', fontsize=fs)

i = 0
try:
    ax[i].plot(mrange, G_dat_Btot, c='b', linestyle='--', marker='*',mfc='yellow',mec
    ='tab:blue',mew=1,markersize = 7)#, label='Average Binned data $B_{tot}$ ($\mu G$)')
    ax[i].plot(mrange, G_dat_Breg, c='r', linestyle='--', marker='*',mfc='yellow',mec
    ='tab:red',mew=1,markersize = 7)#, label='Average Binned data $B_{reg}$ ($\mu G$)')
    ax[i].plot(mrange, G_dat_Bord, c='g', linestyle='dotted', marker='*',mfc='yellow'
    ,mec='green',mew=1,markersize = 7)#, label='Average Binned data $B_{ord}$ ($\mu G$)')
except NameError:
    pass
ax[i].plot(kpc_r, G_scal_Bbartot*1e+6, c='b', linestyle='-', marker='o', mfc='k', mec='k',
              markersize=m, label=r' $B_{tot}=\sqrt{\bar{B}^2+b_{iso}^2+b_{ani}^2}$')
try:
    fill_error(ax[i], G_scal_Bbartot*1e+6,G_scal_Bbartot_err*1e+6, 'b')
    fill_error(ax[i], G_scal_Bbarreg*1e+6,G_scal_Bbarreg_err*1e+6, 'r', 0.2)
    fill_error(ax[i], G_scal_Bbarord*1e+6,G_scal_Bbarord_err*1e+6, 'g', 0.2)
except NameError:
    pass

ax[i].plot(kpc_r, G_scal_Bbarreg*1e+6, c='r', linestyle='-', marker='o',
              mfc='k', mec='k', markersize=m, label=r' $B_{reg} = \bar{B}$')
ax[i].plot(kpc_r, G_scal_Bbarord*1e+6, c='green', linestyle='-', marker='o', mfc='k',
              mec='k', markersize=m, label=r' $B_{ord} = \sqrt{\bar{B}^2+b_{ani}^2}$')

ax[i].set_xlabel(r'Radius ($kpc$)', fontsize=fs)
ax[i].xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax[i].set_ylabel('Magnetic field strength ($\mu G$)', fontsize=fs)
axis_pars(ax[i])

PDF.savefig(fig)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), tight_layout=True)

i = 0
j = 0
omega = Symbol('\Omega')
kalpha = Symbol('K_alpha')
calpha = Symbol('C_alpha')
tau_f = taue_f
omt = datamaker(omega, data_pass, h_f, tau_f)*tau_f
kah = datamaker(kalpha/calpha, data_pass, h_f, tau_f)*(h_f/(tau_f*u_f))

ax[i][j].axhline(y=1, color='g', linestyle='-', label=r'1')
ax[i][j].plot(kpc_r, omt, marker='o', markersize=m,
              c='tab:orange', mfc='k', mec='k', label=r'$\Omega\tau$')
ax[i][j].plot(kpc_r, kah, marker='o',
              markersize=m, c='tab:blue', mfc='k', mec='k', label=r'$\frac{K_\alpha h}{C_\alpha \tau u}$')
axis_pars(ax[i][j])
ax[i][j].set_xlabel('Radius(kpc)', fontsize=fs)
ax[i][j].set_ylabel(r'Condition for $\alpha_k$ (Myr)', fontsize=fs)


j = 1
ax[i][j].plot(kpc_r, taue_f/s_Myr, c='b', markersize=m,
              linestyle='-', marker='o', mfc='k', mec='k', label=r'$\tau^e$')
ax[i][j].plot(kpc_r, taur_f/s_Myr, c='g',
              markersize=m, linestyle='-', marker='o', mfc='k', mec='k', label=r'$\tau^r$')
# ax[i].plot(kpc_r, h_f/(u_f*s_Myr), c='y', markersize=m,
#               linestyle='-', marker='o', mfc='k', mec='k', label=r'$h/u$')
ax[i][j].set_xlabel('Radius(kpc)', fontsize=fs)
ax[i][j].set_ylabel(r'Correlation Time $\tau$ (Myr)', fontsize=fs)
axis_pars(ax[i][j])

i = 1
ax[i][j].plot(kpc_r, dkdc_f, markersize=m, linestyle='-',
              marker='o', mfc='k', mec='k', label=r'$D_k/D_c$')
ax[i][j].plot(kpc_r, 1*np.ones(len(kpc_r)))
ax[i][j].set_xlabel('Radius(kpc)', fontsize=fs)
ax[i][j].set_ylabel(r'$D_k/D_c$',  fontsize=fs)
axis_pars(ax[i][j])
try:
    fill_error(ax[i][j], dkdc_f,dkdc_err, 'tab:orange', 0.5)
except NameError:
    pass

j = 0
ax[i][j].plot(kpc_r, (((np.pi**2)*(tau_f*(u_f**2))/3*(np.sqrt(dkdc_f)-1)/(4*h_f**2))**(-1))/
              (s_Myr*1e+3), c='g', markersize=m, linestyle='-', marker='o', mfc='k', mec='k', label=r'$\gamma$')
ax[i][j].set_xlabel('Radius(kpc)', fontsize=fs)
ax[i][j].set_ylabel(r'local e-folding time $\gamma$ (Gyr)', fontsize=fs)
ax[i][j].legend(fontsize = lfs)

PDF.savefig(fig)
PDF.close()
os.chdir(current_directory)
