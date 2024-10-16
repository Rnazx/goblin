# This file calculates the expressions for the turbulence model

print('#####  Turbulence expression calculation #####')

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from fractions import Fraction
import pickle
import sys
import os

pc_kpc     = 1e3                  # number of pc in one kpc
cm_km      = 1e5                  # number of cm in one km
s_day      = 24*3600              # number of seconds in one day
s_min      = 60                   # number of seconds in one hour
s_hr       = 3600                 # number of seconds in one hour
cm_Rsun    = 6.957e10             # solar radius in cm
g_Msun     = 1.989e33             # solar mass in g
cgs_G      = 6.674e-8             # Gravitational constant in cgs units
cms_c      = 2.998e10             # speed of light in cm/s
g_mH       = 1.6736e-24           # mass of hydrogen atom in grams
g_me       = 9.10938e-28          # mass of electron in grams
cgs_h      = 6.626e-27            # planck constant in cgs units
deg_rad    = 180e0/np.pi          # radians to degree conversion
arcmin_deg = 60e0                 # degrees to arcmin
arcsec_deg = 3600e0               # degrees to arcsec
cm_kpc     = 3.086e+21            # number of centimeters in one parsec
s_Myr      = 1e+6*(365*24*60*60)  # megayears to seconds


################## defining symbols ####################
# Defining the Observables
q        = Symbol('q')
omega    = Symbol('\Omega')
sigma    = Symbol('\Sigma') # works as diffuse gas density when molecular data is included; total gas density when not included
sigmah2  = Symbol('\Sigma_H_2') # to be used only when molecular data is included
sigmatot = Symbol('Sigma_tot')
sigmasfr = Symbol('Sigma_SFR')
T        = Symbol('T')


# Defining the Constants
calpha   = Symbol('C_alpha')
gamma    = Symbol('gamma')
boltz    = Symbol('k_B')
mu       = Symbol('mu')
mh       = Symbol('m_H')
G        = Symbol('G')
xio      = Symbol('xi_0')
delta    = Symbol('\delta')
mstar    = Symbol('m_*')
cl       = Symbol('C_l')
kappa    = Symbol('kappa')
mach     = Symbol('M')
E51      = Symbol('E_51')
Rk       = Symbol('R_k')
zet      = Symbol('zeta')
psi      = Symbol('psi')
kalpha   = Symbol('K_alpha')
bet      = Symbol('beta')
alphak   = Symbol('alpha_k')
Gamma    = Symbol('Gamma')
A        = Symbol('A')
mu_prime = Symbol('Mu')

# Defining the general parameters
u   = Symbol('u')
tau = Symbol('tau')
l   = Symbol('l')
h   = Symbol('h')
cs  = Symbol('c_S')

cs_exp = (gamma*boltz*T/(mu*mh))**Rational(1/2)

galaxy_name       = os.environ.get('galaxy_name')
current_directory = str(os.getcwd())
base_path         = os.environ.get('MY_PATH')

def parameter_read(filepath):
#opening these files and making them into dictionaries
    params = {}
    with open(filepath, 'r') as FH:
        for file in FH.readlines():
            line = file.strip()
            try:
                par_name, value = line.split('= ')
            except ValueError:
                print("Record: ", line)
                raise Exception(
                    "Failed while unpacking. Not enough arguments to supply.")
            try:
                params[par_name] = np.float64(value)
            except ValueError: #required cz of 14/11 in parameter.in file
                try:
                    num, denom = value.split('/')
                    params[par_name] = np.float64(num) / np.float64(denom)
                except ValueError:
                    params[par_name] = value
            
    return params
switch = parameter_read(os.path.join(base_path,'inputs','switches.in'))

##############################################################################################################

# Defining the expressions
rho = sigma/(2*h)
n   = rho/(mu*mh)
nu  = (delta*sigmasfr)/(2*h*mstar)

if switch['incl_moldat'] == 'No':
    sigma_gas = sigma
    n = sigma/((2*h)*(mu)*mh)
else:
    sigma_gas = sigma+sigmah2
    n = ((sigma/(mu)) + (sigmah2/(mu_prime)))/((2*h)*mh)

#defining a seperate expression for h only in terms of u
h_vdisp = (u**2 + (A*cs)**2)/(3*pi*G*(sigma_gas + (sigmatot/zet)))

nos = 3 #nos is the model number- 1, 2 or 3
if nos == 1:
    lsn = psi*cl*h #lsn= driving scale of isolated SNe, psi=fixed parameter used since u isnt same as velocity dispersion
    l   = lsn
    u   = cs
elif nos == 2:
    lsn = psi*cl*h #lsn= driving scale of isolated SNe, psi=fixed parameter used since u isnt same as velocity dispersion
    l   = lsn
    u   = simplify(((4*pi/3)*l*lsn**3*cs**2*nu)**Fraction(1, 3))
#include superbubbles: full expression for l
elif nos == 3:
    lsn = psi*0.14*cm_kpc*(E51)**Fraction(16, 51)*(n/0.1)**Fraction(-19, 51)*(cs/(cm_km*10))**Fraction(-1, 3)
    #Eqn 29 Chamandy and Sukurov (2020)
    l = ((Gamma-1)/Gamma)*cl*lsn
    #Eqn 33 Chamandy and Sukurov (2020)
    u = simplify(((4*pi/3)*l*lsn**3*cs**2*nu)**Fraction(1, 3))
else:
    print('enter 1, 2 or 3 as model number')
l = simplify(l)

hg   = (u**2 + (A*cs)**2)/(3*pi*G*(sigma_gas + (sigmatot/zet)))
# hg = (u**2)/(3*pi*G*(sigma + (sigmatot/zet))) # trial for NGC 6946
hsub = ((A*cs)**2)/(3*pi*G*(sigma_gas + (sigmatot/zet)))
hsup = (u**2)/(3*pi*G*(sigma_gas + (sigmatot/zet)))

taue = simplify(l/u)
taur = simplify(6.8*s_Myr*(1/4)*(nu*cm_kpc**3*s_Myr/50)**(-1)*(E51)
                ** Fraction(-16, 17) * (n/0.1)**Fraction(19, 17)*(cs/(cm_km*10)))

alphak1 = calpha*tau**2*u**2*omega/h
alphak2 = calpha*tau*u**2/h
alphak3 = kalpha*u

turb_expr = hg, h_vdisp, rho, nu, u, l, taue, taur, alphak1, alphak2, alphak3

with open('turb_exp.pickle', 'wb') as f:
    pickle.dump(turb_expr, f)

print('Solved the turbulence expressions')
