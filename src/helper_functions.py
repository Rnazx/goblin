import numpy as np
import pandas as pd
from sympy import *
import inspect
from scipy.optimize import curve_fit, fsolve, root
from scipy.integrate import quad, dblquad
import os
import matplotlib.patches as patches

############################################################################################################################
# Defining the Observables
q        = Symbol('q')
omega    = Symbol('\Omega')
sigma    = Symbol('\Sigma')
sigmah2  = Symbol('\Sigma_H_2')
sigmatot = Symbol('Sigma_tot')
sigmasfr = Symbol('Sigma_SFR')
T        = Symbol('T')


# Defining the Constants
calpha   = Symbol('C_alpha')
gamma    = Symbol('gamma')
boltz    = Symbol('k_B')
mu       = Symbol('mu')
mu_prime = Symbol('mu_prime')
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
K        = Symbol('K')
mu_prime = Symbol('Mu')


# Defining the general parameters
u   = Symbol('u')
tau = Symbol('tau')
l   = Symbol('l')
h   = Symbol('h')
cs = Symbol('c_S')

###############################################################################################
base_path   = os.environ.get('MY_PATH')
galaxy_name = os.environ.get('galaxy_name')



g_Msun = 1.989e33  # solar mass in g
cgs_G  = 6.674e-8  # gravitational constant in cgs units
g_mH   = 1.6736e-24  # mass of hydrogen atom in grams
cgs_kB = 1.3807e-16  # boltzmann constant in cgs units

gval, clval, xioval, mstarval, deltaval, e51val, kaval, Gammaval, Rkval = tuple(
    np.genfromtxt(os.path.join(base_path,'inputs','constants.in'), delimiter='=', dtype=np.float64)[:, -1])

const = [(boltz, cgs_kB), (mh, g_mH), (G, cgs_G), (gamma, gval),
         (cl, clval), (xio, xioval), (mstar, mstarval*g_Msun), (delta, deltaval), (E51, e51val), (kalpha, kaval), (Gamma, Gammaval),(Rk, Rkval)]

######################################################################################################################

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

###################################################################################################################################
def power_law(x, a, b):
    return a*np.power(x, b) 


def list_transpose(x):
    array = np.array(x)
    transposed_array = array.T
    return transposed_array.tolist()
######################################################################################################################


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][-1]

###############################################################################################################

# Function which takes the expression and the data to be substituted
def exp_analytical_data(express, data_pass, cs_f = None):
    # Substitute the constants in the given expression
    express = express.subs(const).simplify(force=True)
    # Substitute the data for the observables as well as the parameters for each radii
    if cs_f is None:
        exp = np.array([express.evalf(subs={sigmatot: sigt, sigma: sig, sigmah2: sigh2, sigmasfr: sigsfr, q: qs, omega: oms, zet: zets, T: t,
                   psi: ps, bet: b, calpha: ca, K: k, mu: m, mu_prime:M, A:a}) for sigt, sig, sigh2, qs, oms, sigsfr, t, zets, ps, b, ca, k, m, M, a in data_pass])
    else:
        exp = np.array([express.evalf(subs={sigmatot: sigt, sigma: sig, sigmah2: sigh2, sigmasfr: sigsfr, q: qs, omega: oms, zet: zets, T: t,
                   psi: ps, bet: b, calpha: ca, K: k, mu: m, mu_prime:M, A:a, cs:csf}) for (sigt, sig, sigh2, qs, oms, sigsfr, t, zets, ps, b, ca, k, m, M, a), csf in zip(data_pass,cs_f)])


    return exp
############################################################################################################################


def datamaker(quan, data_pass, h_f, tau_f=None, alphak_f=None,u_f=None,l_f=None,cs_f= None):
    quan_val = exp_analytical_data(quan, data_pass)
    if tau_f is None: tau_f = np.ones(len(h_f))
    if l_f is None: l_f = np.ones(len(h_f))
    if u_f is None: u_f = np.ones(len(h_f))
    if cs_f is None: cs_f = np.ones(len(h_f))

    if alphak_f is None:
        return np.array([np.float64(quan_val[i].evalf(subs={h: hf, tau: tauf,u:uf,l:lf,cs:csf})) for i, (hf, tauf,uf,lf,csf) in enumerate(zip(h_f, tau_f,u_f,l_f,cs_f))])
    else:
        Bbar_in = np.array([quan_val[i].evalf(subs={h: hf, tau: tauf, alphak: alphakf,u:uf,l:lf,cs:csf}) for i, (
            hf, tauf, alphakf,uf,lf,csf) in enumerate(zip(h_f, tau_f, alphak_f, u_f, l_f,cs_f))])
        # print(Bbar_in)
        return np.float64(Bbar_in*(np.float64(Bbar_in*Bbar_in > 0)))


##############################################################################################################################################
# Function which takes the RHS of the expression of h as input (h_val). h_init is the initial guess for the solution in cgs units
def root_finder(h_val, h_init=7e+25):
    #define an empty array
    h_f = []
    for hv in h_val:
        # Function to find the root for
        def func(x): 
            # h is an expression. hv is the RHS of the expression for h for a particular radii
            return np.array(
            [np.float64((h-hv).evalf(subs={h: i})) for i in x])
        # Derivative of the function
        def dfunc(x): 
            return np.array(
            [np.float64(diff((h-hv), h).evalf(subs={h: i})) for i in x])# diff is a symbolic derivative
        # solve for the function using the fsolve routine. First element of this array is the solution
        h_solution = fsolve(func, h_init, fprime=dfunc )
        # append the solution in an array.
        h_f.append(h_solution[0])
    # Convert array to numpy
    h_f = np.array(h_f)
    return h_f


####################################################################################################################################################################################

def pitch_angle_integrator(kpc_r, tanpB_f, tanpb_f,Bbar_f, bani_f, tanpB_err,tanpb_err, Bbar_err, bani_err):
    pB = np.arctan(-tanpB_f)
    pB_err = -tanpB_err/(1+tanpB_f**2)
    pb = np.arctan(tanpb_f)
    pb_err = tanpb_err/(1+tanpb_f**2)

    def pogen(b, B, pb, pB, s):
        return (np.exp(-b**2/(2*s**2))/
                (np.sqrt(2*(np.pi))*s))*(1+(2*B*b*np.cos(pb-pB))/
                            (b**2 + B**2))*np.arctan((B*np.sin(pB) + b*np.sin(pb))/
                                                        ((B*np.cos(pB)) + b*np.cos(pb)))
    brms = np.sqrt(np.average(bani_f**2))

    h = 1e-8
    def dpodbani(b, B, pb, pB, s):
        return (pogen(b, B, pb, pB, s+h)-pogen(b, B, pb, pB, s-h))/(2*h)
    def dpodBbar(b, B, pb, pB, s):
        return (pogen(b, B+h, pb, pB, s)-pogen(b, B-h, pb, pB, s))/(2*h)
    h = 0.01
    def dpodpB(b, B, pb, pB, s):
        return (pogen(b, B, pb, pB+h, s)-pogen(b, B, pb, pB-h, s))/(2*h)
    def dpodpb(b, B, pb, pB, s):
        return (pogen(b, B, pb+h, pB, s)-pogen(b, B, pb-h, pB, s))/(2*h)

    def integrator(fn, interval = 1e+2): #1e+3
        for i in range(len(kpc_r)):
            print(i,Bbar_f[i], pb[i], pB[i], bani_f[i])
        return np.array([quad(fn, -interval, interval, args=(Bbar_f[i], pb[i], pB[i], bani_f[i]),
                points=[-interval*brms, interval*brms])[0] for i in range(len(kpc_r))])
    po = integrator(pogen)

    inte = 1e+3 
    # error in po due to numerical integration- given as the second element of output of quad function
    po_err = np.array([quad(pogen, -inte, inte, args=(Bbar_f[i], pb[i], pB[i], bani_f[i]),
                points=[-inte*brms, inte*brms])[1] for i in range(len(kpc_r))]) 
    
    # error in po due to errors in the input quantities- propagated using the error propagation formula
    # added to po_err due to integration
    po_err += np.sqrt((integrator(dpodbani,inte)*bani_err)**2 +(integrator(dpodBbar,inte)*Bbar_err)**2
                    +(integrator(dpodpB,inte)*pB_err)**2+(integrator(dpodpb,inte)*pb_err)**2)
    
    return pB, po, pb, pB_err, po_err, pb_err

#differentiating the analytical expression
def analytical_pitch_angle_integrator(kpc_r, tanpB_f, tanpb_f,Bbar_f, bani_f, tanpB_err,tanpb_err, Bbar_err, bani_err):
    b   = Symbol('b')
    B   = Symbol('B')
    p_b = Symbol('p_b')
    p_B = Symbol('p_B')
    s   = Symbol('b_a')

    pB     = np.arctan(tanpB_f)
    pB_err = tanpB_err/(1+tanpB_f**2)
    pb     = np.arctan(tanpb_f)
    pb_err = tanpb_err/(1+tanpb_f**2)

    po = (exp(-b**2/(2*s**2))/
                    (sqrt(2*(pi))*s))*(1+(2*B*b*cos(p_b-p_B))/
                                (b**2 + B**2))*atan((B*sin(p_B) + b*sin(p_b))/
                                                            ((B*cos(p_B)) + b*cos(p_b)))

    pogen    = lambdify([b,B, p_b, p_B, s],po)
    dpodbani = lambdify([b,B, p_b, p_B, s],diff(po,s))
    dpodBbar = lambdify([b,B, p_b, p_B, s],diff(po,B))
    dpodpB   = lambdify([b,B, p_b, p_B, s],diff(po,p_B))
    dpodpb   = lambdify([b,B, p_b, p_B, s],diff(po,p_b))

    brms = np.sqrt(np.average(bani_f**2))
    def integrator(fn, interval = 1e+2):
        return np.array([quad(fn, -interval, interval, args=(Bbar_f[i], pb[i], pB[i], bani_f[i]),
                points=[-interval*brms, interval*brms])[0] for i in range(len(kpc_r))])
    po = integrator(pogen)

    inte   = 1e+3
    po_err = np.array([quad(pogen, -inte, inte, args=(Bbar_f[i], pb[i], pB[i], bani_f[i]),
                points=[-inte*brms, inte*brms])[1] for i in range(len(kpc_r))]) 
    po_err += np.sqrt((integrator(dpodbani,inte)*bani_err)**2 +(integrator(dpodBbar,inte)*Bbar_err)**2
                    +(integrator(dpodpB,inte)*pB_err)**2+(integrator(dpodpb,inte)*pb_err)**2)

    return pB, po, pb, pB_err, po_err, pb_err

#from uniform distrubuted p
def new_pitch_angle_integrator(kpc_r, tanpB_f, Bbar_f, bani_f, tanpB_err, Bbar_err, bani_err, taue_f, tau_err, data_pass, rel_err):
    b   = Symbol('b')
    B   = Symbol('B')
    p_b = Symbol('p_b')
    p_B = Symbol('p_B')
    s   = Symbol('b_a')

    #get parameter values
    params = parameter_read(os.path.join(base_path,'inputs','parameter_file.in'))
    switch = parameter_read(os.path.join(base_path,'inputs','switches.in'))

    pB     = np.arctan(tanpB_f)
    pB_err = tanpB_err/(1+tanpB_f**2)
    q_f = np.float64(exp_analytical_data(q, data_pass))
    Om_f = np.float64(exp_analytical_data(omega, data_pass))
    qOmtau_f = np.float64(q_f*Om_f*taue_f)

    #Number of p values between -\pi/2 and \pi/2
    N = 1000

    #Frequency
    p_min = -np.pi/2
    p_max = np.pi/2

    p = np.arange(N) / (N - 1) * (p_max - p_min) + p_min 
    Ns = len(kpc_r)
    pb_dist = np.zeros((Ns,N))
    pb_err_dist = np.zeros((Ns,N))
    mean_pb = np.zeros((Ns))
    std_pb = np.zeros((Ns))
    err_pb = np.zeros((Ns))

    
    delq = q_f*rel_err['rel_err_q'].values
    delomega = Om_f*rel_err['rel_err_omega'].values

    for i in range(Ns):
        pb_dist[i,:] = np.arctan(np.tan(p[:])/(1-np.tan(p[:])*qOmtau_f[i]))
        pb_err_dist[i, :] = pb_dist[i,:]*(1/(1+np.tan(pb_dist[i,:])**2))*(1/(1-qOmtau_f[i]*np.tan(p[:]))**2)*(np.tan(p[:])**2)*np.sqrt((Om_f[i]*taue_f[i]*delq[i])**2+(q_f[i]*taue_f[i]*delomega[i])**2+(Om_f[i]*q_f[i]*tau_err[i])**2)
        mean_pb[i] = np.mean(pb_dist[i,:])
        std_pb[i] = np.std( pb_dist[i,:] )
        err_pb[i] = np.mean(pb_err_dist[i,:])

    pb     = np.array(mean_pb)


    pb_err = std_pb
    
    Nb = 1000
    po_dist = np.zeros((Ns,N))
    mean_po = np.zeros((Ns))
    std_po = np.zeros((Ns))
    Sum = np.zeros((Ns,N))
    
    #Calculate po from the version of the formula used in the paper
    for i in range(Ns):
        dbtilde = bani_f[i] / Nb * 10
        btilde = ( np.arange(Nb + 1) - Nb / 2 ) / Nb * bani_f[i] * 10 
        for j in range(Nb): 
            Sum[i,:] += np.exp( - btilde[j]**2 / 2 / bani_f[i]**2 ) * ( 1 + 2 * Bbar_f[i] * btilde[j] / ( Bbar_f[i]**2 + btilde[j]**2 ) * np.cos( pb_dist[i,:] - pB[i] ) ) \
        * np.arctan( ( Bbar_f[i] * np.sin( pB[i]) + btilde[j] * np.sin( pb_dist[i,:] ) ) / ( Bbar_f[i] * np.cos( pB[i] ) + btilde[j] * np.cos( pb_dist[i,:] ) ) ) * dbtilde 

        po_dist[i,:] = 1 / ( 2 * np.pi )**0.5 / bani_f[i] * Sum[i,:]

        mean_po[i] = np.mean(po_dist[i,:])
        #std_po[i] = np.sqrt( np.mean( ( po[i,:] - mean_po[i] )**2 ) )
        std_po[i] = np.std( po_dist[i,:] )

    po_new = np.array(mean_po)
    po_new_err = np.array(std_po)
    return pB, po_new, pb, pB_err, po_new_err, err_pb # pb_err

# plot rectangle given four coordinates
def plot_rectangle(ax, x1, y1, x2, y2, color):
    ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, fill=True, color=color, alpha=0.4))

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

def fill_error(ax, x, quan_f, quan_err, color = 'red', alpha = 0.2, error_exists = True):

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
        ax.fill_between(x, yo1, yo2, alpha=alpha, facecolor=color, where = None, interpolate=True, edgecolor=None)  
    else:
        return