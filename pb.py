import math as ma
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
#from astropy import units as u
#from astropy.constants import h,c,k_B

#Choose radians or degrees
Use_degrees = True
#Use_degrees = False

#Specify mean and random anisotropic magnetic field strengths, where bani = \sqrt{B^2 - Bbar^2}
mkG_bani = 5
mkG_Bbar = 2.5
deg_pB = - 10

mkG_B = ma.sqrt(mkG_bani**2 + mkG_Bbar**2)

bani = mkG_bani
Bbar = mkG_Bbar
B = mkG_B
pB = deg_pB * ma.pi / 180

#Specify figure size
length, breadth = [5, 2.5]

#Specify text size
leg_textsize = 6
axis_textsize = 8

#Specify range of p
if (Use_degrees):
    xr=[-90,90]
    yr=[-90,90]
else:
    xr=[-ma.pi/2,ma.pi/2]
    yr=[-ma.pi/2,ma.pi/2]

#Number of p values between -\pi/2 and \pi/2
N = 1000

#Number of q\Omega\tau values to plot
Ns = 6

#Frequency
p_min = -ma.pi/2
p_max = ma.pi/2
deg_p_min = p_min * 180 / ma.pi
deg_p_max = p_max * 180 / ma.pi

p = np.arange(N) / (N - 1) * (p_max - p_min) + p_min 
deg_p = p * 180 / np.pi
#print(p)

#q\Omega\tau
qOmtau = (np.arange(Ns))*0.2
#qOmtau = [0.1,0.2,0.3,0.4,0.5]
print('q*Om*tau',qOmtau)

#Planck
pb = np.zeros((Ns,N))
mean_pb = np.zeros((Ns))
deg_mean_pb = np.zeros((Ns))
std_pb = np.zeros((Ns))
deg_std_pb = np.zeros((Ns))
po_simp = np.zeros((Ns,N))
mean_po_simp = np.zeros((Ns))
std_po_simp = np.zeros((Ns))
deg_std_po_simp = np.zeros((Ns))
deg_mean_po_simp = np.zeros((Ns))
po = np.zeros((Ns,N))
mean_po = np.zeros((Ns))
deg_mean_po = np.zeros((Ns))
std_po = np.zeros((Ns))
deg_std_po = np.zeros((Ns))
Sum = np.zeros((Ns,N))

#Calculate pb, its mean value, and its standard deviation
for i in range(Ns):
    pb[i,:] = np.arctan(np.tan(p[:])/(1-np.tan(p[:])*qOmtau[i]))

    mean_pb[i] = np.mean(pb[i,:])
    deg_mean_pb[i] = mean_pb[i] * 180 / np.pi
    #std_pb[i] = np.sqrt( np.mean( ( pb[i,:] - mean_pb[i] )**2 ) )
    std_pb[i] = np.std( pb[i,:] )
    deg_std_pb[i] = std_pb[i] * 180 / np.pi
    
    print("i = %1d,   qOmtau = %3.1f,   mean_pb = %5.2f,   deg_mean_pb = %4.1f,   std_pb = %5.2f,   deg_std_pb = %4.1f" % ( i,qOmtau[i],mean_pb[i],deg_mean_pb[i],std_pb[i],deg_std_pb[i] ) )

print()

#Calculate po from the simple version of the formula derived in the paper
for i in range(Ns):
    po_simp[i,:] = 0.5 * ( \
    ( 1 + 2 * Bbar * bani / B**2 * np.cos( pb[i,:] - pB ) ) \
    * np.arctan( ( Bbar * ma.sin( pB ) + bani * np.sin( pb[i,:] ) ) / ( Bbar * ma.cos( pB ) + bani * np.cos( pb[i,:] ) ) ) \
    + ( 1 - 2 * Bbar * bani / B**2 * np.cos( pb[i,:] - pB ) ) \
    * np.arctan( ( Bbar * ma.sin( pB ) + bani - np.sin( pb[i,:] ) ) / ( Bbar * ma.cos( pB ) - bani * np.cos( pb[i,:] ) ) ) )

    mean_po_simp[i] = np.mean(po_simp[i,:])
    deg_mean_po_simp[i] = mean_po_simp[i] * 180 / np.pi
    #std_po_simp[i] = np.sqrt( np.mean( ( po_simp[i,:] - mean_po_simp[i] )**2 ) )
    std_po_simp[i] = np.std( po_simp[i,:] )
    deg_std_po_simp[i] = std_po_simp[i] * 180 / np.pi

    print("i = %1d,   qOmtau = %3.1f,   mean_po_simp = %5.2f,   deg_mean_po_simp = %4.1f,   std_po_simp = %5.2f,   deg_std_po_simp = %4.1f" % ( i,qOmtau[i],mean_po_simp[i],deg_mean_po_simp[i],std_po_simp[i],deg_std_po_simp[i] ) )

print()

Nb = 1000
dbtilde = bani / Nb * 10
btilde = ( np.arange(Nb + 1) - Nb / 2 ) / Nb * bani * 10 
print('Nb',Nb)
print('dbtilde',dbtilde)
print('btilde',btilde)
print()

#Calculate po from the version of the formula used in the paper
for i in range(Ns):
    for j in range(Nb): 
        Sum[i,:] += ma.exp( - btilde[j]**2 / 2 / bani**2 ) * ( 1 + 2 * Bbar * btilde[j] / ( Bbar**2 + btilde[j]**2 ) * np.cos( pb[i,:] - pB ) ) \
	* np.arctan( ( Bbar * ma.sin( pB ) + btilde[j] * np.sin( pb[i,:] ) ) / ( Bbar * ma.cos( pB ) + btilde[j] * np.cos( pb[i,:] ) ) ) * dbtilde 

    po[i,:] = 1 / ( 2 * ma.pi )**0.5 / bani * Sum[i,:]

    mean_po[i] = np.mean(po[i,:])
    deg_mean_po[i] = mean_po[i] * 180 / np.pi
    #std_po[i] = np.sqrt( np.mean( ( po[i,:] - mean_po[i] )**2 ) )
    std_po[i] = np.std( po[i,:] )
    deg_std_po[i] = std_po[i] * 180 / np.pi

    print("i = %1d,   qOmtau = %3.1f,   mean_po = %5.2f,   deg_mean_po = %4.1f,   std_po = %5.2f,   deg_std_po = %4.1f" % ( i,qOmtau[i],mean_po[i],deg_mean_po[i],std_po[i],deg_std_po[i] ) )

print()

deg_pb = pb * 180 / np.pi
deg_mean_pb = mean_pb * 180 / np.pi
deg_po = po * 180 / np.pi
deg_mean_po = mean_po * 180 / np.pi

###########

fig, ax1 = plt.subplots(figsize=(length, breadth))
ax1.set_xlim(xr)
ax1.set_ylim(yr)

col_list = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']

if Use_degrees: 
    p = deg_p
    pb = deg_pb
    mean_pb = deg_mean_pb
    p_min = deg_p_min
    p_max = deg_p_max
    std_pb = deg_std_pb

for i in range(Ns):
    plt.plot(p,pb[i,:],label=r"$q\Omega\tau={:.1f}$".format(qOmtau[i]),color=col_list[i])

for i in range(Ns):
    plt.plot([p_min,p_max],[mean_pb[i],mean_pb[i]],label=r"$\langle p_b \rangle = {:.1f}$".format(mean_pb[i]),linestyle='dashed',color=col_list[i])

for i in range(Ns):
    plt.plot([p_min,p_max],[std_pb[i],std_pb[i]],label=r"$\sigma_{{p_{{b}}}} = {:.1f}$".format(std_pb[i]),linestyle='dotted',color=col_list[i])

leg= ax1.legend(loc='upper left',handlelength=2,ncol=3,prop={'size':0.9*leg_textsize,'family':'Times New Roman'},fancybox=False,framealpha=1,handletextpad=0.4,columnspacing=0.6)

ax1.tick_params(axis='both', which='minor', labelsize=axis_textsize, colors='k', length=3 , width=1   )
ax1.tick_params(axis='both', which='major', labelsize=axis_textsize, colors='k', length=5 , width=1.25)

if Use_degrees:
    xlabel = r'$p$ [degrees]'
    ylabel = r'$p_b$ [degrees]'
else:
    xlabel = r'$p$ [radians]'
    ylabel = r'$p_b$ [radians]'

ax1.set_xlabel(xlabel, size=axis_textsize)
ax1.set_xticklabels(ax1.get_xticks(),fontname = "Times New Roman", fontsize=axis_textsize)
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%g'))	
    
ax1.set_ylabel(ylabel, size=axis_textsize)
ax1.set_yticklabels(ax1.get_yticks(),fontname = "Times New Roman", fontsize=axis_textsize)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%g'))	

# plt.savefig('pb.png', format='png', bbox_inches='tight', dpi= 300)
plt.show()
###########

fig, ax2 = plt.subplots(figsize=(length, breadth))
ax2.set_xlim(xr)
ax2.set_ylim(yr)

col_list = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']

if Use_degrees: 
    p = deg_p
    po = deg_po
    mean_po = deg_mean_po
    p_min = deg_p_min
    p_max = deg_p_max
    std_po = deg_std_po

for i in range(Ns):
    plt.plot(p,po[i,:],label=r"$q\Omega\tau={:.1f}$".format(qOmtau[i]),color=col_list[i])

for i in range(Ns):
    plt.plot([p_min,p_max],[mean_po[i],mean_po[i]],label=r"$\langle p_\mathrm{{o}} \rangle = {:.1f}$".format(mean_po[i]),linestyle='dashed',color=col_list[i])

for i in range(Ns):
    plt.plot([p_min,p_max],[std_po[i],std_po[i]],label=r"$\sigma_{{p_\mathrm{{o}}}} = {:.1f}$".format(std_po[i]),linestyle='dotted',color=col_list[i])

plt.text(25,65,r"$b_\mathregular{{ani}} = {:.0f} \overline{{B}}$".format(bani/Bbar),size='small')
plt.text(25,50,r"$p_B$ = {:.1f}$^\circ$".format(deg_pB),size='small')

leg= ax2.legend(loc='upper left',handlelength=1.5,ncol=3,prop={'size':0.99*leg_textsize,'family':'Times New Roman'},fancybox=False,framealpha=1,handletextpad=0.7,columnspacing=0.7)

ax2.tick_params(axis='both', which='minor', labelsize=axis_textsize, colors='k', length=3 , width=1   )
ax2.tick_params(axis='both', which='major', labelsize=axis_textsize, colors='k', length=5 , width=1.25)

if Use_degrees:
    xlabel = r'$p$ [degrees]'
    ylabel = r'$p_\mathrm{ord}$ [degrees]'
else:
    xlabel = r'$p$ [radians]'
    ylabel = r'$p_\mathrm{ord}$ [radians]'

ax2.set_xlabel(xlabel, size=axis_textsize)
ax2.set_xticklabels(ax2.get_xticks(),fontname = "Times New Roman", fontsize=axis_textsize)
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%g'))	
    
ax2.set_ylabel(ylabel, size=axis_textsize)
ax2.set_yticklabels(ax2.get_yticks(),fontname = "Times New Roman", fontsize=axis_textsize)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%g'))	

# plt.savefig('po.png', format='png', bbox_inches='tight', dpi= 300)
plt.show()