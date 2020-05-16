"""
Template of converte RoboPol polarisation to Planck polarsation. 
Use eq.12 from Planck Int. XXI 2015 Q_s = -q_v*P_s/p_v, U_s = -u_v*P_s/p_v
This assumes Delta psi = 0 which is not the case: 
Delta psi = psi_s+90 - psi_v --> psi_s = psi_v - 90 + Delta psi

This gives with q = pcos(2psi), u = -psin(2psi)
Q_s = -P_s*cos[2(psi_v-90)]
U_s = -P_s*sin[2(psi_v-90)]
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import h5py, argparse
import sys, time, glob, os
import scipy.optimize as spo

from astropy import units as u_
from astropy.coordinates import SkyCoord

import convert_units as cu
import tools_mod as tools
import smoothing_mod as smooth
import plotting_mod as plotting
import load_data_mod as load


################################

def template(psi_v, sigma_psi, dpsi, Qs, Us, qu_v, mask, Nside=256):
    """
    Create templates for Q and U polarisation and calculate the spectral 
    index for Modified blackbody radiation.

    Parameters:
    - psi_v, array: the polarisation angels from starlight polarisation
                    in visual.
    - sigma_psi, array: the uncertainty of psi_v.
    - dpsi, array: the difference in polarisation angle between 
                   submillimeter and visual.
    - Qs, array: the Stokes Q polarisation parameter
    - Us, array: the stokes U polarisation parameter
    - mask, array: mask with the pixels used in the analysis.
    - Nside, integer. Map resolution

    """
    psi_v = psi_v #+ 2*np.pi/180 # add a shift in pol.angle
    print(np.mean(psi_v)*180/np.pi)
    Npix = hp.nside2npix(Nside)
    #b = a2t(19.6)
    Urj353 = Kcmb2Krj(353)#*1e-6 # convert to K_RJ units
    Urj217 = Kcmb2Krj(217)#*1e-6
    Urj143 = Kcmb2Krj(143)#*1e-6

    q, q_err, u, u_err = qu_v[:]

    Urj = np.array([Urj143, Urj217, Urj353])
    Ps = np.sqrt(Qs**2 + Us**2) #*Urj353 #uKrj
    #print(psi_v)
    #print(Ps)
    # import uncertainties on data:
    Cfile = 'Data/Planck_Cij_353_2048_full.h5'
    C_ij = load.C_ij(Cfile, Nside) #Kcmb^2
    C_II = C_ij[0,mask]
    C_IQ = C_ij[1,mask]
    C_IU = C_ij[2,mask]
    C_QQ = C_ij[3,mask]
    C_QU = C_ij[4,mask]
    C_UU = C_ij[5,mask]
    #var_P = (Qs**2*C_QQ + C_UU*Us**2 + 2*Qs*Us*C_QU)/(Ps**2)
    #sigma_P = np.sqrt(var_P) * 1e6 #* (Urj353) # uKrj    
    sigma_psi = sigma_psi[mask]
    
    # templates:
    Qtemp_353, Utemp_353 = vis2submm(Ps, q, u) #uKrj
    Ptemp_353 = np.sqrt(Qtemp_353**2 + Utemp_353**2)  # uKrj  
    Q353 = Qs #*(Urj353) # uKrj
    U353 = Us #*(Urj353) # uKrj
    P353 = Ps
    print(Q353, Qtemp_353)
    print(U353, Utemp_353)
    
    # Load 143GHz and 217GHz map
    m143 = read_H5('Data/IQU_Nside{}_143_15arcmin.h5'.format(Nside), 'IQU')
    Q143 = m143[1,mask]*1e6 #*(Urj143) # uKrj
    U143 = m143[2,mask]*1e6 #*(Urj143) # uKrj
    P143 = np.sqrt(Q143**2 + U143**2) # uKrj
    Qtemp_143, Utemp_143 = vis2submm(P143, q, u) # uKrj
    Ptemp_143 = np.sqrt(Qtemp_143**2 + Utemp_143**2) 

    m217 = read_H5('Data/IQU_Nside{}_217_15arcmin.h5'.format(Nside), 'IQU')
    Q217 = m217[1,mask]*1e6 #*(Urj217) #uKrj
    U217 = m217[2,mask]*1e6 #*(Urj217) #uKrj
    P217 = np.sqrt(Q217**2 + U217**2) #uKrj
    Qtemp_217, Utemp_217 = vis2submm(P217, q, u) #uKrj
    Ptemp_217 = np.sqrt(Qtemp_217**2 + Utemp_217**2)

    # Find residuals for each frequency, get a coefficient and calculate a 
    print('Calculate the residuals:')
    Qres353 = residuals(Qtemp_353, Q353)
    Ures353 = residuals(Utemp_353, U353)
    Pres353 = residuals(Ptemp_353, P353)
    print(np.corrcoef(Qtemp_353, Q353))
    print(np.corrcoef(Utemp_353, U353))
    print('Residuals, 353-353 (Q,U,P):', Qres353, Ures353, Pres353)

    Qres217 = residuals(Qtemp_353, Q217)
    Ures217 = residuals(Utemp_353, U217)
    Pres217 = residuals(Ptemp_353, P217)
    print('Residuals, 217-353 (Q,U,P):', Qres217, Ures217, Pres217)
    
    Qres143 = residuals(Qtemp_353, Q143)
    Ures143 = residuals(Utemp_353, U143)
    Pres143 = residuals(Ptemp_353, P143)
    print('Residuals, 143-353 (Q,U,P):', Qres143, Ures143, Pres143)
    #sys.exit()
    print('--- Spectral Index ---')
    data = [Q143, Q217, Q353, U143, U217, U353, P143, P217, P353]
    std_betaQ, std_betaU, std_betaP = sampling(qu_v, np.sqrt(C_QQ),\
                                               np.sqrt(C_UU), data)

    #print(Ures143/Ures353, Ures217/Ures353)
    #print(Qres143/Qres353, Qres217/Qres353)
    print('beta Q')
    beta_Q = get_beta([Qres143, Qres217, Qres353])
    print('beta_Q', beta_Q)
    print(' +/-  ',std_betaQ)
    print('beta U')
    beta_U = get_beta([Ures143, Ures217, Ures353])
    print('beta_U:',beta_U)
    print('  +/-  ',std_betaU)
    print('beta P')
    #beta_P = get_beta([Pres143, Pres217, Pres353])
    #print('beta_P:',beta_P)
    #print('  +/-  ',std_betaP)
    #
    nu = np.array([143, 217, 353])
    
    if (qu_v is not None):
        
        Q = [Q143, Q217, Q353]
        U = [U143, U217, U353]
        #map_temp(Q353, Qtemp_353, mask, 'Q353', '-5')
        #map_temp(U353, Utemp_353, mask, 'U353', '-5')
        #map_temp(Q217, Qtemp_217, mask, 'Q217', '-5')
        #map_temp(U217, Utemp_217, mask, 'U217', '-5')
        #map_temp(Q143, Qtemp_143, mask, 'Q143', '-5')
        #map_temp(U143, Utemp_143, mask, 'U143', '-5')        
        #scatter_plots(Q,U, qu_v, C_ij[:,mask])
        # plott scatter plots, 353, 217, 143
        data = [Q143,Q217,Q353,U143,U217,U353]
        temp = [Qtemp_143,Qtemp_217,Qtemp_353,Utemp_143,Utemp_217,Utemp_353]
        #plot_psi_shift_temp_data(data, temp, qu_v, delta='0')

    plt.show()
        

def vis2submm(P_s, q, u, psi_v=0, dpsi=0):
    """
    Compute Q_s, U_s from visual polarisation using the polarisation intensity P
    
    Parameters:
    - psi_v, array: the polarisation angles in visual
    - Ps, array: the polarisation intensity
    - dpsi, scalar/array: the difference between visula and submillimeter 
                          polarisaion angle. Default is 0. (not in use)
    Returns:
    Q and U template
    """
    pv = np.sqrt(q**2 + u**2)
    R = P_s/pv
    psi_v = psi_v #+5*np.pi/180. #+ dpsi
    Q = -R * q #-P_s * np.cos(2*psi_v)
    U = -R * u #-P_s * np.sin(2*psi_v)  
    return(Q, U)

def sampling(qu_v, sigma_Q, sigma_U, data, N=1000, psi_v=None, sigma_psi=None):
    """
    Sampling function to compute the error on beta. Return the full list
    of beta
    
    - draw P, psi from normal distibution N(data points, error in data)
    - make template for 353GHz
    - calc residuals -> get the minimizing factor 'a'
    - calc beta from the 'a's
    # print the betas and sigma_beta for Q,U and P polarisation
    """

    Q143, Q217, Q353, U143, U217, U353, P143, P217, P353 = data[:]
    qv, q_err, uv, u_err = qu_v[:]
    # Sample to get std on beta
    sigma_Q *= 1e6
    sigma_U *= 1e6
    betaQ = np.zeros((3, N))
    betaU = np.zeros((3, N))
    betaP = np.zeros((3, N))
    aU = np.zeros((3, N))
    temp = np.zeros((N))
    #print(P353/sigma_P)
    #print(psi_v*(180/np.pi)/sigma_psi)
    for i in range(N):
        # Gauss distribute P and psi, and draw:
        Q = np.random.normal(Q353, sigma_Q)
        U = np.random.normal(U353, sigma_U)
        P = np.sqrt(Q**2 + U**2)
        q = np.random.normal(qv, q_err)
        u = np.random.normal(uv, u_err)
        #psi = np.random.normal(psi_v*180/np.pi, sigma_psi)*np.pi/180

        # Make template with the drawn P and psi
        Q353_temp, U353_temp = vis2submm(P, q, u)
        P353_temp = np.sqrt(Q353_temp**2 + U353_temp**2)
        
        # Calculate the residuals, return the minimizing factor
        a_Q353 = residuals(Q353_temp, Q353)
        a_Q217 = residuals(Q353_temp, Q217)
        a_Q143 = residuals(Q353_temp, Q143)
        a_U353 = residuals(U353_temp, U353)
        a_U217 = residuals(U353_temp, U217)
        a_U143 = residuals(U353_temp, U143)
        a_P353 = residuals(P353_temp, P353)
        a_P217 = residuals(P353_temp, P217)
        a_P143 = residuals(P353_temp, P143)
        
        aU[:,i] = a_U143, a_U217, a_U353
        temp[i] = np.mean(U353_temp)
        # Calculate beta
        betaQ[:,i] = get_beta([a_Q143, a_Q217, a_Q353])
        betaU[:,i] = get_beta([a_U143, a_U217, a_U353])
        betaP[:,i] = get_beta([a_P143, a_P217, a_P353])

    #
    plt.figure()
    plt.hist(temp, bins=50)
    print(np.std(temp), np.std(U353))
    print(np.min(betaU[1,:]), np.max(betaU[1,:]))
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, sharey=True)
    ax1.hist(betaU[0,:], bins=50, color='b', histtype='step',\
             label=r'$\beta_{{143/353}}^U$')
    ax1.axvline(x=np.mean(betaU[0,:])-np.std(betaU[0,:]), ls='--',c='k',label=r'1$\sigma$')
    ax1.axvline(x=np.mean(betaU[0,:])+np.std(betaU[0,:]), ls='--',c='k')
    ax2.hist(betaU[1,:], bins=50, color='r', histtype='step',\
             label=r'$\beta_{{217/353}}^U$')
    ax2.axvline(x=np.mean(betaU[1,:])-np.std(betaU[1,:]), ls='--',c='k',label=r'1$\sigma$')
    ax2.axvline(x=np.mean(betaU[1,:])+np.std(betaU[1,:]), ls='--',c='k')

    ax3.hist(aU[0,:]/aU[2,:], bins=50, color='b', histtype='step',\
             label=r'$\alpha_{{143/353}}^U$')
    ax3.axvline(x=np.mean(aU[0,:]/aU[2,:])-np.std(aU[0,:]/aU[2,:]), ls='--',\
                c='k',label=r'1$\sigma$')
    ax3.axvline(x=np.mean(aU[0,:]/aU[2,:])+np.std(aU[0,:]/aU[2,:]), ls='--',c='k')
    ax4.hist(aU[1,:]/aU[2,:], bins=50, color='r', histtype='step',\
             label=r'$\alpha_{{217/353}}^U$')
    ax4.axvline(x=np.mean(aU[1,:]/aU[2,:])-np.std(aU[1,:]/aU[2,:]), ls='--',\
                c='k',label=r'1$\sigma$')
    ax4.axvline(x=np.mean(aU[1,:]/aU[2,:])+np.std(aU[1,:]/aU[2,:]), ls='--',c='k')
    

    #ax1.set_xlabel('143-353')
    ax1.set_ylabel('Number of samples')
    ax1.legend()
    #ax2.set_xlabel('217-353')
    ax2.legend()
    ax3.set_xlabel('143-353')
    ax3.set_ylabel('Number of samples')
    ax3.legend()
    ax4.set_xlabel('217-353')
    ax4.legend()
    plt.tight_layout()
    fig.savefig('Figures/test_betaU_hist.png')
    print(np.mean(betaQ, axis=1), np.mean(betaU, axis=1), np.mean(betaP, axis=1))
    return(np.std(betaQ, axis=1), np.std(betaU, axis=1), np.std(betaP, axis=1))


def residuals(Mtemp, Mfreq):
    """
    Calculate the residuals between the template and data maps.
    
    Parameters:
    - Mtemp, array: the template map
    - Mfreq, array: the frequency map with data.
    Returns:
    - the minimising factor
    """
 
    def min_func(a, Mtemp=Mtemp, Mfreq=Mfreq):
        return(np.sum(((Mfreq - a*Mtemp)/np.abs(Mfreq)))**2)

    minimize = spo.fmin_powell(min_func, x0=1, disp=False)
    #print(minimize)
    return(minimize)


def a2t(nu=[143., 217., 353.]):
    """
    Planck spectrum:
    """
    h = 6.62607994e-34
    k = 1.38064852e-23
    T = 2.72550
    nu = np.asarray(nu)

    x = h*(nu*1e9)/(k*T)
    g = (np.exp(x)-1)**2/(x**2*np.exp(x))
    return(g)

def MBB(nu, nu0=353*1e9, T=19.6):
    h = 6.62607994e-34
    k = 1.38064852e-23
    #T =  19.6 #k
    nu = np.asarray(nu)*1e9
    
    exp1 = np.exp(h*nu0/(k*T)) - 1
    exp2 = np.exp(h*nu/(k*T)) - 1
    mbb = exp1/exp2
    return(mbb)
    
def exp_factor(nu, T=19.6):
    h = 6.62607994e-34
    k = 1.38064852e-23
    x = h*nu*1e9/(k*T)
    return(np.exp(x) - 1)

def x(nu, T=19.6):
    h = 6.62607994e-34
    k = 1.38064852e-23
    return(h*nu*1e9/(k*T))

def get_beta(alpha, nu=[143.,217.,353.]):
    """
    calculate the spectral index from the minimizing factors and the 
    frequencies. convert to the betas to be valid for k_rj units.
    
    parameters:
    - a, list: the minimizing factors, sorted after increasing freq.
    - nu, list: the frequensies. must be of same order and length as a.
    """
    a = np.zeros(len(nu))
    nu = np.asarray(nu)
    b0 = a2t() # (e^x-1)^2 / (x^2 e^x)
    #1.68 FOR 143, 3.20 FOR 217 OG 13.7 FOR 353.

    a[0] = alpha[0]*exp_factor(143)/b0[0] 
    a[1] = alpha[1]*exp_factor(217)/b0[1]
    a[2] = alpha[2]*exp_factor(353)/b0[2]
    log_a = np.log(a/a[2]) # 
    log_nu = np.log(nu/353)
    beta = log_a/log_nu
    return(beta-1)

def map_temp(data, temp, mask, pol, delta, Nside=256):
    map1 = np.full(12*Nside**2, hp.UNSEEN)
    map2 = np.full(12*Nside**2, hp.UNSEEN)
    
    map1[mask] = data
    map2[mask] = temp
    
    hp.gnomview(map1, title='{} polarisation'.format(pol),\
                unit=r'$\mu K_{cmb}$', rot=[104.1,22.225], xsize=100,\
                min=np.min(map1[mask]), max=np.max(map1[mask]))
    hp.graticule()
    plt.savefig('Figures/{}_map.png'.format(pol))

    hp.gnomview(map2, title='{} template shift {} deg'.format(pol, delta),\
                unit=r'$\mu K_{cmb}$', rot=[104.1,22.225], xsize=100,\
                min=np.min(map1[mask]), max=np.max(map1[mask]))
    hp.graticule()
    plt.savefig('Figures/{}_map_temp_{}.png'.format(pol, delta))


def plot_psi_shift_temp_data(data, temp, qu_v, delta='45'):
    Q143,Q217,Q353,U143,U217,U353 = data[:]
    Qtemp143,Qtemp217,Qtemp353,Utemp143,Utemp217,Utemp353 = temp[:]
    q,q_err,u,u_err = qu_v
    
    plt.figure()
    plt.scatter(q, Qtemp353, c='k', marker='x', label=r'Qu temp $\delta={}$'.format(delta))
    plt.scatter(u, Utemp353, c='b', marker='x', label=r'Uu temp $\delta={}$'.format(delta))
    plt.scatter(q, Q353, c='k', marker='.', label='Qu data')
    plt.scatter(u, U353, c='b', marker='.', label='Uu data')
    plt.title('353 GHz')
    plt.xlabel(r'$q, u$')
    plt.ylabel(r'$Q, U$ $[\mu K_{cmb}]$')
    plt.legend()
    plt.savefig('Figures/test_data_vs_temp_{}_353.png'.format(delta))

    plt.figure()
    plt.scatter(q, Qtemp217, c='k', marker='x', label=r'Qu temp $\delta={}$'.format(delta))
    plt.scatter(u, Utemp217, c='b', marker='x', label=r'Uu temp $\delta={}$'.format(delta))
    plt.scatter(q, Q217, c='k', marker='.', label='Qu data')
    plt.scatter(u, U217, c='b', marker='.', label='Uu data')
    plt.title('217 GHz')
    plt.xlabel(r'$q, u$')
    plt.ylabel(r'$Q, U$ $[\mu K_{cmb}]$')
    plt.legend()
    plt.savefig('Figures/test_data_vs_temp_{}_217.png'.format(delta))

    plt.figure()
    plt.scatter(q, Qtemp143, c='k', marker='x', label=r'Qu temp $\delta={}$'.format(delta))
    plt.scatter(u, Utemp143, c='b', marker='x', label=r'Uu temp $\delta={}$'.format(delta))
    plt.scatter(q, Q143, c='k', marker='.', label='Qu data')
    plt.scatter(u, U143, c='b', marker='.', label='Uu data')
    plt.title('143 GHz')
    plt.xlabel(r'$q, u$')
    plt.ylabel(r'$Q, U$ $[\mu K_{cmb}]$')
    plt.legend()
    plt.savefig('Figures/test_data_vs_temp_{}_143.png'.format(delta))

    """
    plot_residualUs(Qres353, Qtemp_353, Q353, Nside, mask, 353, pol='Q')
    plot_residuals(Ures353, Utemp_353, U353, Nside, mask, 353, pol='U')
    plot_residuals(Pres353, Ptemp_353, P353, Nside, mask, 353, pol='P')        
    plot_residuals(Qres217, Qtemp_353, Q217, Nside, mask, 217, pol='Q')
    plot_residuals(Ures217, Utemp_353, U217, Nside, mask, 217, pol='U')
    plot_residuals(Pres217, Ptemp_353, P217, Nside, mask, 217, pol='P')
    plot_residuals(Qres143, Qtemp_353, Q217, Nside, mask, 143, pol='Q')
    plot_residuals(Ures143, Utemp_353, U217, Nside, mask, 143, pol='U')
    plot_residuals(Pres143, Ptemp_353, P217, Nside, mask, 143, pol='P')
    #"""

def scatter_plots(Q, U, qu_v, Cij, Nside=256):
    unit = 287.45*1e-6 # to MJy/sr from uK_cmb
    for i in range(3):
        Q[i] = Q[i]*unit
        U[i] = U[i]*unit
    Q143, Q217, Q353 = Q[:]
    U143, U217, U353 = U[:] 
    q, q_err, u, u_err = qu_v[:]
    print(np.shape(q), np.shape(u))
    qu = np.concatenate((q, u), axis=0)
    QU143 = np.concatenate((Q143, U143), axis=0)
    QU217 = np.concatenate((Q217, U217), axis=0)
    QU353 = np.concatenate((Q353, U353), axis=0)
    print(np.shape(qu), np.shape(QU143))

    R143 = np.corrcoef(qu, QU143)
    R217 = np.corrcoef(qu, QU217)
    R353 = np.corrcoef(qu, QU353)
    print('Correlation coefficients for QU, qu:')
    print('143 GHz:', R143)
    print('217 GHz:', R217)
    print('353 GHz:', R353)

    R143 = np.sqrt(Q143**2 + U143**2)/np.sqrt(q**2 + u**2)
    R217 = np.sqrt(Q217**2 + U217**2)/np.sqrt(q**2 + u**2)
    R353 = np.sqrt(Q353**2 + U353**2)/np.sqrt(q**2 + u**2)
    print('R_P/p (143):', np.mean(R143), np.std(R143))
    print('R_P/p (217):', np.mean(R217), np.std(R217))
    print('R_P/p (353):', np.mean(R353), np.std(R353))

    x = np.linspace(-0.03,0.01)
    plt.figure('correlation')
    from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
    s01 = plt.scatter(q, Q353, marker='d', c='b')#, label=r'$Q_{{353}}, q_v$')
    s02 = plt.scatter(u, U353, marker='d', c='b')#, label=r'$U_{{353}}, u_v$')

    s11 = plt.scatter(q, Q143, marker='x', c='g')#, label=r'$Q_{{143}}, q_v$')
    s12 = plt.scatter(u, U143, marker='x', c='g')#, label=r'$U_{{143}}, u_v$')
    
    s21 = plt.scatter(q, Q217, marker='.', c='r')#, label=r'$Q_{{217}}, q_v$')
    s22 = plt.scatter(u, U217, marker='.', c='r')#, label=r'$U_{{217}}, u_v$')

    leg1 = plt.legend([(s11, s12),(s21,s22),(s01,s02)],[r'143:', r'217:','353:'],\
                      numpoints=1, loc=9, fontsize=11,\
                      handler_map={tuple: HandlerTuple(ndivide=None)}, framealpha=0)
    ax = plt.gca().add_artist(leg1)

    l1 = plt.plot(x, -np.mean(R143)*x, '-.k',\
                  label=r'$R_{{P/p}}={}\pm{}$'.format(round(np.mean(R143),3),\
                                                           round(np.std(R143),3)),)
    l2 = plt.plot(x, -np.mean(R217)*x, '--k',\
                  label=r'$R_{{P/p}}={}\pm{}$'.format(round(np.mean(R217),3),\
                                                           round(np.std(R217),3)))
    l3 = plt.plot(x, -np.mean(R353)*x, '-k',\
                  label=r'$R_{{P/p}}={}\pm{}$'.format(round(np.mean(R353),3),\
                                                           round(np.std(R353),3)))    
    
    #plt.legend(handles=[l1, l2, l3], loc=1)
    #ax = plt.gca().add_artist(leg1)
    plt.xlabel(r'$q_v, u_v$')
    plt.ylabel(r'$Q_s, U_s$')
    plt.legend(loc=1, framealpha=0)
    plt.savefig('Figures/correlations/scatter_{}_QUqu_143_217_353.png'.format(Nside))
    plt.show()

def plot_residuals(a, temp, f_map, Nside, mask, f1,f2=353, pol='P'):
    Npix = hp.nside2npix(Nside)
    map = np.full(Npix, hp.UNSEEN)
    
    map[mask] = (f_map - a*temp)/np.abs(f_map)#((f_map - a*temp)/np.abs(f_map))**2
    lon, lat = 104.1, 22.225
    #print(f_map)
    #print(a*temp)
    #print(map[mask])
    hp.gnomview(map, title=r'Residuals {}GHz: $({}-a*{}_{{temp}})/|{}|$'.\
                format(f1,pol,pol,pol), rot=[lon,lat], xsize=100,\
                unit=r'$\mu K_{{CMB}}$', cmap='jet')
#    hp.gnomview(map, title=r'Residuals {}GHz: $(({}-a*{}_{{temp}})/|{}|)^2$'.\
#                format(f1,pol,pol,pol), rot=[lon,lat], xsize=100, cmap='jet')
    hp.graticule()
    plt.savefig('Figures/template/{}_residuals_{}vs{}template_{}.png'.\
                format(pol,f1,f2,Nside))


def Kcmb2Krj(fref, T=19.6):
    bands = np.array([30,44,70,100,143,217,353,545,857])
    i = np.where(bands==fref)[0]
    f,tau = cu.read_freq()
    #print(f[i][0])
    #print(np.shape(f), np.shape(f[i][0]), i)
    for j in range(3, len(f)):
        ind = np.where((f[j] > fref/2.) & (f[j] < 2*fref))[0]
        f_temp = f[j][ind]
        f[j] = f_temp
    #print(np.shape(f), np.shape(f[i][0]))    
    dBdTrj = cu.dBdTrj(f[i][0])
    #print(i, dBdTrj)
    
    # convert K_cmb to K_RJ
    dBdTcmb = cu.dBdTcmb(f[i][0], T)
    dBdTcmb = np.nan_to_num(dBdTcmb)
    #print('', dBdTcmb)
    U_rj = cu.UnitConv(f[i][0], tau[i][0], dBdTcmb, dBdTrj,\
                              'K_CMB', 'K_RJ', fref)
    #print(U_rj)
    return(U_rj)

    
def regression(a, ylab, nu=[143,217,353]):
    """
    Run regression between the template maps. Convert to K_RJ?
    
    Parameters:
    - a, list of the 'a'-values from residuals
    - ylab, string. Which Stokes parameter to look at
    """
    
    plt.figure()
    plt.scatter(nu, a, c='k', s=10, marker='x')

    plt.xticks(nu)
    plt.xlabel(r'Frequency $\nu$, [GHz]')
    plt.ylabel(r'$a$ for {}-polarisation'.format(ylab))
    plt.savefig('Figures/template/temp_{}_vs_freqbands.png'.format(ylab))
    #

def validate_pol(data, est, eps=1e-5):
    """
    Check if the estimated Q is similar to data Q.
    """
    #print(data)
    #print(est)
    alpha = (np.mean(np.abs(data/est)))
    print(alpha)
    if (alpha > 0.9) and (alpha < 1/0.9):
        print('OK')
    else:
        print('Not OK')
    #

def read_H5(file, name):
    
    f = h5py.File(file, 'r')
    data = np.asarray(f[name])
    #print(f.keys())
    f.close()
    print(np.shape(data))
    return(data)
