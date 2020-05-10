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

def template(psi_v, sigma_psi, dpsi, Qs, Us, mask, Nside=256):
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
    
    Npix = hp.nside2npix(Nside)
    
    Urj353 = Kcmb2Krj(353)#*1e-6
    Urj217 = Kcmb2Krj(217)#*1e-6
    Urj143 = Kcmb2Krj(143)#*1e-6

    Ps = np.sqrt(Qs**2 + Us**2)#*Urj353 #uKrj
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
    var_P = (Qs**2*C_QQ + C_UU*Us**2 + 2*Qs*Us*C_QU)/(Ps**2)
    sigma_P = np.sqrt(var_P) * 1e6 #* (Urj353) # uKrj    
    sigma_psi = sigma_psi[mask]
    
    # templates:
    Qtemp_353, Utemp_353 = vis2submm(psi_v, Ps, dpsi) #uKrj
    Ptemp_353 = np.sqrt(Qtemp_353**2 + Utemp_353**2)  # uKrj  
    Q353 = Qs #*(Urj353) # uKrj
    U353 = Us #*(Urj353) # uKrj
    P353 = Ps
    print(Q353, Qtemp_353)
    print(U353, Utemp_353)
    
    # Load 143GHz and 217GHz map
    m143 = read_H5('Data/IQU_Nside{}_143_15arcmin.h5'.format(Nside), 'IQU')
    Q143 = m143[1,mask]*1e6 #*(Urj353) # uKrj
    U143 = m143[2,mask]*1e6 #*(Urj353) # uKrj
    P143 = np.sqrt(Q143**2 + U143**2) # uKrj
    Qtemp_143, Utemp_143 = vis2submm(psi_v, P143) # uKrj
    Ptemp_143 = np.sqrt(Qtemp_143**2 + Utemp_143**2) 

    m217 = read_H5('Data/IQU_Nside{}_217_15arcmin.h5'.format(Nside), 'IQU')
    Q217 = m217[1,mask]*1e6 #*(Urj353) #uKrj
    U217 = m217[2,mask]*1e6 #*(Urj353) #uKrj
    P217 = np.sqrt(Q217**2 + U217**2) #uKrj
    Qtemp_217, Utemp_217 = vis2submm(psi_v, P217) #uKrj
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
    
    print('--- Spectral Index ---')
    data = [Q143, Q217, Q353, U143, U217, U353, P143, P217, P353]
    #std_betaQ, std_betaU, std_betaP = sampling(psi_v, sigma_psi,\
    #                                           P353, sigma_P, data)

    print('beta Q')
    beta_Q = get_beta([Qres143, Qres217, Qres353])
    print(beta_Q)
    #print(std_betaQ)
    print('beta U')
    beta_U = get_beta([Ures143, Ures217, Ures353])
    print(beta_U)
    #print(std_betaU)
    print('beta P')
    beta_P = get_beta([Pres143, Pres217, Pres353])
    print(beta_P)
    #print(std_betaP)
    #
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

    plt.show()

def sampling(psi_v, sigma_psi, P353, sigma_P, data, N=1000):
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
    # Sample to get std on beta
    N = 10
    betaQ = np.zeros((3, N))
    betaU = np.zeros((3, N))
    betaP = np.zeros((3, N))
    for i in range(N):
        # Gauss distribute P and psi, and draw:
        P = np.random.normal(P353, sigma_P)
        psi = np.random.normal(psi_v*180/np.pi, sigma_psi)*np.pi/180

        # Make template with the drawn P and psi
        Q353_temp, U353_temp = vis2submm(psi, P)
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

        # Calculate beta
        betaQ[:,i] = get_beta([a_Q143, a_Q217, a_Q353])
        betaU[:,i] = get_beta([a_U143, a_U217, a_U353])
        betaP[:,i] = get_beta([a_P143, a_P217, a_P353])

    #
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


def vis2submm(psi_v, P_s, dpsi=0):
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
    #print(dpsi)
    psi_s = psi_v #- np.pi/2. #+ dpsi
    Q = -P_s * np.cos(2*psi_s)
    U = P_s * np.sin(2*psi_s)  # healpix convention??
    #Q = -2*P_s * (np.cos(psi_s+dpsi) * np.cos(psi_s-dpsi))
    #U = -2*P_s * (np.sin(psi_s+dpsi) * np.cos(psi_s-dpsi))
    return(Q, U)

def B(T=2.7255, nu=[143., 217., 353.]):
    """
    Planck spectrum:
    """
    h = 6.62607994e-34
    k = 1.38064852e-23
    T = 2.72550 #?? 19.6 K
    nu = np.asarray(nu)

    x = h*(nu*1e9)/(k*T)
    g = (np.exp(x)-1)**2/(x**2*np.exp(x))
    return(g)

def Kcmb2Krj(fref):
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
    dBdTcmb = cu.dBdTcmb(f[i][0])
    dBdTcmb = np.nan_to_num(dBdTcmb)
    #print('', dBdTcmb)
    U_rj = cu.UnitConv(f[i][0], tau[i][0], dBdTcmb, dBdTrj,\
                              'K_CMB', 'K_RJ', fref)
    #print(U_rj)
    return(U_rj)

def MBB(nu, nu0=353*1e9, T=19.6):
    h = 6.62607994e-34
    k = 1.38064852e-23
    T =  19.6 #k
    nu = np.asarray(nu)*1e9
    
    exp1 = np.exp(h*nu0/(k*T)) - 1
    exp2 = np.exp(h*nu/(k*T)) - 1
    mbb = exp1/exp2
    return(mbb)
    

def get_beta(a, nu=[143.,217.,353.]):
    """
    calculate the spectral index from the minimizing factors and the 
    frequencies. convert to the betas to be valid for k_rj units.
    
    parameters:
    - a, list: the minimizing factors, sorted after increasing freq.
    - nu, list: the frequensies. must be of same order and length as a.
    """

    # planck spectrum
    nu = np.asarray(nu)
    fac = B(T=19.6) # (e^x-1)^2 / (x^2 e^x)
    fac2 = MBB(nu) #(e^x0-1) / (e^x-1)
    print(fac2)
    print(fac)
    #1.68 FOR 143, 3.20 FOR 217 OG 13.7 FOR 353.
    a[0] = a[0] * (fac2[0] / fac[0])
    a[1] = a[1] * (fac2[1] / fac[1])
    a[2] = a[2] * (fac2[2] / fac[2])
    #print(a)
    log_a = np.log(a[-1]/a) # 
    #print(log_a)
    log_nu = np.log(353/nu)
    #print(log_nu)
    beta = log_a/log_nu
    return(beta)



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
    print(f.keys())
    f.close()
    print(np.shape(data))
    return(data)
