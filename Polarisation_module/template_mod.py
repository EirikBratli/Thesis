"""
Template of converte RoboPol polarisation to Planck polarsation. 
Use eq.12 from Planck Int. XXI 2015 Q_s = -q_v*P_s/p_v, U_s = -u_v*P_s/p_v
This assumes Delta psi = 0 which is not the case: 
Delta psi = psi_s+90 - psi_v --> psi_v = psi_s+90 - Delta psi

This gives with q = pcos(2psi), u = -psin(2psi)
Q_s = -P_s*cos[2(psi_s+90-Delta psi)]
U_s = -P_s*sin[2(psi_s+90-Delta psi)]
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

def template(psi_s, psi_v, dpsi, Qs, Us, mask, Nside=256):
    """
    Create templates for Q and U polarisation.
    """

    Npix = hp.nside2npix(Nside)

    Ps = np.sqrt(Qs**2 + Us**2)
    print(np.shape(psi_v), np.shape(psi_s), np.shape(dpsi))
    # test angles:
    validate_angle(psi_s, psi_v, dpsi)
    """
    vinkel = np.arange(0, np.pi/2, 0.01)
    # test:
    rq = []
    ru = []
    for v in vinkel:
        q, u = vis2submm(psi_v, v, Ps)
        p = np.sqrt(q**2 + u**2)
        resQ = residuals(q, Qs)
        resU = residuals(u, Us)
        resP = residuals(p, Ps)
        print('angle:', v*180/np.pi)
        print(resQ, resU, resP)
        print('---')
        rq.append(resQ)
        ru.append(resU)
    plt.plot(vinkel, rq, '-b')
    plt.plot(vinkel, ru, '-r')
    plt.ylim(-1,1)
    plt.show()
    print(np.std(dpsi), np.std(psi_v))
    sys.exit()
    """
    print(np.mean(dpsi), np.median(dpsi))
    #dpsi = np.pi/2 - (dpsi)#88.8*np.pi/180.
    Qtemp_353, Utemp_353 = vis2submm(psi_v, dpsi, Ps)
    Ptemp_353 = np.sqrt(Qtemp_353**2 + Utemp_353**2)
    # test pol:
    validate_pol(Qs, Qtemp_353)
    validate_pol(Us, Utemp_353)
    validate_pol(Ps, Ptemp_353)
    Q353 = Qs
    U353 = Us
    P353 = Ps

    # Load 143GHz and 217GHz map
    m143 = read_H5('Data/IQU_Nside{}_143_15arcmin.h5'.format(Nside), 'IQU')
    Q143 = m143[1,mask]*1e6
    #print(Q143)
    U143 = m143[2,mask]*1e6
    P143 = np.sqrt(Q143**2 + U143**2)
    Qtemp_143, Utemp_143 = vis2submm(psi_v, dpsi, P143)
    Ptemp_143 = np.sqrt(Qtemp_143**2 + Utemp_143**2)
    #print(np.shape(m143))
    #validate_pol(Q143, Qtemp_353)
    #validate_pol(U143, Utemp_353)
    #validate_pol(P143, Ptemp_353)
    
    m217 = read_H5('Data/IQU_Nside{}_217_15arcmin.h5'.format(Nside), 'IQU')
    Q217 = m217[1,mask]*1e6
    U217 = m217[2,mask]*1e6
    P217 = np.sqrt(Q217**2 + U217**2)
    Qtemp_217, Utemp_217 = vis2submm(psi_v, dpsi, P217)
    Ptemp_217 = np.sqrt(Qtemp_217**2 + Utemp_217**2)
    #print(np.shape(m217))
    #validate_pol(Q217, Qtemp_353)
    #validate_pol(U217, Utemp_353)
    #validate_pol(P217, Ptemp_353)
    print(np.std(Q353), np.std(Qtemp_353), np.std(U353), np.std(Utemp_353))
    print(np.std(Q217), np.std(Qtemp_217), np.std(U217), np.std(Utemp_217))
    print(np.std(Q143), np.std(Qtemp_143), np.std(U143), np.std(Utemp_143))
    # Find residuals for each frequency, get a coefficient and calculate a 
    print('Calculate the residuals:')
    Qres353 = residuals(Qtemp_353, Q353)
    Ures353 = residuals(Utemp_353, U353)
    Pres353 = residuals(Ptemp_353, P353)
    #q_temp =np.sqrt(Ptemp_353**2 - Utemp_353**2)
    #qres = residuals(q_temp, np.abs(Q353))
    #print(q_temp,Qtemp_353)
    #print(qres)
    print('Residuals, 353-353 (Q,U,P):', Qres353, Ures353, Pres353)
    print(np.sqrt(Qres353**2 + Ures353**2))
    print(np.corrcoef(Qtemp_353, Q353))
    print(np.corrcoef(Utemp_353, U353))
    plt.figure()
    plt.scatter(Qtemp_353, Q353)
    plt.xlabel('Q template')
    plt.ylabel('Q data')
    plt.savefig('Figures/template/test_scatter_Q.png')
    plt.figure()
    plt.scatter(Utemp_353, U353)
    plt.xlabel('U template')
    plt.ylabel('U data')
    plt.savefig('Figures/template/test_scatter_U.png')
    plt.figure()
    plt.scatter(Ptemp_353, P353)
    plt.xlabel('P template')
    plt.ylabel('P data')

    #"""
    Qres217 = residuals(Q353, Q217)
    Ures217 = residuals(Utemp_353, U217)
    Pres217 = residuals(Ptemp_353, P217)
    print('Residuals, 217-353 (Q,U,P):', Qres217, Ures217, Pres217)
    print(np.sqrt(Qres217**2 + Ures217**2))
    Qres143 = residuals(Q353, Q143)
    Ures143 = residuals(Utemp_353, U143)
    Pres143 = residuals(Ptemp_353, P143)
    print('Residuals, 143-353 (Q,U,P):', Qres143, Ures143, Pres143)
    print(np.sqrt(Qres143**2 + Ures143**2))
    print('--- Spectral Index ---')
    print('beta Q')
    get_beta([Qres143, Qres217, Qres353])
    print('beta U')
    get_beta([Ures143, Ures217, Ures353])
    print('beta P')
    get_beta([Pres143, Pres217, Pres353])
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
    
    # Regression against 143, 217 and 353 GHz maps of Planck:
    regression([Qres143, Qres217, Qres353], 'Q')
    regression([Ures143, Qres217, Ures353], 'U')
    regression([Pres143, Pres217, Pres353], 'P')
    # spectral index, beta
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

def residuals(Mtemp, Mfreq):
    """
    Calculate the residuals between the maps.
    """
    print(Mtemp)
    print(Mfreq)
    print(((Mfreq-Mtemp)/np.abs(Mfreq))**2)
    def min_func(a, Mtemp=Mtemp, Mfreq=Mfreq):
        return(np.sum(((Mfreq - a*Mtemp)/np.abs(Mfreq)))**2)

    minimize = spo.fmin_powell(min_func, x0=1)
    print(minimize)
    return(minimize)


def vis2submm(psi_v, dpsi, P_s, Nside=256):
    """
    Compute Q_s, U_s from visual polarisation
    """
    #print(dpsi)
    psi_s = psi_v - np.pi/2. #+ dpsi
    Q = -P_s * np.cos(2*psi_s)
    U = -P_s * np.sin(2*psi_s)  # healpix convention??
    #Q = -2*P_s * (np.cos(psi_s+dpsi) * np.cos(psi_s-dpsi))
    #U = -2*P_s * (np.sin(psi_s+dpsi) * np.cos(psi_s-dpsi))
    return(Q, U)

def get_beta(a, nu=[143.,217.,353.]):
    h = 6.62607994e-34
    k = 1.38064852e-23
    T = 2.72550 #??
    nu = np.asarray(nu)
    
    x = h*(nu*1e9)/(k*T)
    fac = (np.exp(x)-1)**2/(x**2*np.exp(x))
    print(fac)
    #1.68 for 143, 3.20 for 217 og 13.7 for 353.
    a[0] = a[0]/fac[0]#1.68
    a[1] = a[1]/fac[1]#3.20
    a[2] = a[2]/fac[2]#13.7
    print(a)

    log_a = np.log(a[-1]/a) # 
    print(log_a)
    log_nu = np.log(353/nu)
    print(log_nu)
    beta = log_a/log_nu
    print(beta)
    return(beta)
    
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

def validate_angle(psi_s, psi_v, dpsi, eps=1e-5):
    """
    Check if psi_v = psi_s+90-dpsi
    """
    test = psi_s + np.pi/2 - dpsi
    print(np.mean(test - psi_v))
    #print(test)
    #print(psi_v)
    
    ind = np.where(test-psi_v < eps)[0]
    if len(ind) > len(test)*0.9:
        print('OK')
    else:
        print('Not OK')

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
