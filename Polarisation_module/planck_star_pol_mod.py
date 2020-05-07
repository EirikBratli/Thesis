"""
This is an extra module in the tomography module. Its purpuse is to recreate the
results of Planck XII 2018 polarisation results using Berdyugin stellar data.

I thinck I will do all in this program:
- load data
- get pol.frac q and u
- calculate delta psi, R_P/p, and correlation.

This module does not use smoothed planck polarisation maps.

Might need a module for the reddening later.
"""


import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import h5py
import sys, time, glob, os
import scipy.optimize as spo

from scipy import stats
from astropy import units as u_
from astropy.coordinates import SkyCoord

import convert_units as cu
import tools_mod as tools
import load_data_mod as load
import smoothing_mod as smooth

#######################################

def load_data(file):
    """
    Load the data, input is the filename of the file to load with path. Returns
    the data (l, b ,p_v, err_p_v, psi, err_psi, parallax).
    """
    data = np.genfromtxt(file, delimiter=';', skip_header=116, skip_footer=12,\
                        usecols=[0, 1, 7, 8, 9, 10, 11])

    # reduce data: remove rows with nan
    data = data[~np.isnan(data).any(axis=1)]
    print(np.shape(data))
    return(data)

def q_v(p, psi, sp, s_psi):
    """
    Calculate the q_v fractional polarisation from p_v and psi (galactic) and
    the uncertainty.
    """
    err = np.cos(2*psi)*sp #- p*np.sin(2*psi)*s_psi
    return(p*np.cos(2.*psi), err)

def u_v(p, psi, sp, s_psi):
    """
    Calculate the q_v fractional polarisation from p_v and psi (galactic) and
    the uncertainty. In Healpix convention.
    """
    err = np.sin(2*psi)*sp #+ p*np.cos(2*psi)*s_psi
    return(-p*np.sin(2.*psi), err)



def make_pol_map(pol, sigma, r, mask, pix, Nside=2048):
    """
    Make maps with the fractional polarisations, p,q,u at a given Nside.

    Parameters:
    -----------

    - pol, list of arrays.  The fractional polarisation parameters, [p,q,u]
    - sigma_pol, list.      The uncertainties of [p,q,u] in a list of array.
    - r, arrays             The distance to the stars.
    - mask, array.          The unique pixels of the pixelnumbers of the stars.
    - pix, array.           The pixelnumbers of the stars at the given Nside
    - Nside, integer.

    Returns:
    -----------
    """

    Npix = hp.nside2npix(Nside)
    print(len(pix), len(mask), Npix)

    # Allocate maps:
    p_map = np.zeros(Npix)
    q_map = np.zeros(Npix)
    u_map = np.zeros(Npix)
    r_map = np.zeros(Npix)
    sigma_p = np.zeros(Npix)
    sigma_q = np.zeros(Npix)
    sigma_u = np.zeros(Npix)

    p = pol[0]
    q = pol[1]
    u = pol[2]
    sp = sigma[0]
    sq = sigma[1]
    su = sigma[2]
    
    for i, pixel in enumerate(mask):
        ind = np.where(pix == pixel)[0]
        #print(ind, len(ind))
        p_map[pixel] = np.mean(p[ind])
        q_map[pixel] = np.mean(q[ind])
        u_map[pixel] = np.mean(u[ind])
        r_map[pixel] = np.mean(r[ind])
        sigma_p[pixel] = tools.sigma(sp[ind], len(ind))
        sigma_q[pixel] = tools.sigma(sq[ind], len(ind))
        sigma_u[pixel] = tools.sigma(su[ind], len(ind))

    return(p_map, q_map, u_map, [sigma_p, sigma_q, sigma_u], r_map)

def Corr_plot(Q,q, U, u, sq, su, C_ij, mask, name, Nside=2048, xlab=None,\
              ylab=None, title=None, save=None):
    """
    Function to calculate the correlation and make scatter plots. Get also 
    the best fit lines from lin.reg and chi^2. 
    """
    
    unit = 287.45 * 1e-6 # convert from uK_cmb to MJy/sr 

    QU = np.concatenate((Q[mask], U[mask]), axis=0)
    qu = np.concatenate((q[mask], u[mask]), axis=0)

    Rq = np.corrcoef(q[mask], Q[mask])
    print('Correlation coefficients between Q and q:')
    print(Rq)
    Ru = np.corrcoef(u[mask], U[mask])
    print('Correlation coefficients between U and u:')
    print(Ru)
    R = np.corrcoef(qu, QU)
    print('Correlation coefficients between Q,U and q,u:')
    print(R)
    P = np.sqrt(Q**2 + U**2)
    p = np.sqrt(q**2 + u**2)
    print('R_P/p=', np.mean(P/p), np.std(P/p))


    print('Calculate chi^2')
    params, std, chi2 = tools.Chi2(Q[mask], U[mask], q[mask], u[mask],\
                              C_ij[:,mask], sq[mask], su[mask])

    params = params*unit
    std = std*unit
    print(params, std)
    #sys.exit()
    x = np.linspace(np.min(qu), np.max(qu), 10)
    y = x * 5.42 * (np.sign(R[0,1])) / (unit)
    
    # lin.regression:
    slope, intercept, rval, pval, std_err = stats.linregress(qu, QU)
    print('r, p, std err (qu):', rval, pval, std_err)
    print('Slope of QU-qu: {} MJy/sr'.format(slope*unit), intercept)

    aq, bq, rq, pq, stdq = stats.linregress(q[mask],Q[mask])
    print('r, p, std err (q):', rval, pval, std_err)
    print('Slope of Q-q: {} MJy/sr'.format(slope*unit), intercept)
    au, bu, ru, pu, stdu = stats.linregress(u[mask], U[mask])
    print('r, p, std err (qu):', rval, pval, std_err)
    print('Slope of QU-qu: {} MJy/sr'.format(slope*unit), intercept)
    
    # plotting
    plt.figure()
    plt.plot(q[mask], Q[mask]*unit, '^k', label=r'$Q_s, q_v$')
    plt.plot(u[mask], U[mask]*unit, '^b', label=r'$U_s, u_v$')

    plt.plot(x, slope*x*unit + intercept*unit, '-r',\
             label='lin.reg: ${}\pm{}$ MJy/sr'.\
             format(round(slope*unit, 2), round(std_err*unit,2)))
    plt.plot(x, params[0]*x + params[1], '-g',\
             label=r'$\chi^2:$ {} MJy/sr'.format(round(params[0],3)))
    
    plt.xlabel(xlab)
    plt.ylabel(ylab + 'MJy/sr')
    plt.legend(loc=1)
    plt.title('Correlation between Q,U and q,u. Pearson R={}'.\
              format(round(R[1,0],3)))
    plt.xlim(-0.015, 0.015)
    plt.ylim(-0.1, 0.1)
    plt.savefig('Figures/correlations/planckXII2018_QUqu.png')
    
    #
    plt.figure()
    plt.errorbar(q[mask], Q[mask]*unit, xerr=sq[mask],\
                 yerr=np.sqrt(C_ij[3,mask])*287.45, fmt='none', ecolor='k',\
                 label=r'$Q_s, q_v$')
    plt.errorbar(u[mask], U[mask]*unit, xerr=sq[mask],\
                 yerr=np.sqrt(C_ij[5,mask])*287.45, fmt='none', ecolor='b',\
                 label=r'$U_s, u_v$')

    plt.plot(x, slope*x*unit + intercept*unit, '-r',\
             label='lin.reg: ${}\pm{}$ MJy/sr'.\
             format(round(slope*unit, 2), round(std_err*unit,2)))
    plt.plot(x, params[0]*x + params[1], '-g',\
             label=r'$\chi^2:$ {} MJy/sr'.format(round(params[0],3)))
    
    plt.xlabel(xlab)
    plt.ylabel(ylab + 'MJy/sr')
    plt.legend(loc=1)
    plt.title('Correlation between Q,U and q,u. Pearson R={}'.\
              format(round(R[1,0],3)))
    #plt.xlim(-0.015, 0.015)
    #plt.ylim(-0.1, 0.1)
    plt.savefig('Figures/correlations/planckXII2018_QUqu_ebar.png')
    #

    plt.figure('2d hist')
    plt.hist2d(qu, QU*unit, 100, cmap='viridis', cmin=0.1)
    #plt.plot(x, slope*x*unit + intercept*unit, '-r',\
    #         label='lin.reg: ${}\pm{}$ MJy/sr'.\
    #         format(round(slope*unit, 2), round(std_err*unit,2)))
    plt.plot(x, params[0]*x + params[1], '-r',\
             label=r'$\chi^2$: $a={}\pm{}$ MJy/sr'.\
             format(round(params[0],3), round(std[0], 3)))
    plt.plot(x, -5.42*x, 'orange', label=r'$R_{{P/p}}=5.42$MJy/sr')
    cbar = plt.colorbar(pad=0.02)
    cbar.set_label('counts', rotation=90)
    plt.xlabel(xlab)
    plt.ylabel(ylab + 'MJy/sr')
    plt.title('Correlation between Q,U and q,u. Pearson R={}'.\
              format(round(R[1,0],3)))
    plt.xlim(-0.01, 0.015)
    plt.ylim(-0.1, 0.1)
    plt.legend(loc=1)
    plt.savefig('Figures/correlations/planckXII2018_QUqu_2dhist.png')

    plt.show()
    #



def main_pl(starfile, planckfile=None, dustfile=None, Cij_file=None, \
            Nside=2048, return_data=False):
    """

    Parameters:
    -----------
    -
    -

    Returns:
    -----------
    """
    data = load_data(starfile)
    # method of RoboPol: (Raphael)
    ln = 122.93200023  # vincent
    bn = 27.12843      # vincent
    psi, diff = tools.get_theta_gal(data[:,0], data[:,1], data[:,4], aG=ln, dG=bn)
    q, sq = q_v(data[:,2], psi, data[:,3], data[:,5])
    u, su = u_v(data[:,2], psi, data[:,3], data[:,5])
    r = 1000./data[:,-1]

    theta = np.pi/2. - data[:,1]*np.pi/180.
    phi = data[:,0]*np.pi/180.
    ii = np.where(data[:,1] > 0)[0]
    print(np.min(data[ii,1]), np.min(np.abs(data[:,1])))
    pix = hp.pixelfunc.ang2pix(Nside, theta, phi, nest=False)
    Npix = hp.nside2npix(Nside)
    mask = np.unique(pix)
    print(len(pix), len(mask))
    print(Cij_file)
    p_map, q_map, u_map, sigma, r_map = make_pol_map([data[:,2]/100., q/100., u/100.],\
                                                     [data[:,3]/100., sq/100., su/100.],\
                                                     r, mask, pix, Nside=Nside)

    #plt.hist(sigma[2][mask], bins=50)
    #plt.show()
    #sys.exit()
    # Get Planck maps, IQU and dust at 353 GHz: 
    #Planck_maps = hp.fitsfunc.read_map(planckfile, field=None)
    if planckfile is not None:
        IQU_maps = smooth.read_smooth_maps(planckfile, 'IQU', 3)    # uK_cmb
    if Cij_file is not None:
        C_ij_maps = load.C_ij(Cij_file, Nside)  # K_cmb^2
        print(np.sqrt(C_ij_maps[3,mask])*287.45)
        print(sigma[1][mask])
    if dustfile is not None:
        pass
    if (planckfile is None) and (Cij_file is None):
        sys.exit()
    else:
        #print(np.shape(IQU_maps), np.shape(C_ij_maps))

        # Check if resolution of Planck maps match the resolution of star maps.
        Ns_pl = hp.get_nside(IQU_maps[0])
        print(Nside, Ns_pl)
        if Ns_pl != Nside:
            IQU = hp.ud_grade(IQU_maps, Nside, order_in='RING', order_out='RING')
            I = IQU[0]
            Q = IQU[1]
            U = IQU[2]
            C_ij = hp.ud_grade(C_ij_maps, Nside, order_in='RING', order_out='RING')
        else:
            I = IQU_maps[0]
            Q = IQU_maps[1]
            U = IQU_maps[2]
            C_ij = C_ij_maps
        #
        # Need to include reddening on q and u. Do that in the making? Large std on dpsi
        # smooth fractional pol.
        u_map = smooth.smooth_tomo_map(u_map, mask, Nside, 15)
        q_map = smooth.smooth_tomo_map(q_map, mask, Nside, 15)
        p_map = smooth.smooth_tomo_map(p_map, mask, Nside, 15)

        dPsi = tools.delta_psi(Q[mask], q_map[mask], U[mask], u_map[mask],\
                               plot=False, name='test')
        print('-------')
        print(np.mean(I[mask]/p_map[mask])*287.45*1e-6)
        print(np.median(I[mask]/p_map[mask])*287.45*1e-6)
        
        
        if return_data is True:
            #QU = np.concatenate((Q[mask],U[mask]), axis=0)
            #qu = np.concatenate((q[mask],u[mask]), axis=0)
            print(np.shape(q_map), np.shape(u_map), np.shape(mask))
            return(Q, U, q_map, u_map, C_ij, sigma, mask)
        else:
            # correlation with line fitting and R_P/p:
            Corr_plot(Q, q_map, U, u_map, sigma[1], sigma[2], C_ij, mask, 'hei',\
                      Nside=2048, xlab=r'$q_v, u_v$', ylab=r'$Q_s, U_s$',\
                      title=None, save=None)
        
    #

#########1#########2#########3#########4#########5#########6#########7#########8

starfile = 'Data/vizier_star_table.tsv'
planckfile = 'Data/IQU_Nside2048_15arcmin.h5'
#dustfile = ''
Cij_file = 'Data/Planck_Cij_353_2048_full.h5'
#main_pl(starfile, planckfile, Cij_file=Cij_file, Nside=2048)
#main_pl(starfile, planckfile, Cij_file=Cij_file, Nside=256)
