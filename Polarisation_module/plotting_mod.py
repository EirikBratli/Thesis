"""
Plotting module for the tomography checking with planck polarisation.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import h5py
import sys, time, glob, os
import scipy.optimize as spo

from astropy import units as u_
from astropy.coordinates import SkyCoord

import convert_units as cu
import tools_mod as tools

#####################################

def plot_corr(tomo_map, planck_map, name, mask, dist, Nside=2048, log=False,\
            y_lim=None, x_lim=None, xlab=None, ylab=None, title=None, save=None):
    """
    Plot the correlation between tomography and planck353.

    Parameters:
    -----------

    Return:
    -----------
    """
    path = 'Figures/tomography/correlations/'
    R = np.corrcoef(tomo_map[mask], planck_map[mask])
    print('Correlation coefficient of {}:'.format(title))
    print(R)

    x = np.linspace(0, np.max(tomo_map[mask]), 10)
    y = x * 5.42 / (287.45*1e-6) # to uK_cmb,
    # numbers form planck18, XXI and planck13,IX

    plt.figure('{}'.format(name))
    plt.plot(tomo_map[mask], planck_map[mask], '.k')
    plt.plot(x, y, '-r')
    plt.title('Correlation of {}'.format(title))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if y_lim is not None:
        plt.ylim(y_lim)
    if x_lim is not None:
        plt.xlim(x_lim)
    plt.savefig(path + '{}_{}.png'.format(save, Nside))

    plt.figure()
    plt.hexbin(tomo_map[mask], planck_map[mask], C=dist[mask], bins='log')
    #plt.plot(x, y, '-r', label='y=x*5.42 MJy/sr * (674.7 uK_cmb/MJy/sr)')
    cbar = plt.colorbar(pad=0.01)
    cbar.set_label(r'$log(r)$ [pc]')
    plt.title('Correlation of {}'.format(title))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(path + '{}_dist_color_{}.png'.format(save, Nside))

    if log is True:
        plt.figure('{} log'.format(name))
        plt.semilogy(tomo_map[mask], planck_map[mask], '.k')
        plt.xlabel('Tomography data')
        plt.ylabel('Planck 353GHz')
        plt.savefig(path + '{}_log.png'.format(save))

    #

def plot_gnom(map, lon, lat, label, mask=None, Nside=2048, unit=None, project=None):
    """
    Plotting function viewing in gnomonic projection.

    Parameters:
    -----------
    - map, array. The map to plot.
    - lon, scalar. mean position in longitude
    - lat, scalar. mean position in latitude
    - label, string.  one of Q, U, P, q, u, p.
    -
    -


    Return:
    -------
    """

    path = 'Figures/tomography/maps/'
    hp.gnomview(map, title='Polarization {}_{}'.format(label, project),\
                rot=[lon, lat], xsize=100, unit=unit)
    #hp.gnomview(map, title='Polarization {}'.format(label), rot=[lon,90-lat,180],\
    #            flip='geo', xsize=100)
    hp.graticule()
    plt.savefig(path + '{}_{}_{}.png'.format(label, project, Nside))

    ###

def plot_ratio(tomo, planck, mask, Nside=2048, save=None, label=None, name=None):
    """

    """
    fac = 5.42 / (287.45*1e-6) # to uK_cmb,
    R_P2p, tot, mean = tools.ratio_P2p(tomo, planck, mask)
    print(tot/fac, mean/fac)
    print(R_P2p[mask]/fac)
    pass
