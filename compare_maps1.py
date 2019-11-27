"""
This program compares two maps.
Usually the simulated extinciton maps at a given distance from the sun with a
thermal dust map from Planck (857 GHz). The maps need to be smoothed and
normalized to same magnitude order. Then multipied together and normalized to
unitless.
- smooth, infiles are unsmoothed
- normalize both maps to [0,1]
- multipy m1 and m2
"""


import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import scipy.optimize as spo
import sys, time
import h5py
import Planck_maps

file1 = 'Ag_map_r10000_Nside16.h5'
file2 = 'HFI_SkyMap_857-field-Int_2048_R3.00_full.fits'

def Read_H5(file, name):
    f = h5py.File(file, 'r')
    data = np.asarray(f[name])
    f.close()
    return(data)

def get_maps(Ag_file, dust_file):
    """
    Read in extinction map, and dust map. Then smooth the maps
    """

    Ag_map = Read_H5(Ag_file, 'Ag')
    print(np.shape(Ag_map))
    # get Nside og extinction map
    Nside = hp.pixelfunc.get_nside(Ag_map)
    # get dust map with same Nside as extinction map
    dust_map = Planck_maps.Thermal_map(dust_file, Nside)
    print(np.shape(dust_map))
    # smooth the maps:
    Ag_smap = Planck_maps.smoothing(Ag_map, Nside)
    print(np.max(Ag_smap))
    dust_smap = Planck_maps.smoothing(dust_map, Nside)
    R = np.corrcoef([Ag_smap, dust_smap])
    print(R)


    fac = minimize(Ag_smap, dust_smap)
    print(fac)
    plot_residuals(fac, Ag_smap, dust_smap)

    """
    hp.mollview(np.log10(Ag_smap), title='Smoothed log Ag map')
    plt.savefig('Figures1/smoothed_Ag_map_log.png')
    hp.mollview(np.log10(dust_smap), title='Smoothed log dust map')
    plt.savefig('Figures1/Smoothed_dust_map_log.png')
    """

    print(np.max(dust_smap))
    return(Ag_smap, dust_smap)

def correlation(Ag_map, d_map):
    """
    Multiply mormalized maps together
    """
    print(np.max(Ag_map), np.max(d_map))
    corr = Ag_map*d_map/(np.max(Ag_map*d_map))
    print(np.max(corr), len(corr))
    plot_corr_map(corr)

    # get power spectrum:
    Ag_map1 = Ag_map/np.max(Ag_map)
    d_map1 = d_map/np.max(d_map)
    print(np.max(Ag_map1), np.max(d_map1))
    cl_Ag, el_Ag = Planck_maps.power_spectrum(Ag_map1)
    cl_d, el_d = Planck_maps.power_spectrum(d_map1)


    plot_powspec(cl_Ag, el_Ag, cl_d, el_d)
    plot_Ag_dust(Ag_map, d_map)

    plt.figure()
    plt.semilogy(Ag_map1, corr, '.b', label='Corr vs Ag')
    plt.semilogy(d_map1, corr, '.r', label='Corr vs dust')
    plt.legend(loc=4)
    plt.savefig('Figures1/corr_vs10000.png')


def minimize(Ag_map, d_map):
    """
    minimize (d_map - factor*Ag_map)**2
    """
    def min_func(fac, Ag_map=Ag_map, d_map=d_map):
        return(np.sum(((d_map - fac*Ag_map)/np.abs(d_map)))**2)

    minimize = spo.fmin_powell(min_func, x0=100)#, args=(Ag_map, d_map))

    return(minimize)

def plot_residuals(fac, Ag_map, d_map):
    res = d_map - fac*Ag_map
    hp.mollview(res, title='Residuals')
    plt.savefig('Figures1/Residuals10000.png')


def plot_corr_map(corr):
    Nside = hp.pixelfunc.get_nside(corr)
    hp.mollview(corr, title=r'Correlation between $A_G$ and dust, Nside={}'.format(Nside))
    plt.savefig('Figures1/correlation_map10000.png')

    hp.mollview(np.log10(corr), title=r' log of Correlation between $A_G$ and dust, Nside={}'.format(Nside))
    plt.savefig('Figures1/correlation_map_log10000.png')

def plot_Ag_dust(Ag_map, dust_map):
    plt.figure()
    plt.semilogy(Ag_map, dust_map, '.b')
    plt.xlabel('extinction')
    plt.ylabel('thermal dust')
    plt.savefig('Figures1/extinciton_vs_dust10000.png')

    plt.figure()
    plt.semilogy(Ag_map/np.max(Ag_map), dust_map/np.max(dust_map), '.b')
    plt.xlabel('extinction')
    plt.ylabel('thermal dust')
    plt.savefig('Figures1/extinciton_vs_dust_normalised10000.png')

def plot_powspec(cl_Ag, el_Ag, cl_d, el_d):
    plt.figure()
    plt.semilogy(el_Ag, el_Ag*(el_Ag + 1)*cl_Ag, '-b', label=r'$P_{{A_G}}(l)$')
    plt.semilogy(el_d, el_d*(el_d + 1)*cl_d, '-r', label=r'$P_d(l)$')
    plt.xlabel(r'$l$')
    plt.ylabel(r'$l(l+1)C_l$')
    plt.legend(loc=3)
    plt.savefig('Figures1/power_spectrums_normed10000.png')


Agmap, dmap = get_maps(file1, file2)
correlation(Agmap, dmap)
plt.show()
