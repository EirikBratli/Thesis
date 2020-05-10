"""
Unit convertor from Planck CMB analysis.
Convert between K_RJ to MJy/sr
or K_RJ to K_CMB
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import healpy as hp
import pandas as pd
import sys

datapath = 'Data/'
file1 = 'HFI_RIMO_R3.00.fits'
file2 = 'LFI_RIMO_R3.31.fits'

h = 6.62607004e-34  # m^2 kg / s
kB = 1.38064852e-23 # m^2 kg s^-2 K^-1
c = 299792458.      # m / s


def read_freq(path=datapath, file1=file1, file2=file2):
    """
    Read the transmission data of Planck as function of frequency.
    """

    hdul_hi = fits.open(datapath + file1)
    hdul_lo = fits.open(datapath + file2)

    m_030 = hdul_lo[4].data
    m_044 = hdul_lo[8].data
    m_070 = hdul_lo[13].data
    m_100 = hdul_hi[3].data
    m_143 = hdul_hi[4].data
    m_217 = hdul_hi[5].data
    m_353 = hdul_hi[6].data
    m_545 = hdul_hi[7].data
    m_857 = hdul_hi[8].data

    f = np.array([m_030['WAVENUMBER'], m_044['WAVENUMBER'],m_070['WAVENUMBER'],\
            m_100['WAVENUMBER'][1:], m_143['WAVENUMBER'][1:],\
            m_217['WAVENUMBER'][1:], m_353['WAVENUMBER'][1:],\
            m_545['WAVENUMBER'][1:], m_857['WAVENUMBER'][1:]])

    T = np.array([m_030['TRANSMISSION'], m_044['TRANSMISSION'],\
            m_070['TRANSMISSION'], m_100['TRANSMISSION'][1:],\
            m_143['TRANSMISSION'][1:], m_217['TRANSMISSION'][1:],\
            m_353['TRANSMISSION'][1:], m_545['TRANSMISSION'][1:],\
            m_857['TRANSMISSION'][1:]])

    f[3:] = f[3:]*1e-7*c
    return(f, T)

def Unit_converter(f, tau, func1, func2, unit1, unit2, f_ref=857.):
    """
    Call function for converting units.
    """
    #f, T = read_f857(datapath, file)
    #ii = np.where(f < 2*f_ref)[0]
    #f1 = f[1:max(ii)]
    #T1 = T[1:max(ii)]

    U = UnitConv(f, tau, func1, func2, unit1, unit2)
    return(U)

def UnitConv(f, T, dIdXi, dIdXj, unit1, unit2, f_ref=857.):
    """
    Convert units from K_RJ to f.ex K_cmb or MJy/sr.
    """
    print('Convert {} to {}'.format(unit1, unit2))
    df = (f[-1] - f[0])/(len(f)-1)
    I1 = 0.
    I2 = 0.
    for i in range(len(f)):
        I1 += T[i]*dIdXi[i]*df
        #I1 += T[i]*(2.*kB*(1e9*f[i]/c)**2) * df
        I2 += T[i]*dIdXj[i] * df
    print(I1, I2)
    U = I1/I2
    print('Convertion factor U = {} [{}/{}]'.format(U, unit2, unit1))
    return(U)

def dBdTcmb(f, Tcmb=2.7255):
    nu = f*1e9
    
    fac0 = h*nu/(kB*Tcmb**2)
    fac1 = 2.*h*nu**3 / (c**2 * (np.exp(h*nu/(kB*Tcmb)) - 1.))
    fac2 = np.exp(h*nu/(kB*Tcmb)) / (np.exp(h*nu/(kB*Tcmb)) - 1.)
    return(fac1*fac2*fac0)

def dBdTrj(f):
    nu = f*1e9
    return(2.*nu**2*kB/c**2)

def dBdTiras(f, f_ref=857.):
    return(f_ref/f)

"""
f, T = read_freq(datapath, file1, file2)
ind = np.where((f[6] > 200) & (f[6] < 600))[0]
ii = np.where(f[6] < 600)[0]

f1 = f[6][1:max(ii)]
T1 = T[6][1:max(ii)]
#f1 = f[ind]
#T1 = T[ind]

#Unit_converter(f1, T1, dBdTcmb(f1), dBdTiras(f1, 353.), 'K_cmb', '(MJy/sr) 1e-20', 353.)
Unit_converter(f1, T1, dBdTiras(f1), dBdTcmb(f1, 353.), '(MJy/sr) 1e-20', 'K_cmb', 353.)
#Unit_converter(f1, T1, dBdTrj(f1), dBdTiras(f1), 'K_rj', '(MJy/sr) 1e-20')
#Unit_converter(f1, T1, dBdTrj(f1), dBdTcmb(f1), 'K_rj', 'K_cmb')
"""
