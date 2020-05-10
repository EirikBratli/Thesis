"""
Module for intensity model for CMB foreground components.
"""

import numpy as np
#import healpy as hp
#import matplotlib.pyplot as plt
import sys, time
#import h5py


def Model(nu, b=3., T=25., beta_d=1.5, A_cmb=12., A_s=0.1, beta_s=-2.):
    """
    Make the intensity model 'I_m = I_d + n' for a given frequency 'nu'.
    Intensity given as a modified blackbody spectrum. Make a case where Return
    an array and a case returning a scalar.

    Parameters:
    -----------
    - nu, array.        The frequencies to iterate over
    - b, scalar.        Scaling of the dust amplitude
    - T, scalar.        Dust temperature
    - beta_d, scalar.   spectral index of the dust
    - A_cmb, scalar.    Amplitude of the CMB
    - A_s, scalar.      Amplitude of the synchrotron radiation
    - beta_s, scalar.   spectral index of the synchrotron radiation.

    Return:
    -----------
    I_model, array. The simulated intensity maps for the different frequencies
    """


    I_dust = MBB(nu, b, T, beta_d)
    I_cmb = I_CMB(nu, A_cmb)
    I_s = I_sync(nu, A_s, beta_s)

    I_model = I_dust + I_cmb + I_s
    return(I_model)


def MBB(nu, b=3., T=25., beta=1.5, nu_d=353.):
    """
    Make the modified Planck spectrum of eq.1 in Planck Collaboration 2013 XII.

    Parameters:
    -----------
    nu, scalar.     Frequency in GHz
    b, scalar.      scale factor from A_cloud to A_dust
    T, scalar.      Brightness temperature in K
    beta, scalar.   spectral Index
    A, scalar.      The amplitude of the clouds, default=10.
    nu_d, scalar    The filter frequency to evaluate at, default is 353 GHz.

    Return:
    -----------
    B, array.      The intensity of the modified blackbody
    """

    A = 10.
    h = 6.62607004e-34  # m^2 kg / s
    kB = 1.38064852e-23 # m^2 kg s^-2 K^-1
    c = 299792458.       # m / s
    factor = h*1e9/kB
    #print(b, T, beta)
    freq = (nu/nu_d)**(beta+1.)         # shape of nu and beta
    expo1 = np.exp(factor*nu_d/T) - 1.  # shape of T
    expo2 = np.exp(factor*nu / T) - 1.  # shape of nu and T

    B = b*A*freq*expo1/expo2              # shape of A and nu
    return(B)

def I_sync(nu, A_s=0.1, beta_s=-2., nu_0=408.):
    """
    Calculate the intensity of synchrotron radiation, sampling the amplitude and
    power, nu_0 is set to the nu_0 value in table 4, Plack Collaboration 2015 X.

    Parameters:
    -----------
    - nu, array.        The frequencies to iterate over
    - A_s, scalar.      Amplitude of the synchrotron radiation
    - beta_s, scalar.   Spectral index of the synchrotron radiation.
    - nu_0, scalar.     Scaling frequency.

    Return:
    -----------
    - synchrotron intensity, array.
    """

    return(A_s*(nu/nu_0)**beta_s)

def I_CMB(nu, A_cmb=12., T_cmb=2.7255, nu0=100.):
    """
    Calculate the intensity of the CMB. Sample the amplitude

    Parameters:
    -----------
    - nu, array.        The frequencies to iterate over
    - A_cmb, scalar.    Amplitude of the CMB

    Return:
    -----------
    - I, array.         CMB intensity
    """
    #print(nu)
    h = 6.62607004e-34  # m^2 kg / s
    kB = 1.38064852e-23 # m^2 kg s^-2 K^-1
    x = h*nu*1e9/(kB*T_cmb)
    x0 = h*nu0*1e9/(kB*T_cmb)
    norm = (np.exp(x0) - 1.)**2 / (x0**2*np.exp(x0))

    I = A_cmb * (x**2*np.exp(x)) / ((np.exp(x) - 1.)**2)
    #print(I*norm, norm)
    return(I * norm)
