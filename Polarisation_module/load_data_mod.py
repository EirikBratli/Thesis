"""
Module for loading data maps.
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


######################################

def load_tomographydata(file, colname):
    """
    Read the tomography file or an other .csv file.

    Parameters:
    -----------
    - file, string. The name of the file to read
    - colname, list/array. The name of the columns to read.

    Return:
    -------
    - data, ndarray. A data array with the data read.
    """

    colnum = getCols(file, colname)
    data = np.genfromtxt(file, delimiter=',', skip_header=1, usecols=colnum)
    return(data)

def getCols(file, colname):
    """
    Find the column numbers of the columns to read.
    Parameters:
    -----------
    - file, string. The name of the file to read
    - colname, list/array. The name of the columns to read.

    Return:
    -------
    - colnum, array. A array with the column numbers to read.
    """
    a = np.genfromtxt(file, delimiter=',', names=True)
    b = np.asarray(a.dtype.names)
    colnum = []
    for j in range(len(colname)):
        for i in range(len(b)):
            if colname[j] == b[i]:
                colnum.append(i)
    return(colnum)

def tomo_map(data, Nside=2048):
    """
    Make healpix map for the given Nside for the tomography data

    Parameters:
    -----------
    - data, ndarray.    All of the data array.
    - Nside, integer.   The resolution of the map to make, default is 2048

    Return:
    -------
    - p_map, array.     Array with the fractional polarization, p=sqrt(q^2+u^2)
    - q_map, array.     Array with the fractional q=Q/I polarization.
    - u_map, array.     Array with the fractional u=U/I polarization.
    - sigma_p, array.   Array with the uncertainty of p
    - sigma_q, array.   Array with the uncertainty of q.
    - sigma_u, array.   Array with the uncertainty of u.
    - pix, array.       The pixels where there exist data.
    """

    # convert to galactic coordinates:
    l, b = tools.convert2galactic(data[:,0], data[:,1])
    theta = np.pi/2. - b * np.pi/180.
    phi = l * np.pi/180.
    print(np.min(theta), np.max(theta))
    # get pixel numbers
    pix = hp.pixelfunc.ang2pix(Nside, theta, phi, nest=False)

    # polariasation rotation:
    q_gal, u_gal = tools.rotate_pol(data[:,0], data[:,1], data[:,2], data[:,4],\
                                    data[:,6])


    print(hp.pixelfunc.nside2pixarea(Nside, degrees=True))
    # Create maps
    Npix = hp.nside2npix(Nside)
    p_map = np.zeros(Npix)
    q_map = np.zeros(Npix)
    u_map = np.zeros(Npix)
    r_map = np.zeros(Npix)

    sigma_p = np.zeros(Npix)
    sigma_q = np.zeros(Npix)
    sigma_u = np.zeros(Npix)

    print(Npix, np.shape(p_map))
    print(len(np.unique(pix)))
    uniqpix = np.unique(pix)
    #l, b = hp.pix2ang(Nside, uniqpix)
    index = []
    #print(uniqpix)
    print(len(pix))
    for k, i in enumerate(uniqpix):
        ind = np.where(pix == i)[0]
        #print(ind, psi[ind])
        #psi2 = 2*np.mean(psi[ind])

        # Use mean instead??
        p = np.mean(data[ind, 2])
        q = np.mean(q_gal[ind])
        u = np.mean(u_gal[ind])

        p_map[i] = p
        q_map[i] = q #* np.sin(psi2)
        u_map[i] = u #* np.cos(psi2)

        sigma_p[i] = np.mean(data[ind, 3])
        sigma_q[i] = np.mean(data[ind, 5])
        sigma_u[i] = np.mean(data[ind, 7])
        r_map[i] = np.mean(data[ind, 8])
        #print(r_map[i], data[ind,8])

    #print(q_map[uniqpix])
    #print(np.sum(p_map==0))
    sys.exit()
    return(p_map, q_map, u_map, [sigma_p, sigma_q, sigma_u], r_map, pix)

def load_planck_map(file, p=False):
    """
    Read the .fits file containing the 353GHz planck map.

    Parameters:
    -----------
    - file, string. Name of the planck file.
    - p, bool.      If to load polarised map into 3 maps (I,Q,U). Optional.
                    Returns then 4 maps (T, P, Q, U) instead of one
    Return:
    -------
    - m353, array.  The data of the map
    """
    if p is True:
        T, Q, U, hdr = hp.fitsfunc.read_map(file, field=(0,1,2), h=True)
        #print(T)
        #print(Q)
        #print(U)
        #print(hdr)
        P = (np.sqrt(Q**2 + U**2))
        return(T, P, Q, U)
    else:
        m353, hdr = hp.fitsfunc.read_map(file, h=True)
        #print(hdr)
        return(m353)

def get_PlanckSky_chunk(map, pix, Nside=2048):
    """
    Get the planck data in the same pixels as the tomography map.

    Parameters:
    -----------
    - map, array. The planck map
    - pix, array. The pixels from the tomography map.
    - Nside, integer. The resolution of the maps.

    Return:
    -------
    - Skychunk, array. Array with planck data for the same pixels in tomography
    """

    Npix = hp.nside2npix(Nside)
    Skychunk = np.zeros(Npix)
    uniqpix = np.unique(pix)
    Skychunk[uniqpix] = map[uniqpix]
    return(Skychunk)
