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

def select_stars(l_in, b_in, lim=0.16):
    """
    Selcect stars according to distance or position. 
    Input: 
    - The center coordinated of the region (lon, lat) in degrees.
    - lim of the region to select in radi from center, in degrees.
    Return a bool array with the selected stars
    """
    from astropy.io import ascii
    dt = ascii.read('Data/total_tomography.csv')
    
    stars = SkyCoord(ra=dt['ra'], dec=dt['dec'],\
                     frame='icrs', unit='deg')
    
    # Define centre of circle to invesigate
    target_center = SkyCoord(l=l_in,b= b_in, frame='galactic', unit='deg')
    
    # Select stars within the limit of the centre
    target_cond = stars.separation(target_center).deg < lim
    return(target_cond)

def tomo_map(data, Nside=2048, starsel='all'):
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
    
    # Select stars
    if starsel == 'all':
        print('Use all areas')
        pass
    elif starsel == '1cloud': 
        ii = select_stars(103.9, 21.97, 0.16) # 1-cloud region
        data = data[ii,:]
        #ii = np.where(in 1 cloud)[0]     # 1 cloud region
    elif starsel == '2cloud':
        ii = select_stars(104.08, 22.31, 0.16) # 2 cloud region
        data = data[ii,:]
    print(np.shape(data))
    jj = np.where(data[:,10] > 360)[0]
    data = data[jj,:] # remove close stars, use r > 360 pc
    print(Nside, np.shape(data))
    #sys.exit()
    
    # convert to galactic coordinates:
    l, b = tools.convert2galactic(data[:,0], data[:,1])
    theta = np.pi/2. - b * np.pi/180.
    phi = l * np.pi/180.
    #print(np.min(theta), np.max(theta))

    # get pixel numbers
    pix = hp.pixelfunc.ang2pix(Nside, theta, phi, nest=False)

    # polariasation rotation (IAU convention):
    print('Rotate polarisation angle from equitorial to galactic.')
    q_gal, u_gal = tools.rotate_pol(data[:,0], data[:,1], data[:,2],\
                                    data[:,6], data[:,8], data[:,4])
    # correct for extinction:
    correction = tools.extinction_correction(l, b, data[:,10])
    q_gal = q_gal*correction
    u_gal = u_gal*correction
    j = np.where(u_gal == np.max(u_gal))[0]
    print(j, u_gal[j], l[j], b[j], data[j,10])

    q_err = data[:,7]*correction
    u_err = data[:,9]*correction
    
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
    sigma_psi = np.zeros(Npix)
    print(Npix, np.shape(p_map))
    print(len(np.unique(pix)))
    uniqpix = np.unique(pix)
    #l, b = hp.pix2ang(Nside, uniqpix)
    index = []
    #print(uniqpix)
    print(len(pix))
    for k, i in enumerate(uniqpix): # have a x arcmin beam instead to get the mean?
        ind = np.where(pix == i)[0]

        p_map[i] = np.mean(data[ind, 2])
        q_map[i] = np.mean(q_gal[ind])
        u_map[i] = np.mean(u_gal[ind])

        sigma_p[i] = tools.sigma_x(data[ind, 3], len(ind)) #np.mean(data[ind, 3])
        sigma_q[i] = tools.sigma_x(q_err[ind], len(ind)) #np.mean(data[ind, 7])
        sigma_u[i] = tools.sigma_x(u_err[ind], len(ind)) #np.mean(data[ind, 9])
        sigma_psi[i] = tools.sigma_x(data[ind, 5], len(ind))
        r_map[i] = np.mean(data[ind, 10])

        #print(r_map[i], data[ind,8])

    #print(q_map[uniqpix])
    print(len(u_map))
    #sys.exit()
    return(p_map, q_map, u_map, [sigma_p,sigma_q,sigma_u,sigma_psi], r_map, pix)

def pix2star_tomo(data, Nside, starsel='all'):
    """
    Method where the pixels are asigned to a star. Remove stars closer 
    than 360 pc, since not polarised.
    """
    # Select stars
    if starsel == 'all':
        print('Use all areas')
        pass
    elif starsel == '1cloud': 
        ii = select_stars(103.9, 21.97, 0.16) # 1-cloud region
        data = data[ii,:]
        #ii = np.where(in 1 cloud)[0]     # 1 cloud region
    elif starsel == '2cloud':
        ii = select_stars(104.08, 22.31, 0.16) # 2 cloud region
        data = data[ii,:]
    print(np.shape(data))
    jj = np.where(data[:,10] > 360)[0]
    data = data[jj,:] # remove close stars, use r > 360 pc
    print(Nside, np.shape(data))
    #sys.exit()
    
    # convert to galactic coordinates:
    l, b = tools.convert2galactic(data[:,0], data[:,1])
    theta = np.pi/2. - b * np.pi/180.  # in healpix
    phi = l * np.pi/180.
    #print(np.min(theta), np.max(theta))

    # get pixel numbers
    #pix = hp.pixelfunc.ang2pix(Nside, theta, phi, nest=False)

    # polariasation rotation (IAU convention):
    print('Rotate polarisation angle from equitorial to galactic.')
    q_gal, u_gal = tools.rotate_pol(data[:,0], data[:,1], data[:,2],\
                                    data[:,6],data[:,8], data[:,4])
    q_err, u_err = tools.rotate_pol(data[:,0], data[:,1], data[:,3],\
                                    data[:,7], data[:,9], data[:,4])
    q_err = data[:,7]
    u_err = data[:,9]
    
    p_gal = np.sqrt(q_gal**2 + u_gal**2)
    sigma = [data[:,3], q_err, u_err]
    r = data[:,10]
    pix_stars = hp.ang2pix(Nside, theta, phi)
    print(pix_stars)
    print(len(pix_stars))
    return(p_gal, q_gal, u_gal, sigma, r, pix_stars)

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
        #T, Q, U, hdr = hp.fitsfunc.read_map(file, field=(0,1,2), h=True)
        data = hp.fitsfunc.read_map(file, field=None, h=True)
        #print(T)
        #print(Q)
        print(np.shape(data))
        #print(data)
        #print(hdr)
        #P = (np.sqrt(Q**2 + U**2))
        return(data[0:3])
    else:
        #m353, hdr = hp.fitsfunc.read_map(file, h=True)
        m353, hdr = hp.fitsfunc.read_map(file, field=None, h=True)
        print(hdr)
        print(np.shape(m353))
        return(m353)

def load_C_ij(file, outfile):
    """
    Read in the uncertainty of Planck 353GHz, and write to a file only containing 
    the uncertainties. They comes as the covariance elements (C_ij = sigma^2): 
    C_II, C_QI, C_UI, C_QQ, C_QU, C_UU.
    """
    data = hp.fitsfunc.read_map(file, field=None, h=True)
    print(data[-1])
    print(np.shape(data[0]))
    data = data[0]
    C_ij = data[4:10]
    print(np.shape(C_ij))
    print(C_ij)
    f = h5py.File(outfile, 'w')
    f.create_dataset('C_ij', data=C_ij)
    f.close()
    
def C_ij(file, Nside):
    """
    Load the uncertainty file and set new resolution.
    """
    
    Nside_old = 2048
    C_ij_in = tools.Read_H5(file, 'C_ij')
    C_ij_new = hp.ud_grade(C_ij_in, Nside, order_in='RING', order_out='RING',\
                           power=2)
    
    return(C_ij_new)

#fin = '/mn/stornext/u3/hke/bratli/SRoll20_SkyMap_353psb_full.fits'
#fout = 'Data/Sroll_Cij_353_2048_full.h5'
#load_C_ij(fin,fout)

def load_tau():
    """
    Load reddening and compute tau
    """
    pass

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

#load_C_ij('Data/HFI_SkyMap_353-psb_2048_R3.01_full.fits')
