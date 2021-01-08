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

def load_tomographydata(file, colname, delimiter=','):
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

    if file == 'Data/total_tomography_2.csv':
        delimiter = ';'
    else:
        delimiter = delimiter

    colnum = getCols(file, colname, delimiter)
    data = np.genfromtxt(file, delimiter=delimiter, skip_header=1,\
                         usecols=colnum)
    print(data[0,:])
    return(data)

def getCols(file, colname, delimiter):
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
    a = np.genfromtxt(file, delimiter=delimiter, names=True)
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

def tomo_map(data, Nside=2048, starsel='all', part='all', distcut=360):
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
    jj = np.where(data[:,11] > 360)[0]
    cut_intr = np.logical_and(data[:,4] <= 0, data[:,4]) 
    data = data[cut_intr,:] # use stars with negative evpa
    print(np.shape(data))

    if part == 'LVC':
        if len(distcut) == 2:
            print('min and max distance for LVC', distcut)
            clean = np.logical_and(data[:,11] > distcut[0],\
                                   data[:,11] < distcut[1])
        else:
            print('LVC for dist >', distcut)
            clean = np.logical_and(data[:,11] > distcut[0], data[:,11])
    else:
        clean = np.logical_and(data[:,11] > distcut[0], data[:,11]) 
   
    data = data[clean,:] # remove close stars, use r > 360 pc
    print(np.shape(data))    
    # remove pol.angle outliers:
    mean = np.mean(data[:,4])#*180/np.pi
    sigma3 = 2.5*np.std(data[:,4])#*180/np.pi
    clean_3sigma = np.logical_and(data[:,4], data[:,4]-data[:,5] < mean+sigma3)
    data = data[clean_3sigma,:]
    print(np.shape(data))
    #s2n = data[:,2]/data[:,3] >= 3
    #data = data[s2n,:]
    #print(np.shape(data), '-')

    # convert to galactic coordinates:
    l, b = tools.convert2galactic(data[:,0], data[:,1])
    theta = np.pi/2. - b * np.pi/180.
    phi = l * np.pi/180.
    #print(np.min(theta), np.max(theta))
    print(np.shape(data))

    # get pixel numbers
    pix = hp.pixelfunc.ang2pix(Nside, theta, phi, nest=False)
    
    # make cut regarding IVC, have only stars not affected by the IVC
    print('Remove stars affected by the IVC')
    if part != 'none':
        IVC_cut = tools.IVC_cut(pix, data[:,11], distcut, Nside=Nside,\
                                clouds=part)
        data = data[IVC_cut,:]
        pix = pix[IVC_cut]
    else:
        pass

    #cut_max_d = np.logical_and(data[:,11] < 1000, data[:,11])
    #data = data[cut_max_d,:]    
    #pix = pix[cut_max_d]

    #sys.exit()

    # Debias p, since > 0.
    pmas = tools.MAS(data[:,2], data[:,3])

    # polariasation rotation (IAU convention):
    # compute pol. in galactic coordinates of q and u.
    print('Rotate polarisation angle from equitorial to galactic.')
    q_gal, u_gal, theta_gal = tools.rotate_pol(data[:,0], data[:,1],\
                                               pmas, data[:,6],\
                                               data[:,8], data[:,4])
    sq_gal, su_gal = tools.error_polangle(pmas, data[:,3],\
                                          theta_gal, np.radians(data[:,5]))

    # correct for extinction:
    #correction = tools.extinction_correction(l, b, data[:,10])
    #q_gal = q_gal*correction
    #u_gal = u_gal*correction
    q_err = sq_gal#*correction
    u_err = su_gal#*correction
    
    psi = 0.5*np.arctan2(-u_gal, q_gal)
    
    # Create maps
    Npix = hp.nside2npix(Nside)
    p_map = np.full(Npix, hp.UNSEEN)
    q_map = np.full(Npix, hp.UNSEEN)
    u_map = np.full(Npix, hp.UNSEEN)
    r_map = np.full(Npix, hp.UNSEEN)
    psi_map = np.full(Npix, hp.UNSEEN)
    sigma_p = np.full(Npix, hp.UNSEEN)
    sigma_q = np.full(Npix, hp.UNSEEN)
    sigma_u = np.full(Npix, hp.UNSEEN)
    sigma_psi = np.full(Npix, hp.UNSEEN)
    err_psi = np.full(Npix, hp.UNSEEN)

    print(len(np.unique(pix)), len(pix))
    uniqpix = np.unique(pix)
    index = []
    for k, i in enumerate(uniqpix): 
        ind = np.where(pix == i)[0]
     
        q, qerr = tools.weightedmean(q_gal[ind], q_err[ind])
        u, uerr = tools.weightedmean(u_gal[ind], u_err[ind])
        p, perr = tools.weightedmean(pmas[ind], data[ind,3])
        #psi2, psierr = tools.weightedmean(psi[ind]*180/np.pi, data[ind,5])
            
        p_map[i] = p #np.mean(data[ind, 2])
        q_map[i] = q #np.mean(q_gal[ind])
        u_map[i] = u #np.mean(u_gal[ind])
            
        sigma_p[i] = perr #tools.sigma_x(data[ind, 3], len(ind)) 
        sigma_q[i] = qerr #+ np.std(q_gal[ind]) #tools.sigma_x(q_err[ind], len(ind))#
        sigma_u[i] = uerr #+ np.std(u_gal[ind]) #tools.sigma_x(u_err[ind], len(ind))#
        a = np.std(u_gal[ind])#tools.sigma_x(u_err[ind], len(ind))

        sigma_psi[i] = tools.sigma_x(data[ind, 5], len(ind)) #np.std(psi[ind]) 
        r_map[i] = np.mean(data[ind, 10])
     
    #
    print(u_map[uniqpix])
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

    jj = np.where(data[:,11] > 360)[0]
    clean = np.logical_and(data[:,11] > 360, data[:,11])
    #print(clean)
    data = data[clean,:] # remove close stars, use r > 360 pc
    # remove pol.angle outliers:
    mean = np.mean(data[:,4])#*180/np.pi
    sigma3 = 2.5*np.std(data[:,4])#*180/np.pi
    clean_3sigma = np.logical_and(data[:,4], data[:,4]-data[:,5] < mean+sigma3)
    data = data[clean_3sigma,:]
    
    #jj = np.where(data[:,10] > 360)[0]
    #data = data[jj,:] # remove close stars, use r > 360 pc
    print(Nside, np.shape(data))
    #sys.exit()
    
    # convert to galactic coordinates:
    l, b = tools.convert2galactic(data[:,0], data[:,1])
    theta = np.pi/2. - b * np.pi/180.  # in healpix
    phi = l * np.pi/180.

    # get pixel numbers
    pix = hp.pixelfunc.ang2pix(Nside, theta, phi, nest=False)

    # clean for IVC:
    IVC_cut = tools.IVC_cut(pix, data[:,11], distcut=900, Nside=Nside, clouds='LVC')
    data = data[IVC_cut,:]    
    theta = theta[IVC_cut]
    phi = phi[IVC_cut]
    
    # Debias p, since > 0.
    pmas = tools.MAS(data[:,2], data[:,3])
    # polariasation rotation (IAU convention):
    print('Rotate polarisation angle from equitorial to galactic.')
    q_gal, u_gal, evpa = tools.rotate_pol(data[:,0], data[:,1], data[:,2],\
                                    data[:,6],data[:,8], data[:,4])
    #q_err, u_err, evpa0 = tools.rotate_pol(data[:,0], data[:,1], data[:,3],\
    #                                data[:,7], data[:,9], data[:,4])
    sq_gal, su_gal = tools.error_polangle(pmas, data[:,3],\
                                          evpa, np.radians(data[:,5]))
    #print(len(q_gal), len(u_gal), len(evpa), '.')
    # correct for extinction:
    #correction = tools.extinction_correction(l, b, data[:,10])
    #q_gal = q_gal*correction
    #u_gal = u_gal*correction
    j = np.where(u_gal == np.max(u_gal))[0]
    #print(j, u_gal[j], l[j], b[j], data[j,10], data[j,4])
    #print(np.mean(u_gal), np.mean(data[:,4]))
    q_err = sq_gal#*correction
    u_err = su_gal#*correction

    #q_err = data[:,7]
    #u_err = data[:,9]
    
    p_gal = np.sqrt(q_gal**2 + u_gal**2)
    sigma = [data[:,3], q_err, u_err]
    r = data[:,10]
    pix_stars = hp.ang2pix(Nside, theta, phi)
    #print(pix_stars)
    #print(len(pix_stars), len(u_gal))
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
#f = 'Data/HFI_SkyMap_217_2048_R3.01_full.fits'   
#load_C_ij(f, 'Data/Planck_Cij_217_2048_full.h5')
#load_C_ij(f, 'Data/Planck_Cij_143_2048_full.h5')

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
