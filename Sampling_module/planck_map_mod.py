"""
Module for the reading and fixing the planck maps to my need.
"""

import numpy as np
import healpy as hp
#import matplotlib.pyplot as plt
import sys, time
import convert_units as cu
#import h5py


def load_planck_map(file):
    """
    Load a planck map into an array.

    Parameters:
    -----------
    - file, string. The filename of the map, need to be an .fits file.

    Returns:
    -----------
    - m, array. The map read in from file.
    """
    m, hdr = hp.fitsfunc.read_map(file, h=True)
    #print(hdr)
    return(m)

def ChangeMapUnits(files, fref):
    """
    Change the units of the maps provides form PLA to K_RJ. The data come in
    units K_cmb or MJy/sr.

    Parameters:
    -----------
    - files, list/array.    Sequence with the file names of the planck maps as
                            strings.
    - fref. list/array.     The reference frequencies of each map.

    Returns:
    -----------
    - out_maps, ndarray.    List with each of the maps read in and converted to
                            units of K_RJ.
    """

    f, tau = cu.read_freq()
    U_rj = np.zeros(len(f))
    out_maps = []
    # use only values close to f_ref since zero else.
    for j in range(3, len(f)):
        ind = np.where((f[j] > fref[j]/2.) & (f[j] < 2*fref[j]))[0]
        f_temp = f[j][ind]
        f[j] = f_temp


    for i in range(len(f)):
        # maps are sorted after frequency, from low to high.
        print('Load frequency map for {} GHz'.format(fref[i]))
        in_map = load_planck_map(files[i])

        dBdTrj = cu.dBdTrj(f[i])
        #print(i, dBdTrj)
        if i < 7:
            # convert K_cmb to K_RJ
            dBdTcmb = cu.dBdTcmb(f[i])
            dBdTcmb = np.nan_to_num(dBdTcmb)
            #print('', dBdTcmb)
            U_rj[i] = cu.UnitConv(f[i], tau[i], dBdTcmb, dBdTrj,\
                                    'K_CMB', 'K_RJ', fref[i])
        else:
            # Convert MJy/sr to K_RJ
            dBdTiras = cu.dBdTiras(f[i], f_ref=fref[i])
            dBdTiras = np.nan_to_num(dBdTiras)
            #print('', dBdTiras)
            U_rj[i] = cu.UnitConv(f[i], tau[i], dBdTiras, dBdTrj,\
                                    'MJy/sr 1e-20', 'K_RJ', fref[i]) * 1e-20
        #
        #print(U_rj[i])
        out_map = in_map*U_rj[i]
        print('-----------')
        out_maps.append(out_map)
    #
    #print(U_rj)
    return(out_maps)


def fix_resolution(map, new_Nside, ordering='RING'):
    """
    Parameters:
    -----------
    - map, array.           The map upgrade/degrade resolution of.
    - new_Nside. scalar.    The resolution to fix the map to.
    - ordering, string.     Which Healpix ordering to use, default is RING.

    Return:
    -----------
    - m, array.             The new map with new resolution.
    """

    print('Fix resolution to Nside={}'.format(new_Nside))
    m = hp.pixelfunc.ud_grade(map, new_Nside, order_in=ordering,\
                            order_out=ordering)
    return(m)

def remove_badpixel(maps, Npix=1, val=1e6):
    """
    Function to handle pixels with extrem values, much larger/smaller than all
    other pixel values. Fixes the bad pixel values to None.

    Parameters:
    -----------
    - maps, list/seq.   A list with maps.
    - val, scalar.      The +/- limit of the pixel values to be more extrem of
                        if to reduce.

    Return:
    -----------
    - maps, list/seq.   The updated list of maps.
    - index, dict.      The indices of the bad pixels, comes with
                        key = map number (0,1,2...) and array with the indices.
    """
    print('Remove bad pixels')
    index = {}
    mask = np.full(Npix, True, dtype=bool)
    new_maps = []#np.zeros((9, Npix))
    for i in range(len(maps)):
        ind_low = np.where(maps[i] < -val)[0]
        ind_hi = np.where(maps[i] > val)[0]
        #print(len(ind_low), len(ind_hi))
        if (len(ind_hi) == 0) and (len(ind_low) == 0):
            #new_maps[i] = maps[i]
            pass

        else:
            #a = maps[i] > -val # masking the map.
            #key = i
            #mask[key] = a
            ind = np.empty(0)
            ind = np.append(ind_low, ind_hi)
            mask[ind] = False

            key = '{}'.format(i)
            index[key] = ind
            print(i, ind, index)
            #m = np.delete(maps[i], ind)
            m = maps[i]
            m[ind] = None

            #print(len(m), len(maps[i]))
            #print(len(m), len(maps[i]))
        #
    print('Removed in "map: pixels"')
    print(index)
    print(mask)
    return(maps, index, mask)
