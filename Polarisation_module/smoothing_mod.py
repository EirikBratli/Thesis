"""
Module for smoothing of maps. Smoothes, writes and reads smoothed maps.
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

##################################################

def smoothing(map, Nside, iter=3, res=15):
    """
    Function to smooth a map to 15 arcseconds.
    Parameters:
    -----------
    - map, array.       The map to smooth
    - Nside, integer.   The resolution of the map.
    - iter, integer.    The number of iterations the smoothing does. default = 3
    Return:
    - smap, array.      The smoothed map
    """
    FWHM = (np.sqrt(float(res)**2 - 5.**2)/60.) * (np.pi/180) # 15 arcmin
    smap = hp.sphtfunc.smoothing(map, fwhm=FWHM, iter=iter)
    return(smap)

def smooth_tomo_map(in_map, mask, Nside=2048, res=15):
    """
    Smooth the tomography maps using realspace smoothing.

    Parameters:
    -----------
    - in_map, array.    The tomography map to be smoothed in real space. Have
                        the same length as mask
    - mask, array.      The unique pixels in the tomography map.
    - l, array.         longitude angles in degrees, same length as mask
    - b, array.         latitude angles in degrees, same length as mask
    - Nside, integer.   The resolution of the map. default is 2048

    Return:
    -----------
    - smap. array.      The smoothed map with length Npix
    """
    Npix = hp.nside2npix(Nside)
    N_map = np.zeros(Npix)
    smap = np.zeros(Npix)

    b, l = hp.pix2ang(Nside, mask)
    #print((l)*180./np.pi, 90-(b)*180./np.pi)
    #sys.exit()
    phi = l
    theta = np.pi/2. - b
    #print(phi*180/np.pi)
    s = (float(res)/60.)*(np.pi/180.)/np.sqrt(8.*np.log(2.))
    #print(s)
    #print(mask)
    for p in mask:

        # smoothing:
        #print(phi*180/np.pi)
        r = dist2neighbours(p, phi, theta, Nside)
        #print(r)
        #print(in_map[p])
        if len(in_map) == len(mask):
            I1 = np.sum(in_map * np.exp(-0.5*(r/s)**2))
        else:
            I1 = np.sum(in_map[mask] * np.exp(-0.5*(r/s)**2))
        I2 = np.sum(np.exp(-0.5*(r/s)**2))

        smap[p] = I1/I2
        #print(I1, I2, smap[p])
        #sys.exit()
    #
    #print(smap[mask])
    #sys.exit()
    return(smap)

def smooth_maps(maps, Nside, iterations=3, res=15):
    """
    Smooth a list of maps.

    Parameters:
    -----------
    - maps, list.           A list of maps to be smoothed.
    - Nside, integer.       The resolution of the maps.
    - iterations, integer.  The number of iterations the smoothing happens.

    Return:
    -----------
    - out_maps, list.       A list of smoothed maps.
    """

    out_maps = []
    t0 = time.time()
    for i in range(len(maps)):
        t1 = time.time()
        print('Smooth map {}'.format(i))
        m = smoothing(maps[i], Nside, iter=iterations, res=res)
        out_maps.append(m)
        t2 = time.time()
        print('smoothing time map {}: {}s'.format(i, t2-t1))

    t3 = time.time()
    print('Total smoothing time: {}'.format(t3-t0))
    return(out_maps)

def Write_smooth_map(maps, name, Nside=2048, res=15, iterations=3):
    """
    Write a smoothed map to file, using only the tomography pixels

    Parameters:
    -----------
    - maps, list.           A list of maps to be smoothed.
    - name, sting.          The data name/column name of the date to be written
                            to file.
    - Nside, integer.       The resolution of the maps.
    - res, integer.         The smoothing resolution in arcmin. default 15 arcmin
    - iterations, integer.  The number of iterations the smoothing does.
                            Default is 3
    """
    print(len(maps))
    if len(maps) == 4:
        smap = smooth_maps(maps, Nside, iterations, res=res)
        pl_maps = smap[0:3]
        dust_map = smap[-1]

        print('Writing planck maps:')
        #hp.fitsfunc.write_map('Data/{}_Nside{}_smoothed{}arcmin.fits'.\
        #                    format(name[1], Nside, res), pl_maps)
        write_H5(pl_maps, name[0], Nside, res=res)
        print('Writing dust maps')
        #hp.fitsfunc.write_map('Data/{}_smoothed{}arcmin.fits'.\
        #                    format(name[2], Nside, res), dust_map)
        write_H5(dust_map, name[-1], Nside, res=res)

    else:
        print('...')
        smap = smooth_maps(maps, Nside, iterations)
        print('Writing {} map'.format(name[0]))
        write_H5(smap, name[0], Nside, res=res)
        #hp.fitsfunc.write_map('Data/{}_Nside{}_smoothed{}arcmin.fits'.\
        #                    format(name[0], Nside, res), smap)

def read_smooth_maps(filename, name, shape):#(names, shape, Nside=2048, resolution=15):
    """
    Read files containing smoothed maps. Require a certain ending of the file
    names to read, '_smoothed{}arcmin.h5' where {} is the smoothing resolution.

    Parameters:
    -----------
    - names, list.          The names of the columns in the different files.
    - Nside, integer.       Map resolution set to default 2048.
    - resolution, integer.  The smoothing resolution, default is 15 arcmin.

    Return:
    -----------
    smaps, list.    A list of the smoothed maps read in, last list element is a
                    string describing the content: 'all', 'IQU', 'tomo', 'dust'.
    """
    path = 'Data/Smooth_maps/'
    print('Read in smoothed maps in units uK_cmb')
    if (name == 'IQU') or (name == 'IQU_planck'):
        if (shape == 3):
            # read IQU file
            I, Q, U = Read_H5(filename, name, shape)
            smaps = [I, Q, U]

    elif (name == 'dust') or (name == 'I_dust'):
        if (shape == 1):
            dust = Read_H5(filename, name, shape)
            smaps = [dust*1e-6]  # dust files are in K_cmb not uK_cmb

    else:
        print('Wrong column name ({IQU, IQU_planck}, {dust, I_dust}) or shape (1, 3)')
        sys.exit()

    return(smaps)

    # find smoothed files
    """
    print(os.path.abspath(os.curdir))
    smooth_files = glob.glob(path + '*{}arcmin.h5'.format(resolution))
    print(smooth_files, len(smooth_files))
    print(names[0])
    smaps = []

    if len(smooth_files) == 3:
        for file in smooth_files:
            name = file.split('/')[-1]
            args = name.split('_')
            arg = args[0]
            Ns = args[1]
            print(arg)
            # load smoothed files

            if arg == 'IQU':
                print('planck')
                #I, P, Q, U = load_planck_map(smooth_pl_file, p=True)
                I, Q, U = Read_H5(file, names[1], shape=3)

            elif arg == 'tomo':
                print('tomo pass, need smoothing in real space')
                p = ['p']
                q = ['q']
                u = ['u']
                #i, p, q, u = load_planck_map(smooth_tomofile, p=True)
                #p, q, u = Read_H5(file, names[0], shape=3)

            elif arg == 'dust':
                print('dust')
                #dust = load_planck_map(smooth_dust)
                dust = Read_H5(file, names[2], shape=1)

            else:
                print('Filenames dont contain "_IQU_", "_tomo_" or "_dust_".')
                sys.exit()
        smaps = [I, Q, U, p, q, u, dust, 'all']

    if len(names) == 1:
        for file in smooth_files:
            # if fname[0] == names[0] and res=res -> load
            fname = file.split('/')[-1]
            name = fname.split('_')
            l = 'smoothed{}arcmin.h5'.format(resolution)
            print(name, '---')
            print(name[0], names[0])
            if name[0] == names[0]:
                print('..')
                if l == name[-1]:
                    smap = Read_H5(file, names[0], shape)
                    print(np.shape(smap))
                    print('----')

            #break

        # load smoothed files
        if name.any() == 'IQU':
            #I, P, Q, U = load_planck_map(smooth_pl_file, p=True)
            I, Q, U = Read_H5(file, names[0], shape=3)
            smaps = [I, P, Q, U, 'planck']
        elif name.any() == 'tomo':
            #p, p, q, u = load_planck_map(smooth_tomofile, p=True)
            p, q, u = Read_H5(file, names[0], shape=3)
            smaps = [i, p, q, u, 'tomo']
        elif name.any() == 'dust':
            #dust = load_planck_map(smooth_dust)
            dust = Read_H5(file, names[0], shape=1)
            smaps = [dust, 'dust']
        else:
            print('Filename dont contain "_IQU_","_tomo_" or "_dust_".')
            sys.exit()


    else:
        print('Find no smoothed maps. Check that smoothed maps are availible')
        print('Filenames of smoothed maps must end with "_smoothed15arcmin.fits"')
        #smaps = smooth_maps(maps, Nside)
        #smaps.append('maps')
        sys.exit()
    return(smaps)
    """
# Helping functions

def dist2neighbours(pix, phi, theta, Nside=2048):
    """
    Calculate the angular distance to the other pixels, assume small angels.

    Parameters:
    -----------
    - pix, integer.     The pixel to evaluate from.
    - phi, array.       longitude array in radians.
    - theta, array.     The latitude array in radians.
    - Nside, integer.   The resolution of the pixel map. default is 2048

    Return:
    -----------
    - r, array.         The length between the input pixel and the other pixels.
    """
    #print(pix)
    theta0, phi0 = hp.pix2ang(Nside, pix)
    theta0 = np.pi/2. - theta0
    #print(phi0*180/np.pi, theta0*180/np.pi)
    #print(np.mean(phi)*180/np.pi, np.mean(theta)*180/np.pi)
    dphi = phi0 - phi
    dtheta = theta0 - theta
    #print(dphi*180/np.pi)
    #print(dtheta*180/np.pi)
    R2 = dphi**2 + dtheta**2
    return(np.sqrt(R2))

def write_H5(map, name, Nside, res=15, band=353):
    """
    Write a smoothed map to a HDF file.

    Parameters:
    -----------
    - map, array.       The smoothed map to be written to file.
    - name, string.     The data name/ column name of the data.
    - Nside, integer.   The map resolution, set to default 2048.
    - res, integer.     The smoothing resolution. Default is 15 arcmin.

    """
    f = h5py.File('Data/{}_Nside{}_{}_{}arcmin.h5'.\
                    format(name, Nside, band, res), 'w')
    f.create_dataset('{}'.format(name), data=map)
    f.close()


def Read_H5(filename, name, shape):
    """
    Function to read in a HDF file. Handles multicolumn data files, either 1 or
    3 columns.

    Parameters:
    -----------
    - filename, string. The name of the file to read.
    - name, string.     The column names of the data.
    - shape, integer.   The number of columns in the HDF file, must be either
                        3 or 1

    Return:
    -----------
    - maps, array.      If shape is 1, an array with the map is returned.
    - (I, Q, U), arrays.If shape is 3, returning 3 arrays containing the total
                        intesity, Q stokes parameter and U stokes parameter.
    """
    print(filename)
    f = h5py.File(filename, 'r')
    maps = np.asarray(f[name])
    f.close()
    print(np.shape(maps))
    if shape == 3:
        I = maps[0,:]
        Q = maps[1,:]
        U = maps[2,:]
        return(I, Q, U)
    else:
        return(maps)


def smooth(file, Nside=256, res=15, band=353):
    """
    Smooth a map from fits file to a given resolution, 
    degrade to given Nside = 256 (default), 
    write new map to h5 file. unit 217: Kcmb, unit 143: Kcmb
    """
    
    maps, hdr = hp.fitsfunc.read_map(file, field=None, h=True)
    print(np.shape(maps))
    print(hdr)
    Ns_map_in = hp.get_nside(maps[0])
    print(Ns_map_in)
    # smooth the map:
    #if len(map[:,0]) <= 3:
    smaps = smooth_maps(maps, Ns_map_in)
    #else:
    #    smaps = smooth_maps(maps[:3,:], Ns_map_in)
    # degrade to given Nside:
    if Nside == Ns_map_in:
        new_maps = smaps
    else:
        new_maps = hp.ud_grade(smaps, Nside, order_in='RING', order_out='RING')
        
    print(np.shape(new_maps))
    # Write map to h5 file:
    write_H5(new_maps*1e6, 'IQU', Nside, res=res, band=band)
    
    pass

"""
f3 = '/mn/stornext/u3/hke/bratli/comm_polfrac_353delta.fits'
f4 = '/mn/stornext/u3/hke/bratli/gnilc_polfrac_80arc_n256_v3.fits'
f5 = '/mn/stornext/u3/hke/bratli/npipe6v20_353_map_QUADCOR_ZODICOR_n2048_uK.fits'
f6 = '/mn/stornext/u3/hke/bratli/SRoll20_SkyMap_353psb_full.fits'
f1 = 'Data/HFI_SkyMap_217-field-IQU_2048_R3.00_full.fits'
f2 = 'Data/HFI_SkyMap_143-field-IQU_2048_R3.00_full.fits'
#smooth(f1, Nside=256, res=15, band=217) # done Kcmb
#smooth(f2, Nside=256, res=15, band=143)
#smooth(f3, Nside=256, res=15, band='353comm') # p_s, q_s, u_s
#smooth(f4, Nside=256, res=15, band='gnilc')   # p_s, q_s, u_s
#smooth(f5, Nside=256, res=15, band='353npipe')# IQU, Kcmb
smooth(f6, Nside=256, res=15, band='353Sroll')# IQU+Cij, Kcmb, Kcmb^2. 
#Need to fix Cij here, wrong power Cij*(Ns_in/Ns_out)^2
#"""
