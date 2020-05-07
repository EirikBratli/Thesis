"""
Program to compare Planck 353Ghz polarization map to tomagraphy data of Raphael
etal 2019.
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


#######################
# Tomography analysis #
#######################

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

def convert2galactic(ra, dec):
    """
    Function to convert coordinates into galactic coordinates

    Parameters:
    -----------
    - ra, array.    A data array containing the right ascension angular position
                    of the data points
    - dec, array.   A data array containing the declination angular position
                    of the data points
    Return:
    -------
    - lon, array. The longitude galactic coordinates
    - lat, array. The latitude galactic coordinates
    """
    coord = SkyCoord(ra=ra*u_.deg, dec=dec*u_.deg)
    coord_gal = coord.galactic

    lon = coord_gal.l.value
    lat = coord_gal.b.value

    return(lon, lat)

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
    l, b = convert2galactic(data[:,0], data[:,1])
    theta = np.pi/2. - b * np.pi/180.
    phi = l * np.pi/180.
    print(np.min(theta), np.max(theta))
    # get pixel numbers
    pix = hp.pixelfunc.ang2pix(Nside, theta, phi, nest=False)

    print(hp.pixelfunc.nside2pixarea(Nside, degrees=True))
    # Create maps
    Npix = hp.nside2npix(Nside)
    p_map = np.zeros(Npix)
    q_map = np.zeros(Npix)
    u_map = np.zeros(Npix)
    R1_map = np.zeros(Npix)

    sigma_p = np.zeros(Npix)
    sigma_q = np.zeros(Npix)
    sigma_u = np.zeros(Npix)

    print(len(np.unique(pix)))
    uniqpix = np.unique(pix)
    index = []
    #print(uniqpix)
    print(len(pix))
    for i in uniqpix:
        ind = np.where(pix == i)[0]
        # Use mean instead??
        p = np.mean(data[ind, 2])
        q = np.mean(data[ind, 4])
        u = np.mean(data[ind, 6])

        p_map[i] = p
        q_map[i] = q
        u_map[i] = u

        sigma_p[i] = np.mean(data[ind, 3])
        sigma_q[i] = np.mean(data[ind, 5])
        sigma_u[i] = np.mean(data[ind, 7])
        R1_map[i] = np.mean(data[ind, -1])

    #print(p_map[uniqpix])
    #print(np.sum(p_map==0))

    return(p_map, q_map, u_map, sigma_p, sigma_q, sigma_u, pix)

def plot_gnom(map, lon, lat, label, path):
    """
    Plotting function viewing in gnomonic projection.

    Parameters:
    -----------
    - map, array.     The map to plot.
    - lon, scalar.    mean position in longitude
    - lat, scalar.    mean position in latitude
    - label, string.  The name of the stokes parameters or similar description
    - path, string.   The path to the folder to save created figures.
    Return:
    -----------
    """
    #path = 'Figures/'
    hp.gnomview(map, title='Polarization {}'.format(label), rot=[lon, lat],\
                xsize=100)
    #hp.gnomview(map, title='Polarization {}'.format(label), rot=[lon,90-lat,180],\
    #            flip='geo', xsize=100)
    hp.graticule()
    plt.savefig(path + '{}.png'.format(label))



#######################
# Planck data 353 GHz #
#######################

def smoothing(map, Nside, iter=3):
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
    FWHM = (np.sqrt(15.**2-5.**2)/60.) * (np.pi/180) # 15 arcmin
    smap = hp.sphtfunc.smoothing(map, fwhm=FWHM, iter=iter)
    return(smap)

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
        (T,Q, U),hdr = hp.fitsfunc.read_map(file, field=(0,1,2), h=True)
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

#def get_pixels(lmin, lmax, bmin, bmax):
def compare(pix, tomo_map, planck_map, sigma, lon=None, lat=None):
    """
    Function to compare the tomography data with the planck data, compute the
    difference, difference squared, correlation and the residuals. Produces
    plots of the compared data.

    Parameters:
    -----------
    - pix, array.
    - tomo_map, array.
    - planck_map, array.
    - sigma, array.
    - lon, scalar.
    - lat, scalar.

    Return:
    -------
    """
    uniq = np.unique(pix)
    diff_map = tomo_map - planck_map
    #ratio_map = tomo_map/planck_map
    diff2_map = (tomo_map - planck_map)**2
    corr_map = tomo_map*planck_map/(np.max(tomo_map*planck_map))
    res_fac = 1#minimize(tomo_map, planck_map, sigma)
    residual = tomo_map - res_fac*planck_map

    if (lon is not None) and (lat is not None):
        plot_gnom(diff_map, lon, lat, 'P_difference')
        plot_gnom(diff2_map, lon, lat, 'P_difference^2')
        #plot_gnom(ratio_map, lon, lat, 'ratio')
        plot_gnom(corr_map, lon, lat, 'P_correlation')
        plot_corr(tomo_map, planck_map)
        plot_gnom(residual, lon, lat, 'P_residual')
    return(diff_map, corr_map, residual)

def Correlation(tomo_map, planck_map, mask, Nside=2048):
    """
    Compute a correlation map between tomograpy map and planck map, together
    with the correlation coefficients which is printed.

    Parameters:
    -----------
    - tomo_map, array.      The tomography map to use in the correlation
    - planck_map, array.    The planck map to correlate with the tomography map,
                            must be of same shape.
    - mask, array.            Pixels where the there is data inthe tomography map.
    - Nside, integer.         The resolution of the maps, default is 2048.

    Returns:
    -----------
    - Corr, Array.          The correlation map, have the same shape as the input
                            maps, but only values inside the mask.
    """
    Npix = hp.nside2npix(Nside)
    Corr = np.zeros(Npix)
    R = np.corrcoef(tomo_map[mask], planck_map[mask])
    print('Correlation coefficient:')
    print(R)

    sigma_t = np.std(tomo_map[mask])
    sigma_pl = np.std(planck_map[mask])
    #print(sigma_t, sigma_pl)
    cov = np.cov(tomo_map[mask], planck_map[mask])
    #print(cov)
    #Corr[mask] = cov/(sigma_t*sigma_pl)
    #print(cov/(sigma_t*sigma_pl))
    return(Corr)

def Difference(tomo_map, planck_map, mask, Nside=2048):
    """
    Compute the difference between to maps in the masked tomography area.

    Parameters:
    -----------
    - tomo_map, array.      The tomography map to use in the correlation
    - planck_map, array.    The planck map to correlate with the tomography map,
                            must be of same shape.
    - mask, array.            Pixels where the there is data inthe tomography map.
    - Nside, integer.         The resolution of the maps, default is 2048.

    Returns:
    -----------
    - Diff, array.       The difference map computed by the two input maps.

    """
    Npix = hp.nside2npix(Nside)
    Diff = np.zeros(Npix)
    Diff[mask] = (tomo_map[mask] - planck_map[mask])/(tomo_map[mask])
    return(Diff)

def minimize(map1, map2, sigma):
    """
    Function to minimize the difference between the tomography and planck with
    respect to a contant. Used to calculate the residuals.

    Parameters:
    -----------
    - map1, array. Usually the tomography map.
    - map2, array. Usually the planck map.
    - sigma, array. The uncertainty map of tomography.

    Return:
    -------
    - minimize. scalar, the constant that minimize ((m1-a*m2)/|s1|)^2
    """
    ind = np.where(map1 != 0.)[0]
    def min_func(fac, map1=map1[ind], map2=map2[ind], sigma=sigma[ind]):
        return(np.sum(((map1 - fac*map2)/np.abs(sigma)))**2)

    minimize = spo.fmin_powell(min_func, x0=100)
    return(minimize)

def plot_corr(tomo_map, planck_map, name, path, log=False, y_lim=False, x_lim=False):
    """
    Plot the correlation between tomography and planck353 in a scatter plot.
    
    - tomo_map, array.    The tomography map to use in the plotting.
    - planck_map, array.  The Planck map to use in the plotting.
    - name, string.       The name of the figure describing what we look at.
    - log, bool.          If true plot also on logarithmic scales on the y axis
    - y_lim, bool.        If true use limits of (-0.1, 0.1) on the y axis
    - x_lim, bool-        If true use limits of (-1, 1) on the x axis.
    """
    #print(np.shape(tomo_map), np.shape(planck_map))
    ind = np.where(tomo_map != 0)[0]
    R = np.corrcoef(tomo_map, planck_map)
    print('Correlation coefficient: (biased??)')
    print(R)

    ind = np.where(tomo_map != 0.)[0]
    plt.figure('{}'.format(name))
    plt.plot(tomo_map[ind], planck_map[ind], '.k')
    plt.xlabel('Tomography data')
    plt.ylabel('Planck 353GHz')
    if y_lim is True:
        plt.ylim(-0.1, 0.1)
    if x_lim is True:
        plt.xlim(-1,1)
    plt.tight_layout()
    plt.savefig(path + '{}.png'.format(name))


    if log is True:
        plt.figure('{} log'.format(name))
        plt.semilogy(tomo_map[ind], planck_map[ind], '.k')
        plt.xlabel('Tomography data')
        plt.ylabel('Planck 353GHz')
        plt.tight_layout()
        plt.savefig(path + '{}_log.png'.format(name))

def smooth_maps(maps, Nside, iterations=3):
    """
    This function smooths a list of maps calling healpys smoothing function.
    
    Parameters:
    -----------
    - maps, list.       A list with the maps, all maps must be of the same Nside.
    - Nside, integer.   The resolution of the maps.
    - iterations, integer. The number of iterations the smooting function does.

    Return:
    ----------
    -out_maps, list.     A list of smoothed maps.
    """

    out_maps = []
    t0 = time.time()
    for i in range(len(maps)):
        t1 = time.time()
        print('Smooth map {}'.format(i))
        m = smoothing(maps[i], Nside, iter=iterations)
        out_maps.append(m)
        t2 = time.time()
        print('smoothing time map {}: {}s'.format(i, t2-t1))

    t3 = time.time()
    print('Total smoothing time: {}'.format(t3-t0))
    return(out_maps)

def dist2neighbours(pix, phi, theta, Nside=2048):
    """
    Calculate the angular disance to the other pixels, assume small angels.

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

    theta0, phi0 = hp.pix2ang(Nside, pix)
    theta0 = np.pi/2. - theta0
    #print(theta0*180/np.pi, phi0*180/np.pi)
    dphi = phi0 - phi
    dtheta = theta0 - theta
    #print(dphi*180/np.pi)
    #print(dtheta*180/np.pi)
    R2 = dphi**2 + dtheta**2
    return(np.sqrt(R2))

def smooth_tomo_map(in_map, mask, Nside=2048):
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
    #N_map = np.zeros(Npix)
    smap = np.zeros(Npix)

    b, l = hp.pix2ang(Nside, mask)
    #print(np.mean(l)*180./np.pi, 90-np.mean(b)*180./np.pi)
    #sys.exit()
    phi = l
    theta = np.pi/2. - b

    s = (15./60.)*(np.pi/180.)/np.sqrt(8.*np.log(2.))
    #print(s)
    for p in mask:

        # smoothing:
        r = dist2neighbours(p, phi, theta)
        #print(r)
        I1 = np.sum(in_map[mask] * np.exp(-0.5*(r/s)**2))
        I2 = np.sum(np.exp(-0.5*(r/s)**2))
        #print(I1, I2)
        smap[p] = I1/I2
        #sys.exit()
    #
    #print(smap[mask])
    return(smap)

def write_H5(map, name, Nside=2048, res=15):
    """
    Function to write a HDF5 file of a map or array of data.

    Parameters:
    -----------
    - map, array.  The map to write to the .h5 file, The array may be mulitdimensional
    - name, string. The column name of the map.
    - Nside, integer. The resolution of the map, default is 2048.
    - res, integer.   The smoothing resolution of the map.
    """
    f = h5py.File('Data/Smooth_maps/{}_Nside{}_smoothed{}arcmin.h5'.\
                    format(name, Nside, res), 'w')
    f.create_dataset('{}'.format(name), data=map)
    f.close()


def Read_H5(filename, name, shape):
    """
    Function to read a hdf5 file, and load the data into a numpy array. handels maps
    with I, Q, U polarisation

    Parameters:
    -----------
    - filename, string.  The name of the file to read.
    - name, string.      The column name of the map loaded
    - shape, integer.    To check if the map contain more columns, like when the map 
                         is of I, Q, U polarisation.

    Return:
    -----------
    - maps/I,Q,U. Array. If shape is 3 returns the I, Q, and U polarisation maps in 
                         separated arrays, else return one map.
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

def Write_smooth_map(maps, name, Nside=2048, res=15, iterations=3):
    """
    Write a smoothed map to file, using only the tomography pixels. This function calls 
    Write_H5. It can either write 7 maps at a time or one, Specified by the 3 
    polarisation maps from planck, 3 tomography maps and one dust temperature map.

    Parameters:
    -----------
    - maps, list.     The list of smoothed maps to write to file.
    - name, string.   The column names of the maps.
    - Nside, integer. The map resolution.
    - res, integer.   The smoothing resolution of the maps.
    - iterations, integer. How many interation the smoothing shall use.
    """

    print(len(maps))
    if len(maps) == 7:
        smap = smooth_maps(maps, Nside, iterations)
        tomo_maps = smap[0:3]
        pl_maps = smap[3:6]
        dust_map = smap[-1]
        print('Writing tomo maps:')
        #hp.fitsfunc.write_map('Data/{}_Nside{}_smoothed{}arcmin.fits'.\
        #                    format(name[0], Nside, res), tomo_maps)
        write_H5(tomo_maps, name[0])
        print('Writing planck maps:')
        #hp.fitsfunc.write_map('Data/{}_Nside{}_smoothed{}arcmin.fits'.\
        #                    format(name[1], Nside, res), pl_maps)
        write_H5(pl_maps, name[1])
        print('Writing dust maps')
        #hp.fitsfunc.write_map('Data/{}_smoothed{}arcmin.fits'.\
        #                    format(name[2], Nside, res), dust_map)
        write_H5(dust_map, name[2])

    else:
        print('...')
        smap = smooth_maps(maps, Nside, iterations)
        print('Writing {} map'.format(name[0]))
        write_H5(smap, name[0])
        #hp.fitsfunc.write_map('Data/{}_Nside{}_smoothed{}arcmin.fits'.\
        #                    format(name[0], Nside, res), smap)

def read_smooth_maps(names, Nside=2048, resolution=15):
    """
    Read smoothed maps of a given smoothing resolution. Handels if there are 3 
    files with smoothed maps, 1 file and if cant find any smoothed map files. Do not 
    read smoothed tomography files. Also first part of filename must be either 'IQU' if 
    Planck polarisation, 'tomo' if tomography file or 'dust' if dust file.
    
    Parameters:
    -----------
    - names, list.     A list with the column names of the maps.
    - Nside, integer.  The resolution of the maps.
    - resolution, integer. The smoothing resolution of the maps.

    Returns:
    ----------
    - smaps, list.   List with the read in maps. Last linst element is a string 
                     describing the content of the list. Is either: 'all' for all 
                     maps available, 'dust' for dust map, 'tomo' for tomography map, 
                     and 'IQU' for planck maps.
    """

    # find smoothed files
    path = ''
    print(os.path.abspath(os.curdir))
    smooth_files = glob.glob(path + '*_smoothed{}arcmin.h5'.format(resolution))
    print(smooth_files, len(smooth_files))
    smaps = []
    if len(smooth_files) == 3:
        for file in smooth_files:
            name = file.split('/')[-1]
            arg = name.split('_')[0]
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

    elif len(smooth_files) == 1:
        name = smooth_files.split('_')
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

def Krj2Kcmb(map, f_ref=353.):
    """
    Function to convert from uK_rj(antenna temp) to K_cmb.
    
    Parameters:
    -----------
    - map, array. The map with units in uK_rj which is to be converted.
    - f_ref, scalar. The reference frequency in GHz, default is 353GHz.
    
    Return:
    - The map with units converted to K_cmb.
    """
    # Convert dust units to pol units, uK_RJ -> K_cmb
    f, tau = cu.read_freq()
    f = f[6]
    tau = tau[6]
    ind = np.where((f > f_ref/2.) & (f < 2*f_ref))[0]
    f_353 = f[ind]
    dBdTrj = cu.dBdTrj(f_353)
    dBdTcmb = cu.dBdTcmb(f_353)

    Ucmb = cu.UnitConv(f_353, tau, dBdTrj, dBdTcmb, 'K_RJ', 'K_CMB', f_ref)
    return(Ucmb*map*1e-6)

def get_Stokes(fractional, intensity, mask, Nside=2048):
    """
    Compute the Stokes parameter from a fractional polarisation.

    Parameters:
    -----------
    - fractional, array. Map with Nside 2048 containing fractional/relative 
                         polarisation values.
    - intensity, array.  The intensity map to multiply the fractional map with, 
                         must have the same shape.
    - mask, array.       An array with the relevant pixels, no need in using array 
                         elements equal zero.
    - Nside, integer.    The resolution of the maps.

    Return:
    ----------
    - X, array.         The polarisation map containing stokes parameters.
    """
    Npix = hp.nside2npix(Nside)
    X = np.zeros(Npix)
    #print(np.shape(X), np.shape(X[mask]), np.shape(fractional))
    X[mask] = fractional[mask] * intensity[mask]
    return(X)

def get_Fractional(stokes, intensity, mask, Nside=2048):
    """
    Compute the fractional Stokes parameter from the Stokes parameter

    Parameters:
    -----------
    - stokes, array.   Map with containing a Stokes parameter with a given Nside
    - intensity, array.  The intensity map to multiply the fractional map with, 
                         must have the same shape.
    - mask, array.       An array with the relevant pixels, no need in using array 
                         elements equal zero.
    - Nside, integer.    The resolution of the maps.    

    Return:
    ----------
    - X, array.         The polarisation map containing stokes parameters.
    """
    Npix = hp.nside2npix(Nside)
    X = np.zeros(Npix)
    X[mask] = stokes[mask]/intensity[mask]
    return(X)

def map_analysis_function(frac_tomo, PlanckIQU, dust_map, mask, Nside=2048):
    """
    Compute the maps to return, incluedes parameter maps, difference maps,
    correlation maps and dust map in use. The correlation map is not finished.

    Parameters:
    -----------
    - frac_tomo, array. The fractional tomography map, of either u, q of p.
    - PlanckIQU, array. The Planck map containing one of the stokes parameters
    - dust_map, array.  The temperature map of the dust. Must have the same shape 
                        as the above maps.
    - mask, array.      Array containing the relevant pixel numbers.
    - Nside, integer.   The map resolution.

    Return:
    -----------
    - tot_res, list.    A list with maps containing Stokes parameter from tomography,
                        Planck, difference map and correlation map.
    - frac_res, list.   List with the relative Stoke parameter of tomography map,
                        Planck, difference map and correlation map.
    - dust_map, array.  The dust map, no operation unsed on this?

    """
    print('convert to stokes or fractional')
    Tot_tomo_map = get_Stokes(frac_tomo, dust_map, mask)
    frac_planck_map = get_Fractional(PlanckIQU, dust_map, mask)

    # Difference:
    print('compute difference')
    diff_Tot = Difference(Tot_tomo_map, PlanckIQU, mask, Nside)
    diff_frac = Difference(frac_tomo, frac_planck_map, mask, Nside)

    # correlation maps:
    print('compute correlation')
    corr_tot = Correlation(Tot_tomo_map, PlanckIQU, mask, Nside)
    corr_frac = Correlation(frac_tomo, frac_planck_map, mask, Nside)

    tomo_frac = np.zeros(len(frac_tomo))
    Planck_IQU = np.zeros(len(PlanckIQU))
    tomo_frac[mask] = frac_tomo[mask]
    Planck_IQU[mask] = PlanckIQU[mask]
    # return maps: fractionals, stokes, dust
    frac_res = [tomo_frac, frac_planck_map, diff_frac, corr_frac]
    tot_res = [Tot_tomo_map, Planck_IQU, diff_Tot, corr_tot]
    return(tot_res, frac_res, dust_map)


def main(tomofile, colnames, planckfile, dustfile, Ppol=False,\
        Qpol=False, Upol=False, write=False, read=False):
    """
    The main function of the program. Do all the calling to the functions used
    to calculate the comparison between the Tomography data and Planck
    polarisation data.

    Parameters:
    -----------
    - tomofile, string.     The name of the tomography file.
    - colnames, list.       List of the column names of the tomography file.
    - planckfile, string.   Name of the planck file to compare with.
    - dustfile, string.     Name of the dust intensity file.
    - Ppol, bool.           If true use the total polarisation/Intensity/temperature.
    - Qpol, bool.           If true use the Q Stokes parameters.
    - Upol, bool.           If true use the U Stokes parameters.
    - write, bool.          If true, smooth the polarisation maps and write them to files.
                            Quits the program when done.
    - read, bool.           If true, read in the smoothed polarisation maps. el el      jfgks
    Return:
    -------
    """

    data = load_tomographydata(tomofile, colnames)
    print(data[0,:])
    p_map, q_map, u_map, sigma_p, sigma_q, sigma_u, pix = tomo_map(data)
    #u_map = -u_map
    mask = np.unique(pix)
    #print(mask)

    l, b = convert2galactic(data[:,0], data[:,1])

    lon = np.mean(l)
    lat = np.mean(b)
    print(lon, lat)
    print('pixel size, arcmin:', hp.pixelfunc.nside2resol(2048, arcmin=True))
    print('pixel size, radian:', hp.pixelfunc.nside2resol(2048))
    #sys.exit()


    #print('load planck 353GHz data')
    #T, P, Q, U = load_planck_map(planckfile, p=True)
    #d353 = load_planck_map(dustfile)
    #dust353 = Krj2Kcmb(d353)
    names = ['pqu_tomo', 'IQU_planck', 'I_dust']
    # write smoothed maps
    if write is True:
        # load planck
        print('load planck 353GHz data')
        T, P, Q, U = load_planck_map(planckfile, p=True)
        d353 = load_planck_map(dustfile)
        dust353 = Krj2Kcmb(d353)

        print('Write smoothed maps to file')
        Write_smooth_map([p_map, q_map, u_map, T, Q, U, dust353],\
                        ['tomo','IQU', 'dust'])
        # ['pqu_tomo', 'IQU_planck', 'I_dust'] used to be
        #Write_smooth_map([p_map], ['p_tomo2'], iterations=3)

        sys.exit()
    #
    # Read in  smoothed maps
    if read is True:
        print('Load smoothed maps')
        smaps = read_smooth_maps(names)
        if smaps[-1] == 'all':
            T_smap, Q_smap, U_smap = smaps[0:3] # does this work??
            p_smap_sph, q_smap_sph, u_smap_sph = smaps[3:6] # spherical harmonic
            dust_smap = smaps[-2]



        elif smaps[-1] == 'planck':
            T_smap, Q_smap, U_smap = smaps[0:3]

        elif smaps[-1] == 'tomo':
            p_smap_sph, q_smap_sph, u_smap_sph = smaps[0:3] # spherical harmonic

        elif smaps[-1] == 'dust':
            dust_smap = smaps[0]

        print('smoothed maps loaded')
        #print(np.shape(smaps))

    else:
        print('Use non smoothed maps')
        # load planck
        print('load planck 353GHz data')
        T, P, Q, U = load_planck_map(planckfile, p=True)
        d353 = load_planck_map(dustfile)
        dust353 = Krj2Kcmb(d353)

    #
    """
    Work with smoothed maps reduced to given area of Tomography data
    """

    #sys.exit()


    if Ppol == True:
        print('-- P polarisation --')
        p_smap = smooth_tomo_map(p_map, mask)
        if read is True:
            tot_res, frac_res, dust = map_analysis_function(p_smap, T_smap,\
                                                            dust_smap, mask)

        else:
            tot_res, frac_res, dust = map_analysis_function(p_map, T,\
                                                            dust353, mask)

        return(tot_res, frac_res, dust, lon, lat)

    elif Qpol == True:
        print('-- Q polarisation --')
        q_smap = smooth_tomo_map(q_map, mask)
        if read is True:
            tot_res, frac_res, dust = map_analysis_function(q_smap, Q_smap,\
                                                            dust_smap, mask)
        else:
            tot_res, frac_res, dust = map_analysis_function(q_map, Q,\
                                                            dust353, mask)

        return(tot_res, frac_res, dust, lon, lat)

    elif Upol == True:
        print('-- U polarisation --')
        u_smap = smooth_tomo_map(u_map, mask)
        if read is True:
            tot_res, frac_res, dust = map_analysis_function(u_smap, U_smap,\
                                                            dust_smap, mask)
        else:
            tot_res, frac_res, dust = map_analysis_function(u_map, U,\
                                                            dust353, mask)

        return(tot_res, frac_res, dust, lon, lat)



########################


########################

tomofile = 'total_tomography.csv'
planckfile = 'HFI_SkyMap_353-psb-field-IQU_2048_R3.00_full.fits'
dustfile = 'dust_353_commander_temp_n2048_7.5arc.fits'

colnames = ['ra', 'dec', 'p', 'p_er', 'q', 'q_er', 'u', 'u_er', 'dist',\
            'dist_low', 'dist_up', 'Rmag1']
#load_planck_map(planckfile)


#main(tomofile, colnames, planckfile, dustfile, Ppol=True)
#main(tomofile, colnames, planckfile, dustfile, Qpol=True, write=True)

# input args from commando line:

if len(sys.argv) == 1:
    print('Need input arguments: (write/read/unsmooth), (U/Q/P) and (plot)')
    sys.exit()

elif len(sys.argv) > 1:
    print('----------------')
    if sys.argv[1] == 'write':
        # Write smoothed maps:
        main(tomofile, colnames, planckfile, dustfile, write=True)


    elif sys.argv[1] == 'read':
        # Read in smoothed maps:
        if sys.argv[2] == 'U':
            Tot_res, frac_res, dust, lon, lat = main(tomofile, colnames,\
                                    planckfile, dustfile, Upol=True, read=True)

        elif sys.argv[2] == 'Q':
            Tot_res, frac_res, dust, lon, lat = main(tomofile, colnames,\
                                    planckfile, dustfile, Qpol=True, read=True)

        elif sys.argv[2] == 'P':
            Tot_res, frac_res, dust, lon, lat = main(tomofile, colnames,\
                                    planckfile, dustfile, Ppol=True, read=True)

    elif sys.argv[1] == 'unsmooth':
        # Read in smoothed maps:
        if sys.argv[2] == 'U':
            Tot_res, frac_res, dust, lon, lat = main(tomofile, colnames,\
                                    planckfile, dustfile, Upol=True)

        elif sys.argv[2] == 'Q':
            Tot_res, frac_res, dust, lon, lat = main(tomofile, colnames,\
                                    planckfile, dustfile, Qpol=True)

        elif sys.argv[2] == 'P':
            Tot_res, frac_res, dust, lon, lat = main(tomofile, colnames,\
                                    planckfile, dustfile, Ppol=True)

        # the returned lists are sorted like:
        # [tomo_map, planck_map, difference_map, correlation_map]

    else:
        print('First input argument should be "write", "read" or "unsmooth"')
        sys.exit()
    print('----------------')

if len(sys.argv) == 4:
    print('Plotting')
    savepath = '/mn/stornext/u3/eiribrat/mineFiler/Thesis_figures/'
    # Plotting:
    arg = sys.argv[2]
    if (sys.argv[3] == 'plot'): 
        if (sys.argv[1] == 'unsmooth'):
            plot_corr(Tot_res[0], Tot_res[1], 'corr_{}_tot'.format(arg), savepath)
            plot_corr(frac_res[0], frac_res[1], 'corr_{}_frac'.format(arg), savepath)
            #plot_gnom(Tot_res[3], lon,lat, 'corr_U_map1')
            #plot_gnom(frac_res[3], lon, lat, 'curr_u_frac_map1')

            plot_gnom(Tot_res[0], lon, lat, '{}_tomo'.format(arg), savepath)
            plot_gnom(frac_res[0], lon, lat, '{}_frac_tomo'.format(arg), savepath)
            plot_gnom(Tot_res[1], lon, lat, '{}_planck'.format(arg), savepath)
            plot_gnom(frac_res[1], lon, lat, '{}_frac_planck'.format(arg), savepath)

            #plot_gnom(Tot_res[2], lon, lat, 'diff_U1')
            #plot_gnom(frac_res[2], lon, lat, 'diff_U_frac1')

        else:
            plot_corr(Tot_res[0], Tot_res[1], 'corr_{}_tot_smooth'.format(arg),\
                      savepath)
            plot_corr(frac_res[0], frac_res[1], 'corr_{}_frac_smooth'.format(arg),\
                      savepath)
            #plot_gnom(Tot_res[3], lon,lat, 'corr_U_map1')
            #plot_gnom(frac_res[3], lon, lat, 'curr_u_frac_map1')

            plot_gnom(Tot_res[0], lon, lat, '{}_tomo_smooth15arc'.format(arg),\
                      savepath)
            plot_gnom(frac_res[0], lon, lat, '{}_frac_tomo_smooth15arc'.format(arg),\
                      savepath)
            plot_gnom(Tot_res[1], lon, lat, '{}_planck_smooth15arc'.format(arg),\
                      savepath)
            plot_gnom(frac_res[1], lon, lat, '{}_frac_planck_smooth15arc'.format(arg),\
                      savepath)

            #plot_gnom(Tot_res[2], lon, lat, 'diff_U1')
            #plot_gnom(frac_res[2], lon, lat, 'diff_U_frac1')


    else:
        sys.exit()


#plt.show()
#
