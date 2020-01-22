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
    theta = b * np.pi/180.
    phi = l * np.pi/180.
    print(np.min(theta), np.max(theta))
    # get pixel numbers
    pix = hp.pixelfunc.ang2pix(Nside, theta, phi)

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

    print(Npix, np.shape(p_map))
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

def plot_gnom(map, lon, lat, label):
    """
    Plotting function viewing in gnomonic projection.

    Parameters:
    -----------
    - map, array. The map to plot.
    - lon, scalar. mean position in longitude
    - lat, scalar. mean position in latitude
    - label, string.
    Return:
    -------
    """
    path = 'Figures/tomography/'
    hp.gnomview(map, title='Polarization {}'.format(label), rot=[lon,90-lat,180],\
                flip='geo', xsize=100)
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

    Corr[mask] = tomo_map[mask]*planck_map[mask] /\
                                        (np.max(tomo_map[mask]*planck_map[mask]))
    return(Corr)

def Difference(tomo_map, planck_map, mask, Nside=2048):
    """
    Compute the difference between to maps in the masked tomography area.

    Parameters:
    -----------

    Returns:
    -----------

    """
    Npix = hp.nside2npix(Nside)
    Diff = np.zeros(Npix)
    Diff[mask] = tomo_map[mask] - planck_map[mask]
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

def plot_corr(tomo_map, planck_map, name, log=False, y_lim=False, x_lim=False):
    """
    Plot the correlation between tomography and planck353.
    """
    ind = np.where(tomo_map != 0)[0]
    R = np.corrcoef(tomo_map[ind], planck_map[ind])
    print('Correlation coefficient:')
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
    plt.savefig('Figures/tomography/{}.png'.format(name))


    if log is True:
        plt.figure('{} log'.format(name))
        plt.semilogy(tomo_map[ind], planck_map[ind], '.k')
        plt.xlabel('Tomography data')
        plt.ylabel('Planck 353GHz')
        plt.savefig('Figures/tomography/{}_log.png'.format(name))

def smooth_maps(maps, Nside, iterations=3):

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

def write_H5(map, name, Nside=2048, res=15):
    """

    """
    f = h5py.File('Data/{}_Nside{}_smoothed15arcmin.h5'.format(name, Nside, res), 'w')
    f.create_dataset('{}'.format(name), data=map)
    f.close()


def Read_H5(filename, name, shape):
    """

    """
    f = h5py.File(filename, 'r')
    maps = np.asarray(f[name])
    f.close()
    if shape == 3:
        I = maps[0,:]
        Q = maps[1,:]
        U = maps[2,:]
        return(I, Q, U)
    else:
        return(maps)

def Write_smooth_map(maps, name, Nside=2048, res=15, iterations=3):
    """
    Write a smoothed map to file, using only the tomography pixels
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

def read_smooth_maps(maps, Nside=2048, resolution=15):
    # find smoothed files
    smooth_files = glob.glob('Data/*_smoothed{}arcmin.fits'.format(resolution))
    smaps = []
    if len(smooth_files) == 3:
        for file in smooth_files:
            name = file.split('_')
            # load smoothed files
            if name.any() == 'IQU':
                print('planck')
                #I, P, Q, U = load_planck_map(smooth_pl_file, p=True)
                I, Q, U = Read_H5(file, name, shape=3)

            elif name.any() == 'tomo':
                print('tomo')
                #i, p, q, u = load_planck_map(smooth_tomofile, p=True)
                p, q, u = Read_H5(file, name, shape=3)
            elif name.any() == 'dust':
                print('dust')
                #dust = load_planck_map(smooth_dust)
                dust = Read_H5(file, name, shape=1)
            else:
                print('Filenames dont contain "_IQU_","_tomo_" or "_dust_".')
                sys.exit()
        smaps = [I, Q, U, p, q, u, dust, 'all']

    elif len(smooth_files) == 1:
        name = smooth_files.split('_')
        # load smoothed files
        if name.any() == 'IQU':
            #I, P, Q, U = load_planck_map(smooth_pl_file, p=True)
            I, Q, U = Read_H5(file, name, shape=3)
            smaps = [I, P, Q, U, 'planck']
        elif name.any() == 'tomo':
            #p, p, q, u = load_planck_map(smooth_tomofile, p=True)
            p, q, u = Read_H5(file, name, shape=3)
            smaps = [i, p, q, u, 'tomo']
        elif name.any() == 'dust':
            #dust = load_planck_map(smooth_dust)
            dust = Read_H5(file, name, shape=1)
            smaps = [dust, 'dust']
        else:
            print('Filename dont contain "_IQU_","_tomo_" or "_dust_".')
            sys.exit()

    else:
        print('Find no smoothed maps. Check that smoothed maps are availible')
        print('Filenames of smouthed maps must end with "_smoothed15arcmin.fits"')
        #smaps = smooth_maps(maps, Nside)
        #smaps.append('maps')
        sys.exit()
    return(smaps)

def Krj2Kcmb(map, f_ref=353.):
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
    Compute the Stokes parameter from a fractional polarisation
    """
    Npix = hp.nside2npix(Nside)
    X = np.zeros(Npix)
    print(np.shape(X), np.shape(X[mask]), np.shape(fractional))
    X[mask] = fractional[mask] * intensity[mask]
    return(X)

def get_Fractional(stokes, intensity, mask, Nside=2048):
    """
    Compute the fractional Stokes parameter from the Stokes parameter
    """
    Npix = hp.nside2npix(Nside)
    X = np.zeros(Npix)
    X[mask] = stokes[mask]/intensity[mask]
    return(X)

def map_analysis_function(frac_tomo, PlanckIQU, dust_map, mask, Nside=2048):
    """
    Compute the maps to return, incluedes parameter maps, difference maps,
    correlation maps and dust map in use.
    """
    Tot_tomo_map = get_Stokes(frac_tomo, dust_map, mask)
    frac_planck_map = get_Fractional(PlanckIQU, dust_map, mask)

    # Difference:
    diff_Tot = Difference(Tot_tomo_map, PlanckIQU, mask, Nside)
    diff_frac = Difference(frac_tomo, frac_planck_map, mask, Nside)

    # correlation maps:
    corr_tot = Correlation(Tot_tomo_map, PlanckIQU, mask, Nside)
    corr_frac = Correlation(frac_tomo, frac_planck_map, mask, Nside)
    # return maps: fractionals, stokes, dust
    frac_res = [frac_tomo, frac_planck_map, diff_frac, corr_frac]
    tot_res = [Tot_tomo_map, PlanckIQU, diff_Tot, corr_tot]
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

    Return:
    -------
    """

    data = load_tomographydata(tomofile, colnames)
    print(data[0,:])
    p_map, q_map, u_map, sigma_p, sigma_q, sigma_u, pix = tomo_map(data)
    #sys.exit()
    #ind = (np.where(p_map > 0)[0])
    #ind1 = (np.where(q_map != 0)[0])
    #ind2 = (np.where(u_map != 0)[0])
    #print(len(ind), len(ind1), len(ind2))
    mask = np.unique(pix)
    #print(mask, len(mask), len(pix))

    l, b = convert2galactic(data[:,0], data[:,1])

    lon = np.mean(l)
    lat = np.mean(b)
    print(lon, lat)
    print('pixel size, arcmin:', hp.pixelfunc.nside2resol(2048, arcmin=True))
    print('pixel size, radian:', hp.pixelfunc.nside2resol(2048))
    #sys.exit()

    # load planck
    print('load planck 353GHz data')
    T, P, Q, U = load_planck_map(planckfile, p=True)
    d353 = load_planck_map(dustfile)
    dust353 = Krj2Kcmb(d353)

    # write smoothed maps
    if write is True:
        print('Write smoothed maps to file')
        #Write_smooth_map([p_map, q_map, u_map, Skychunk_T, Skychunk_Q,\
                    #Skychunk_U, dust353], ['pqu_tomo','IQU_planck', 'I_dust'])

        Write_smooth_map([p_map], ['p_tomo'], iterations=1)

        sys.exit()
    #
    # Read in  smoothed maps
    if read is True:
        smaps = read_smooth_maps([u_map, U, dust353])
        if smaps[-1] == 'all':
            T_smap, P_smap, Q_smap, U_smap = smaps[0:4] # does this work??
            p_smap, p_smap1, q_smap, u_smap = smaps[4:8]
            dust_smap = smaps[-2]

        elif smaps[-1] == 'planck':
            T_smap, P_smap, Q_smap, U_smap = smaps[0:4]

        elif smaps[-1] == 'tomo':
            p_smap, p_smap1, q_smap, u_smap = smaps[0:4]

        elif smaps[-1] == 'dust':
            dust_smap = smaps[0]

    else:
        print('Use non smoothed maps')
        pass
    #
    """
    Work with smoothed maps reduced to given area of Tomography data
    """
    """
    Skychunk_T = get_PlanckSky_chunk(T, pix)
    Skychunk_P = get_PlanckSky_chunk(P, pix)
    Skychunk_Q = get_PlanckSky_chunk(Q, pix)
    Skychunk_U = get_PlanckSky_chunk(U, pix)
    Skychunk_dust = get_PlanckSky_chunk(dust353, pix)
    plot_gnom(u_map, lon, lat, 'u_test')
    plot_gnom(U, lon, lat, 'U_full')
    plot_gnom(Skychunk_U, lon, lat, 'U_part')

    #plt.show()
    #sys.exit()
    Ptomo = p_map*Skychunk_dust
    p_pl = T/Skychunk_dust
    Qtomo = q_map*Skychunk_dust
    q_pl = Q/Skychunk_dust
    Utomo = u_map*Skychunk_dust
    u_pl = U/Skychunk_dust

    plot_corr(Ptomo, T, 'Scatter_corr_T')
    plot_corr(p_map, p_pl, 'Scatter_corr_p_frac')
    plot_corr(Qtomo, Q, 'Scatter_corr_Q')
    plot_corr(q_map, q_pl, 'Scatter_corr_q_frac')
    plot_corr(Utomo, U, 'Scatter_corr_U')
    plot_corr(u_map, u_pl, 'Scatter_corr_u_frac')
    #plt.show()
    #sys.exit()

    # Smooth map to 15 arcmin
    #print('Smooth dust maps to 15 arcmin')
    #maps = smooth_maps([Skychunk_dust], 2048)
    #dust_smap = maps[0]

    #
    #plot_gnom(Skychunk_T, lon, lat, 'planck353_T')
    #plot_gnom(Skychunk_dust, lon, lat, 'dust353')
    """

    # find stokes params for Tomography:
    #P_tomo_smap = get_Stokes(p_smap, dust_smap, mask)
    #Q_tomo_smap = get_Stokes(q_smap, dust_smap, mask)
    #U_tomo_smap = get_Stokes(u_smap, dust_smap, mask)

    # find fractional stokes params for Planck:
    #p_planck_smap = get_Fractional(T_smap, dust_smap, mask)
    #q_planck_smap = get_Fractional(Q_smap, dust_smap, mask)
    #u_planck_smap = get_Fractional(U_smap, dust_smap, mask)

    if Ppol == True:
        print('-- P polarisation --')
        if read is True:
            tot_res, frac_res, dust = map_analysis_function(p_smap, T_smap,\
                                                            dust_smap, mask)

        else:
            tot_res, frac_res, dust = map_analysis_function(p_map, T,\
                                                            dust353, mask)

        return(tot_res, frac_res, dust, lon, lat)
        """
        P_tomo_smap = get_Stokes(p_smap, dust_smap, mask)
        p_planck_smap = get_Fractional(T_smap, dust_smap, mask)

        # Difference:
        diff_P = Difference(P_tomo_smap, P_smap, mask)
        diff_p_frac = Difference(p_smap, p_planck_smap, mask)

        # correlation maps:
        corr_P_smap = Correlation(P_tomo_smap, T_smap, mask)
        corr_p_frac_smap = Correlation(p_smap, p_planck_smap, mask)
        # return maps: fractionals, stokes
        frac_res = [p_smap, p_planck_smap, diff_p_frac, corr_p_frac_smap]
        tot_res = [P_tomo_smap, T_smap, diff_P, corr_P_smap]

        # Smooth map to 15 arcmin, this takes ca 15 min per map!
        print('Smooth maps to 15 arcmin')
        smaps = smooth_maps([p_map, Skychunk_T], 2048)
        p_smap = smaps[0]
        T_smap = smaps[1]

        P_tomo_smap = p_smap*dust_smap
        p_planck_smap = T_smap/dust_smap  # T gets to p when fractional

        diff_p, corr_p, res_p = compare(pix, p_smap, p_planck_smap, sigma_p)
        diff_T, corr_T, res_T = compare(pix, P_tomo_smap, T_smap, sigma_p)

        # diff plots
        plot_gnom(diff_p, lon, lat, 'p_frac_difference')  #
        plot_gnom(diff_T, lon, lat, 'temp_difference')
        #plot_gnom(corr_p, lon, lat, 'p_frac_correlation')
        #plot_gnom(corr_T, lon, lat, 'Temp_correlation')
        # smooth plots
        #plot_gnom(p_smap, lon, lat, 'p_frac_tomo_smooth15arc')         #
        #plot_gnom(p_planck_smap, lon, lat, 'planck353_p_frac_smooth15arc') # T
        #plot_gnom(P_tomo_smap, lon, lat, 'P_tomo_smooth15arc')
        #plot_gnom(T_smap, lon, lat, 'planck353_Temp_smooth15arc')
        # correlation plots
        #print(p_smap[np.where(p_smap > 0)])
        plot_corr(P_tomo_smap, T_smap, 'Scatter_corr_P_smooth15arc')
        plot_corr(p_smap, p_planck_smap, 'Scatter_corr_p_frac_smooth15arc', y_lim=True)
        """

    if Qpol == True:
        print('-- Q polarisation --')
        if read is True:
            tot_res, frac_res, dust = map_analysis_function(q_smap, Q_smap,\
                                                            dust_smap, mask)
        else:
            tot_res, frac_res, dust = map_analysis_function(q_map, Q,\
                                                            dust353, mask)

        return(tot_res, frac_res, dust, lon, lat)
        """
        Q_tomo_smap = get_Stokes(q_smap, dust_smap, mask)
        q_planck_smap = get_Fractional(Q_smap, dust_smap, mask)

        # Difference:
        diff_Q = Difference(Q_tomo_smap, Q_smap, mask)
        diff_q_frac = Difference(q_smap, q_planck_smap, mask)

        # Correlation maps:
        corr_Q_smap = Correlation(Q_tomo_smap, Q_smap, mask)
        corr_q_frac_smap = Correlation(q_smap, q_planck_smap, mask)

        frac_res = [q_smap, q_planck_smap, diff_q_frac, corr_q_smap]
        tot_res = [Q_tomo_smap, Q_smap, diff_Q, corr_Q_smap]

        # Smooth map to 15 arcmin
        print('Smooth maps to 15 arcmin')
        smaps = smooth_maps([q_map, Skychunk_Q], 2048)
        q_smap = smaps[0]
        Q_pl_smap = smaps[1]

        Q_tomo = q_smap*dust_smap
        q_planck_smap = Q_pl_smap/dust_smap

        diff_q, corr_q, res_q = compare(pix, q_smap, q_planck_smap, sigma_p)
        diff_Q, corr_Q, res_Q = compare(pix, Q_tomo, Q_pl_smap, sigma_q)

        # diff plots
        plot_gnom(diff_q, lon, lat, 'q_frac_difference')
        #plot_gnom(corr_q, lon, lat, 'q_correlation')
        plot_gnom(diff_Q, lon, lat, 'Q_difference')
        #plot_gnom(corr_Q, lon, lat, 'Q_correlation')
        # smooth plots
        plot_gnom(q_smap, lon, lat, 'q_frac_tomo_smooth15arc')
        plot_gnom(q_planck_smap, lon, lat, 'planck353_q_frac_smooth15arc')
        plot_gnom(Q_tomo, lon, lat, 'Q_tomo_smooth15arc')
        plot_gnom(Q_pl_smap, lon, lat, 'planck353_Q_smooth15arc')
        # correlation plots
        plot_corr(Q_tomo, Q_pl_smap, 'Scatter_corr_Q')
        plot_corr(q_smap, q_planck_smap, 'Scatter_corr_q_frac')
        """
    if Upol == True:
        print('-- U polarisation --')
        if read is True:
            tot_res, frac_res, dust = map_analysis_function(u_smap, U_smap,\
                                                            dust_smap, mask)
        else:
            tot_res, frac_res, dust = map_analysis_function(u_map, U,\
                                                            dust353, mask)

        return(tot_res, frac_res, dust, lon, lat)

        """
        U_tomo_smap = get_Stokes(u_smap, dust_smap, mask)
        u_planck_smap = get_Fractional(U_smap, dust_smap, mask)

        # Difference:
        diff_U = Difference(U_tomo_smap, U_smap, mask)
        diff_u_frac = Difference(u_smap, u_planck_smap, mask)

        # Correlation maps:
        corr_U_smap = Correlation(U_tomo_smap, U_smap, mask)
        corr_u_frac_smap = Correlation(u_smap, u_planck_smap, mask)

        frac_res = [u_smap, u_planck_smap, diff_u_frac, corr_u_smap]
        tot_res = [U_tomo_smap, U_smap, diff_U, corr_U_smap]

        # Smooth map to 15 arcmin
        print('Smooth maps to 15 arcmin')
        smaps = smooth_maps([u_map, Skychunk_U], 2048)
        U_pl_smap = smaps[1]
        u_smap = smaps[0]

        U_tomo = u_smap*dust_smap
        u_planck_smap = U_pl_smap/dust_smap

        diff_u, corr_u, res_u = compare(pix, u_smap, u_planck_smap, sigma_p)
        diff_U, corr_U, res_U = compare(pix, U_tomo, U_pl_smap, sigma_u)

        # diff plots
        plot_gnom(diff_u, lon, lat, 'u_frac_difference')
        #plot_gnom(corr_u, lon, lat, 'u_correlation')
        plot_gnom(diff_U, lon, lat, 'U_difference')
        #plot_gnom(corr_U, lon, lat, 'U_correlation')
        # smooth plots
        plot_gnom(u_smap, lon, lat, 'u_frac_tomo_smooth15arc')
        plot_gnom(u_planck_smap, lon, lat, 'planck353_u_smooth15arc')
        plot_gnom(U_tomo, lon, lat, 'U_tomo_smooth15arc')
        plot_gnom(U_pl_smap, lon, lat, 'planck353_U_smooth15arc')
        # correlation plots
        plot_corr(U_tomo, U_pl_smap, 'Scatter_corr_U')
        plot_corr(u_smap, u_planck_smap, 'Scatter_corr_u_frac')
        """

    #plt.figure()
    #plt.hexbin(l, b, p_map, bins=1)
    #plt.gca().invert_xaxis()
    #plt.savefig('Figures/tomography/hex_map.png')

    #plot_gnom(R1_map, lon, lat, 'R1mag')


    #plt.show()

########################
a = [np.arange(10), np.arange(10)*2., np.arange(10)*3]
#print(a)
f = h5py.File('test.h5', 'w')
f.create_dataset('test', data=a)
f.close()

f = h5py.File('test.h5', 'r')
dt = np.asarray(f['test'])
f.close()

print(dt)
print(len(dt[1]), np.shape(dt))


########################

tomofile = 'Data/total_tomography.csv'
planckfile = 'Data/HFI_SkyMap_353-psb-field-IQU_2048_R3.00_full.fits'
dustfile = 'Data/dust_353_commander_temp_n2048_7.5arc.fits'

colnames = ['ra', 'dec', 'p', 'p_er', 'q', 'q_er', 'u', 'u_er', 'dist',\
            'dist_low', 'dist_up', 'Rmag1']
#load_planck_map(planckfile)


#main(tomofile, colnames, planckfile, dustfile, Ppol=True)
#main(tomofile, colnames, planckfile, dustfile, Qpol=True, write=True)
#Tot_res, frac_res, dust_map, lon, lat = main(tomofile, colnames, planckfile,\
#                                            dustfile, Upol=True, write=True)

# Plotting:
#plot_corr(Tot_res[0], Tot_res[1], 'test/corr_U1')
#plot_corr(frac_res[0], frac_res[1], 'test/corr_u_frac1')
#plot_gnom(Tot_res[0], lon, lat, 'test/U_1')




# plt.show()
#
