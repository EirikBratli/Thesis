"""
Program to compare Planck 353Ghz polarization map to tomagraphy data of Raphael
etal 2019.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import h5py
import sys, time
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

    return(p_map, q_map, u_map, sigma_p, sigma_q, sigma_u, R1_map, pix)

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
                flip='geo')
    hp.graticule()
    plt.savefig(path + '{}.png'.format(label))



#######################
# Planck data 353 GHz #
#######################

def smoothing(map, Nside):
    FWHM = (np.sqrt(15.**2-5.**2)/60.) * (np.pi/180) # 15 arcmin
    smap = hp.sphtfunc.smoothing(map, fwhm=FWHM, iter=3)
    return(smap)

def load_planck_map(file, p=False):
    """
    Read the .fits file containing the 353GHz planck map.

    Parameters:
    -----------
    - file, string. Name of the planck file.
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
    plt.savefig('Figures/tomography/{}.png'.format(name))
    if y_lim is True:
        plt.ylim(-0.1, 0.1)
    if x_lim is True:
        plt.xlim(-1,1)

    if log is True:
        plt.figure('{} log'.format(name))
        plt.semilogy(tomo_map[ind], planck_map[ind], '.k')
        plt.xlabel('Tomography data')
        plt.ylabel('Planck 353GHz')
        plt.savefig('Figures/tomography/{}_log.png'.format(name))

def smooth_maps(maps, Nside):
    out_maps = []
    t0 = time.time()
    for i in range(len(maps)):
        t1 = time.time()
        m = smoothing(maps[i], Nside)
        out_maps.append(m)
        t2 = time.time()
        print('smoothing time map {}: {}s'.format(i, t2-t1))

    t3 = time.time()
    print('Total smoothing time: {}'.format(t3-t0))
    return(out_maps)

def Write_smooth_map(maps, name, Nside=2048, fits=False):
    """
    Write a smoothed map to file, using only the tomography pixels
    """
    smap = maps#smooth_maps(maps, Nside)
    tomo_maps = smap[0:3]
    pl_maps = smap[3:6]
    dust_map = smap[-1]
    print('Writing tomo maps:')
    #hp.fitsfunc.write_map('Data/{}_smoothed15arcmin_{}.fits'.\
    #                    format(name[0], Nside), tomo_maps)
    print('Writing planck maps:')
    #hp.fitsfunc.write_map('Data/{}_smoothed15arcmin_{}.fits'.\
    #                    format(name[1], Nside), tomo_maps)
    print('Writing dust maps')
    #hp.fitsfunc.write_map('Data/{}_smoothed15arcmin_{}.fits'.\
    #                    format(name[2], Nside), tomo_maps)

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


def main(tomofile, colnames, planckfile=None, dustfile=None, Ppol=False,\
        Qpol=False, Upol=False, write=False):
    """
    The main function of the program. Do all the calling to the functions used
    to calculate the comparison.

    Parameters:
    -----------
    - tomofile, string. The name of the tomography file.
    - colnames, list. List of the column names of the tomography file.
    - planckfile, string. Name of the planck file to compare with. Optional.

    Return:
    -------
    """

    data = load_tomographydata(tomofile, colnames)
    print(data[0,:])
    p_map, q_map, u_map, sigma_p, sigma_q, sigma_u, R1_map, pix = tomo_map(data)
    #sys.exit()
    ind = (np.where(p_map > 0)[0])
    ind1 = (np.where(q_map != 0)[0])
    ind2 = (np.where(u_map != 0)[0])
    print(len(ind), len(ind1), len(ind2))

    l, b = convert2galactic(data[:,0], data[:,1])

    lon = np.mean(l)
    lat = np.mean(b)
    print(lon, lat)
    print('pixel size, arcmin:', hp.pixelfunc.nside2resol(2048, arcmin=True))
    print('pixel size, radian:', hp.pixelfunc.nside2resol(2048))
    #sys.exit()
    # load planck
    if planckfile is not None:
        T, P, Q, U = load_planck_map(planckfile, p=True)
        d353 = load_planck_map(dustfile)
        dust353 = Krj2Kcmb(d353)
        hp.mollview(u_map)
        plot_gnom(U, lon, lat, 'U_full')


        Skychunk_T = get_PlanckSky_chunk(T, pix)
        Skychunk_P = get_PlanckSky_chunk(P, pix)
        Skychunk_Q = get_PlanckSky_chunk(Q, pix)
        Skychunk_U = get_PlanckSky_chunk(U, pix)
        Skychunk_dust = get_PlanckSky_chunk(dust353, pix)

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
        print('Smooth dust maps to 15 arcmin')
        #maps = smooth_maps([Skychunk_dust], 2048)
        #dust_smap = maps[0]
        if write is True:
            print('Write smoothed maps to file')
            Write_smooth_map([p_map, q_map, u_map, Skychunk_T, Skychunk_Q,\
                            Skychunk_U, dust353], ['pqu_tomo',\
                            'IQU_planck', 'I_dust'], fits=True)

            sys.exit()
        #
        #plot_gnom(Skychunk_T, lon, lat, 'planck353_T')
        #plot_gnom(Skychunk_dust, lon, lat, 'dust353')


        if Ppol == True:
            print('-- p polarisation --')

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


        if Qpol == True:
            print('-- Q polarisation --')

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

        if Upol == True:
            print('-- U polarisation --')

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


    #plt.figure()
    #plt.hexbin(l, b, p_map, bins=1)
    #plt.gca().invert_xaxis()
    #plt.savefig('Figures/tomography/hex_map.png')

    #plot_gnom(R1_map, lon, lat, 'R1mag')


    #plt.show()



tomofile = 'Data/total_tomography.csv'
planckfile = 'Data/HFI_SkyMap_353-psb-field-IQU_2048_R3.00_full.fits'
dustfile = 'Data/dust_353_commander_temp_n2048_7.5arc.fits'

colnames = ['ra', 'dec', 'p', 'p_er', 'q', 'q_er', 'u', 'u_er', 'dist',\
            'dist_low', 'dist_up', 'Rmag1']
#load_planck_map(planckfile)
#main(tomofile, colnames, planckfile, dustfile, Ppol=True)
#main(tomofile, colnames, planckfile, dustfile, Qpol=True, write=True)
main(tomofile, colnames, planckfile, dustfile, Upol=True)
