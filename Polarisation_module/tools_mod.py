"""
Module containing analytic functions and helping functions.
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



###########################################

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
    coord = SkyCoord(ra=ra*u_.deg, dec=dec*u_.deg, frame='icrs')
    coord_gal = coord.galactic

    lon = coord_gal.l.value
    lat = coord_gal.b.value

    return(lon, lat)

def theta_Eq(p, q, u, psi):
    """
    Find the parameter theta_gal from equitorial angle.

    Parameters:
    -----------

    return:
    -----------
    """
    cos2t = -q/p
    sin2t = u/p
    psi = psi*np.pi/180.
    theta_q = np.arccos(cos2t)/2 #- np.pi/2

    theta_u = np.arcsin(sin2t)
    theta_u[theta_u < 0.] += np.pi/2
    theta = np.arctan(u/q) # -u ??
    theta[theta < 0.] += np.pi/2
    x = (0.5*np.arctan2(u, q))
    x[x <0.] += np.pi/2
    psi[psi <0.] += np.pi/2
    print(np.mean(theta_q), np.min(theta_q), np.max(theta_q))
    print(np.mean(theta_u), np.min(theta_u), np.max(theta_u))
    print(np.mean(theta), np.min(theta), np.max(theta))
    print(np.mean(psi), np.min(psi), np.max(psi))
    print(np.mean(x), np.min(x), np.max(x))
    print(np.min(x)*180/np.pi, np.max(x)*180/np.pi)
    #print(psi*180/np.pi)
    #sys.exit()
    """
    plt.figure()
    plt.hist(theta_q - theta_u, bins=50)
    plt.title(r'$\theta_q - \theta_u$')
    plt.savefig('theta_diff_qu.png')
    plt.figure()
    plt.hist(theta - theta_q, bins=50)
    plt.title(r'$\theta - (\pi/4 - \theta_q)$')
    plt.savefig('theta_diff_q.png')
    plt.figure()
    plt.hist(theta - theta_u, bins=50)
    plt.title(r'$\theta - \theta_u$')
    plt.savefig('theta_diff_u.png')
    plt.figure()
    plt.hist(theta - psi, bins=50)
    plt.title(r'$\theta - \psi$')
    plt.savefig('theta_diff_psi.png')
    plt.figure()
    plt.hist(x - psi, bins=50)
    plt.title(r'$\theta_2 - \psi$')
    plt.savefig('theta2_diff_psi.png')

    plt.figure()
    plt.hist(theta_q, bins=50)
    plt.title(r'$\pi/4 - \theta_q$')
    plt.savefig('theta_q.png')
    plt.figure()
    plt.hist(theta_u, bins=50)
    plt.title(r'$\theta_u$')
    plt.savefig('theta_u.png')
    plt.figure()
    plt.hist(theta, bins=50)
    plt.title(r'$\theta$')
    plt.savefig('theta.png')
    plt.figure()
    plt.hist(psi, bins=50)
    plt.title(r'$\psi$')
    plt.savefig('psi.png')
    plt.figure()
    plt.hist(x, bins=50)
    plt.title(r'$\theta_2$')
    plt.savefig('theta2.png')

    sys.exit()
    #"""
    #print(np.mean(cos2t**2 + sin2t**2))
    # test if equal:
    d = theta_q - theta_u
    #print(d)
    print(np.mean(theta)*180/np.pi)
    #if np.abs(np.mean(d)) < 1e-5:
    return(theta)
    #else:
    #    print('theta_q != theta_u, check input')
    #    sys.exit()

def get_theta_gal(ra, dec, polang, aG=122.86, dG=27.13):
    """
    Calculate theta_eq - theta_gal, using equation 16 in Hutsemeters 97.

    Parameters:
    -----------
    - ra, array.            The right ascensions of the objects
    - dec. array.           The declination of the objects.
    - polang, array.        The polarisation angles in eqiutorial coord.
    - (aG, dG), scalars.    The eq. coords of the Northern galactic pole
                            (192.86, 27.13). If aG = 123 reduce to no rotation
                            of pol.angles??

    return:
    -----------
    - diff, array.          The result of tan(x - x_N), where x is the pol.angles
                            in equitorial coords, and x_N is the polarisation
                            angle in the new frame.
    """
    torad = np.pi/180.
    dG, aG = dG*torad, aG*torad
    dec, ra = dec*torad, ra*torad

    X = np.sin(aG - ra)
    Y = np.tan(dG)*np.sin(dec) - np.sin(dec)*np.cos(aG - ra)
    #tandiff = np.sin(aG-ra)/(np.tan(dG)*np.sin(dec) - np.sin(dec)*np.cos(aG-ra))
    #diff = np.arctan(tandiff)
    diff = np.arctan2(X, Y)

    theta_eq = polang*np.pi/180.
    theta_gal = theta_eq - diff  # + or - ??
    theta_gal[theta_gal<0] += np.pi
    theta_gal[theta_gal>=np.pi] -= np.pi
    #print(diff)
    return(diff)


def rotate_pol(ra, dec, p, q, u, polang):
    """
    Rotate the polarisation angle from Equitorial to Galactic coordinates.

    Parameters:
    -----------
    - ra, array.            The right ascensions of the objects
    - dec. array.           The declination of the objects.
    - p, array.             total polarisation fraction.
    - q, array.             The fractional q polarisation.
    - u, array.             The fractional u polarisation.
    - polang, array.        The polarisation angles in eqiutorial coordinates.

    return:
    -----------
    - q_gal, array.         The q polarisation in galactic coord.
    - u_gal, array.         The u polarisation in galactic coord.
    """
    # testing:
    print(np.mean(p))
    print(np.mean(np.sqrt(q**2 + u**2)))
    print(np.mean(p - np.sqrt(q**2 + u**2)))

    theta_gal = get_theta_gal(ra, dec, polang)
    theta_eq = theta_Eq(p, q, u, polang)
    #theta_eq = polang*np.pi/180.
    #theta_gal = theta_eq - diff  # + or - ??
    #theta_gal[theta_gal<0] += np.pi
    #theta_gal[theta_gal>=np.pi] -= np.pi
    #print((theta_gal*180/np.pi), '-')
    q_gal = p*np.cos(2.*theta_gal)
    u_gal = -p*np.sin(2.*theta_gal)
    return(q_gal, u_gal)



def Correlation(tomo_map, planck_map, mask, Nside=2048):
    """
    Compute a correlation map between tomograpy map and planck map, together
    with the correlation coefficients which is printed.

    Parameters:
    -----------
    - tomo_map, array.      The tomography map to use in the correlation
    - planck_map, array.    The planck map to correlate with the tomography map,
                            must be of same shape.
    - mask, array.          Pixels where the there is data inthe tomography map.
    - Nside, integer.       The resolution of the maps, default is 2048.

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

    Returns:
    -----------

    """
    Npix = hp.nside2npix(Nside)
    Diff = np.zeros(Npix)
    Diff[mask] = (tomo_map[mask] - planck_map[mask])/(tomo_map[mask])
    return(Diff)

def Krj2Kcmb(map, f_ref=353.):
    """

    Parameters:
    -----------

    Return:
    -----------
    """
    # Convert dust units to pol units, uK_RJ -> uK_cmb
    f, tau = cu.read_freq()
    f = f[6]
    tau = tau[6]
    ind = np.where((f > f_ref/2.) & (f < 2*f_ref))[0]
    f_353 = f[ind]
    dBdTrj = cu.dBdTrj(f_353)
    dBdTcmb = cu.dBdTcmb(f_353)

    Ucmb = cu.UnitConv(f_353, tau, dBdTrj, dBdTcmb, 'K_RJ', 'K_CMB', f_ref)
    return(Ucmb*map)

def get_Stokes(fractional, intensity, mask, Nside=2048):
    """
    Compute the Stokes parameter from a fractional polarisation

    Parameters:
    -----------

    Return:
    -----------
    """
    Npix = hp.nside2npix(Nside)
    X = np.full(Npix, hp.UNSEEN) # np.zeros(Npix)
    #print(np.shape(X), np.shape(X[mask]), np.shape(fractional))
    X[mask] = fractional[mask] * intensity[mask]
    return(X)

def get_Fractional(stokes, intensity, mask, Nside=2048):
    """
    Compute the fractional Stokes parameter from the Stokes parameter

    Parameters:
    -----------

    Return:
    -----------
    """
    Npix = hp.nside2npix(Nside)
    X = np.full(Npix, hp.UNSEEN) # np.zeros(Npix)
    X[mask] = stokes[mask]/intensity[mask]
    return(X)

def map_analysis_function(frac_tomo, PlanckIQU, dust_map, mask, Nside=2048):
    """
    Compute the maps to return, incluedes parameter maps, difference maps,
    correlation maps and dust map in use.

    Parameters:
    -----------

    Return:
    -----------
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

    #tomo_frac = np.zeros(len(frac_tomo))
    #Planck_IQU = np.zeros(len(PlanckIQU))
    tomo_frac = np.full(len(frac_tomo), hp.UNSEEN)
    Planck_IQU = np.full(len(PlanckIQU), hp.UNSEEN)
    tomo_frac[mask] = frac_tomo[mask]
    Planck_IQU[mask] = PlanckIQU[mask]

    # return maps: fractionals, stokes, dust
    frac_res = [tomo_frac, frac_planck_map, diff_frac, corr_frac]
    tot_res = [Tot_tomo_map, Planck_IQU, diff_Tot, corr_tot]
    ratios = []
    return(tot_res, frac_res, dust_map)

def ratio_S2V(tomo_map, frac_planck, mask, Nside=2048, Rv=3.1):
    """
    Compute the ratio of the polarisation fracitons of 353GHz and visual from
    stars.

    Parameters:
    -----------

    Return:
    -----------
    """
    Npix = hp.nside2npix(Nside)
    R = np.zeros(Npix)
    #Av = Rv*Ebv  # Ebv from greens et.al. 2019
    #tau = Av/1.086
    denominator = tomo_map[frac]/tau[frac]
    R[frac] = frac_planck[mask] / denominator
    pass

def ratio_P2p(fractional, IQU_planck, mask, Nside=2048):
    """
    Compute the polarisation fraction ratio, to check the efficentcy of producing
    polarised submillimeter emission. Units is same as IQU_planck

    Parameters:
    -----------

    Return:
    -----------
    """
    Npix = hp.nside2npix(Nside)
    R = np.zeros(Npix)

    R[mask] = IQU_planck[mask] / fractional[mask]
    return(R, np.sum(R[mask]), np.mean(R[mask]))

def ud_grade_maps(maplist, mask=None, Nside_in=2048, new_Nside=512):
    """
    Downgrade the maps from default resolution 2048 to a lower resolution. Set
    the empty pixels from 0 to UNSEEN. Maps must be of RING ordering.

    Parameters:
    -----------

    Return:
    -----------
    """
    Npix = hp.nside2npix(Nside_in)
    pixels = np.arange(Npix, dtype=int)
    if mask is None:
        ind0 = []
    else:
        ind0 = np.isin(pixels, mask, invert=True)
        print(len(mask), Npix, len(ind0))
    print(Npix, 12*new_Nside**2, len(pixels))
    print(ind0)

    new_maps = []
    # work with the one map in map list
    for i in range(len(maplist)):
        map = maplist[i]
        map[ind0] = hp.UNSEEN
        new_maps.append(map)
        print(len(map), hp.nside2npix(new_Nside))
        print(map)
    # Downgrade map
    out_maps = hp.ud_grade(new_maps, new_Nside, order_in='RING', order_out='RING')
    return(out_maps)



############

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
