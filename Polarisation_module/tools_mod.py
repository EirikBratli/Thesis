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

def get_theta_gal(ra, dec, polang, aG=122.93200023, dG=27.12843):
    """
    Calculate theta_eq - theta_gal, using equation 16 in Hutsemeters 97.

    Parameters:
    -----------
    - ra, array.            The right ascensions of the objects
    - dec. array.           The declination of the objects.
    - polang, array.        The polarisation angles in eqiutorial coord.
    - (aG, dG), scalars.    The eq. coords of the Northern galactic pole
                            (192.86, 27.13). Or use Northern celestial pole
                            coord: (123., 27.4)

    return:
    -----------
    - diff, array.          The result of tan(x - x_N), where x is the pol.angles
                            in equitorial coords, and x_N is the polarisation
                            angle in the new frame.
    """
    torad = np.pi/180.
    dG, aG = dG*torad, aG*torad
    dec, ra = dec*torad, ra*torad
    print(aG)
    print(dG)

    X = np.sin(aG - ra)
    Y = np.tan(dG)*np.cos(dec) - np.sin(dec)*np.cos(aG - ra)

    diff = np.arctan2(np.sin(aG - ra),\
                    np.tan(dG)*np.cos(dec) - np.sin(dec)*np.cos(aG - ra))

    theta_eq = polang*np.pi/180.
    #print(np.mean(polang), np.min(polang), np.max(polang))
    theta_gal = theta_eq + diff  # + or - ??
    #print(theta_gal*180/np.pi)
    theta_gal[theta_gal<0.] += np.pi
    theta_gal[theta_gal>=np.pi] += -np.pi

    print(np.mean(diff*180/np.pi), np.min(diff)*180/np.pi, np.max(diff)*180/np.pi)
    print(np.mean(theta_gal)*180/np.pi, np.min(theta_gal)*180/np.pi, np.max(theta_gal)*180/np.pi)
    return(theta_gal, diff)


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

    l, b = convert2galactic(ra, dec)
    #"""
    # use astropy for this: Vincent use this!
    ag = 192.86948 # wiki
    dg = 27.12825  # wiki
    theta_gal1, diff1 = get_theta_gal(ra, dec, polang, aG=ag, dG=dg) # 1.
    print('-----')

    # method of RoboPol: (Raphael)
    ln = 122.93200023  # vincent
    bn = 27.12843      # vincent
    theta_gal, diff = get_theta_gal(l, b, polang, aG=ln, dG=bn) # 2. best corr

    print('----')
    q_gal_raf = p*np.cos(2.*theta_gal)
    u_gal_raf = p*np.sin(2.*theta_gal)


    a = 2.*diff               # give anti correlation
    #a1 = -np.pi/2. + 2*diff1
    a1 = -2*diff1
    q_gal = q*np.cos(a) - u*np.sin(a)
    u_gal = (q*np.sin(a) + u*np.cos(a))
    q_gal1 = q*np.cos(a1) - u*np.sin(a1)
    u_gal1 = (q*np.sin(a1) + u*np.cos(a1))
    print(np.mean(q_gal_raf), np.min(q_gal_raf), np.max(q_gal_raf))
    print(np.mean(q_gal), np.min(q_gal), np.max(q_gal))
    print(np.mean(q_gal1), np.min(q_gal1), np.max(q_gal1))
    print(np.mean(u_gal_raf), np.min(u_gal_raf), np.max(u_gal_raf))
    print(np.mean(u_gal), np.min(u_gal), np.max(u_gal))
    print(np.mean(u_gal1), np.min(u_gal1), np.max(u_gal1))
    print('Difference between BP method(lon, lat) and Rafaels method in; q_gal, u_gal:')
    print(np.mean(q_gal-q_gal_raf), np.mean(u_gal-u_gal_raf))
    print('Difference between BP method(ra, dec) and Rafaels method in; q_gal, u_gal:')
    print(np.mean(q_gal1-q_gal_raf), np.mean(u_gal1-u_gal_raf))
    #sys.exit()
    return(q_gal, u_gal)

def delta_psi(Qs, qv, Us, uv, plot=False, name='smooth0'):
    """
    Compute the difference in polarisation angles between submillimeter and visual
    """
    X = (Us*qv - Qs*uv)
    Y = (Qs*qv + Us*uv)  # + or - Y ??
    psi = 0.5*np.arctan2(X, -Y)
    #psi[psi>0.] -= np.pi  # for a=2diff1
    #psi[psi>=np.pi] -= np.pi
    print('-------')

    #print(np.min(psi)*180/np.pi, np.max(psi)*180/np.pi)
    print('Delta psi:', np.mean(psi)*180/np.pi, np.median(psi)*180/np.pi)
    print(np.std(psi)*180/np.pi)

    psi_s = 0.5*np.arctan2(-Us, Qs)
    #psi_s[psi_s<0.] += np.pi/2
    #psi_s[psi_s>=np.pi] -= np.pi/2
    print(np.mean(psi_s)*180/np.pi,np.min(psi_s)*180/np.pi, np.max(psi_s)*180/np.pi)
    psi_v = 0.5*np.arctan2(-uv, qv)
    #psi_v[psi_v<0.] += np.pi/2
    #psi_v[psi_v>=np.pi] -= np.pi/2
    print(np.mean(psi_v)*180/np.pi,np.min(psi_v)*180/np.pi, np.max(psi_v)*180/np.pi)
    dpsi = (psi_s + np.pi/2.) - psi_v  # Seems like psi_s is already rotated 90 deg
    dpsi[dpsi > np.pi/2] -= np.pi
    #print(dpsi*180/np.pi)
    #print(np.min(dpsi)*180/np.pi, np.max(dpsi)*180/np.pi)
    print(np.mean(dpsi)*180/np.pi, np.median(dpsi)*180/np.pi)
    print(np.std(dpsi)*180/np.pi)
    mean_psi = np.mean(psi)*180/np.pi
    sig = np.std(psi)*180/np.pi
    print('dpsi:', mean_psi, 'sigma/n:', sig/np.sqrt(len(psi)))
    if plot is True:
        #c, b = np.histogram(psi*180/np.pi, bins=50)
        plt.figure()
        plt.hist(psi*180/np.pi, bins=10, histtype='step', color='k',\
                 density=True, stacked=True)
        #plt.hist(dpsi*180/np.pi, bins=50, histtype='step', color='r',\
        #         density=True, stacked=True)
        plt.axvline(x=mean_psi, color='r', linestyle=':',\
                    label=r'$\Delta\psi_{{s/v}}={}^{{\circ}}\pm{}^{{\circ}}$'.\
                    format(round(mean_psi, 2),round(sig/len(psi),2)))
        plt.xlabel(r'$\Delta \psi_{{s/v}}$ [deg]')
        plt.ylabel('Probability density')
        # apply sigma, mean/median.
        plt.savefig('Figures/Delta_psi_sv_{}.png'.format(name))

        plt.show()
        #sys.exit()
    return(psi, psi_v, psi_s)

def extinction_correction(l, b, r_star):
    """
    Correction for extinction, using extinction data from Green19
    Work in galactic coordinated.
    """
    N = len(l)
    correction = np.zeros(N)
    
    # Get extinction data
    file1 = np.load('Av_los_RoboPol.npz')
    
    x = (file1['r'])
    ind = np.where(x > 360)[0]
    x = x[ind]
    lat = file1['b'][ind]
    lon = file1['l'][ind]
    Av = file1['Av'][:,ind]
    Av_err = file1['err'][:,ind]

    r_pol = np.sqrt(l**2 + b**2)
    r_Av = np.sqrt(lon**2 + lat**2)
    
    for i, r in enumerate(r_pol):
        temp1 = np.sqrt((r - r_Av)**2)
        temp2 = np.abs(dist[i] - x)

        ind1 = np.where(temp1 == np.min(temp1))[0]
        ind2 = np.where(temp2 == np.min(temp2))[0]

        correction[i] = Av[ind1,-1]/Av[ind1,ind2]
    #
    return(correction)
        

def sigma(s_in, N):
    """
    Find the uncertainty of values in a bin.
    """
    return(np.sqrt(np.sum(s_in**2))/float(N))

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
    print(np.mean(frac_tomo[mask]), np.mean(dust_map[mask]), np.mean(PlanckIQU[mask]))
    print('convert to stokes or fractional')
    Tot_tomo_map = get_Stokes(frac_tomo, dust_map, mask, Nside)
    frac_planck_map = get_Fractional(PlanckIQU, dust_map, mask, Nside)
    print(np.mean(Tot_tomo_map[mask]))
    # Difference:
    print('compute difference')
    diff_Tot = Difference(Tot_tomo_map, PlanckIQU, mask, Nside)
    diff_frac = Difference(frac_tomo, frac_planck_map, mask, Nside)
    #print(np.mean(diff_Tot[mask]))
    # correlation maps:
    print('compute correlation')
    corr_tot = Correlation(Tot_tomo_map, PlanckIQU, mask, Nside)
    corr_frac = Correlation(frac_tomo, frac_planck_map, mask, Nside)

    #tomo_frac = np.zeros(len(frac_tomo))
    #Planck_IQU = np.zeros(len(PlanckIQU))
    #print(len(frac_tomo), len(PlanckIQU))
    tomo_frac = np.full(len(frac_tomo), hp.UNSEEN)
    Planck_IQU = np.full(len(PlanckIQU), hp.UNSEEN)
    tomo_frac[mask] = frac_tomo[mask]
    Planck_IQU[mask] = PlanckIQU[mask]
    #print(frac_tomo[mask])
    #print(Planck_IQU[mask])
    #hp.mollview(tomo_frac)
    #hp.mollview(Planck_IQU)
    #hp.mollview(Tot_tomo_map)
    #hp.mollview(frac_planck_map)
    # return maps: fractionals, stokes, dust
    frac_res = [tomo_frac, frac_planck_map, diff_frac, corr_frac]
    tot_res = [Tot_tomo_map, Planck_IQU, diff_Tot, corr_tot]

    #plt.show()
    #sys.exit()
    return(tot_res, frac_res, dust_map)



def sigma_x(x, N):
    """
    Compute the standard deviation of tomography data for a bin.
    """

    s = np.sqrt(np.sum(x**2))/N
    return(s)

def Ebv_rate(Ebv_inf, Ebv_star, r_map):
    """
    Compute the ratio between reddening from background and reddening of star.
    """
    r = np.nan_to_num(r_map, 1)
    print(np.mean(r), np.median(r))
    f = ((np.mean(r) - r)/r)**2
    print(f)
    return(Ebv_inf / (Ebv_star*f))


def ratio_S2V(tomo_map, frac_planck, Ebv, mask, Nside=512, Rv=3.1):
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
    Av = Rv*Ebv  # Ebv from greens et.al. 2019
    tau = Av/1.086
    denominator = tomo_map[mask]/tau[mask]
    R[mask] = frac_planck[mask] / denominator
    return(R, np.mean(R[mask]))

def ratio_P2p(fractional, IQU_planck, mask, Nside=2048):
    """
    Compute the polarisation fraction ratio, to check the efficentcy of producing
    polarised submillimeter emission. Units is same as IQU_planck [K_cmb]

    Parameters:
    -----------

    Return:
    -----------
    """
    Npix = hp.nside2npix(Nside)
    R = np.zeros(Npix)

    R[mask] = IQU_planck[mask] / fractional[mask]
    return(R, np.mean(R[mask]))

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
    #print(Npix, 12*new_Nside**2, len(pixels))
    #print(ind0)

    new_maps = []
    # work with the one map in map list
    for i in range(len(maplist)):

        map = maplist[i]
        map[ind0] = hp.UNSEEN

        new_maps.append(map)
        print(len(map), hp.nside2npix(new_Nside))
        #print(map)
    # Downgrade map
    out_maps = hp.ud_grade(new_maps, new_Nside, order_in='RING', order_out='RING')
    return(out_maps)

def Chi2(Q, U, q, u, C_ij, sq, su, I=None, tau=None):
    """
    Compute the chi^2 of the Stokes parameters for Planck and star polarisation.
    The input arguments can be normalised to dimensionless variables. Q/I or q/tau
    Follow the chi^2 computation of Planck XII 2015. The data arrays must be masked.
    
    Parameters:
    -----------
    - Q,U, arrays.   Stokes parameters from Planck.
    - q,u, arrays.   Fractional stokes parameters from stellar data like RoboPol.
    - C_ij, ndarray. The covariance elements of Planck.
    - sq,su. arrays. Uncertainty of polarisation fractions of stellar data.
    - I, array.      Dust intensity. optional
    - tau, array.    Optical depth to the stars. Optional.
    """
    
    if I is None:
        Q = Q #* (287.45*1e-6)
        U = U #* (287.45*1e-6)
        q = q #* (287.45*1e-6)
        u = u #* (287.45*1e-6)
        C_qu = C_ij[4,:] *(1e6)**2#* (287.45)**2
        C_qq = C_ij[3,:] *(1e6)**2
        C_uu = C_ij[5,:] *(1e6)**2
        sq = sq #* (287.45*1e-6)
        su = su #* (287.45*1e-6)
    else:
        C_qu = (I**2*C_ij[4,:] + Q*U*C_ij[0,:] - I*Q*C_ij[2,:] - I*U*C_ij[1,:])/I**4
        C_qu = C_qu * (287.45*1e-6)**2
        q = q * (287.45*1e-6)/tau
        u = u * (287.45*1e-6)/tau
        sq = sq * (287.45*1e-6)/tau**2
        su = su * (287.45*1e-6)/tau**2
        Q = Q * (287.45*1e-6)
        U = U * (287.45*1e-6)
    #

    def min_func(param, Q=Q, U=U, q=q, u=u, C_qu=C_qu, C_qq=C_qq,\
                 C_uu=C_uu, sq=sq, su=su):
        """
        V.shape = 2,K, V.T: K,2
        M.shape = 2,2,K
        indexes: i=2, j=50, k=2
        """
        a = param[0]
        b = param[1]
        res = 0
        if Q is None:
            V = U - a*u - b
            M = C_uu + a**2*su**2
            Minv = 1./M
            
            d = V*Minv*V.T

        elif U is None:
            V = Q - a*q - b
            M = C_qq + a**2*sq**2
            Minv = 1./M
            
            d = V*Minv*V.T
            
        else:
            V = np.array([Q - a*q - b, U - a*u - b])
            M = np.array([[C_qq + a**2*sq**2, C_qu], [C_qu, C_uu + a**2*su**2]])
            
            Minv = np.linalg.inv(M.T).T
            c = np.einsum('ikj,jk->ij', Minv, V.T)
            d = (np.einsum('ij,ij->j', V, c))
        """
        for i in range(len(Q)):
            V = np.array([Q[i] - a*q[i] - b, U[i] - a*u[i] - b])
            #print(np.einsum('ij,ji->j',V, V.T))
            M = np.array([[C_qq[i] + a**2*sq[i]**2, C_qu[i]],\
                          [C_qu[i], C_uu[i] + a**2*su[i]**2]])
            #print(M, V)
            Minv = np.linalg.inv(M)
            #print(Minv)
            c = np.dot(Minv, V.T)
            d = np.dot(V, c)
            #print(np.linalg.eigvals(M))
            #print(d, c, V)
            if d < 0:
                break
            res += d
        #print(res)
        """

        #print(np.where(np.linalg.eigvals(M.T) < 0)[0])
        #sys.exit()
        
        #print(np.sum(d), res)
        #sys.exit()
        #Minv = np.linalg.inv(M)
        return(np.sum(d))
    #m = min_func([-5, 0])
    res = spo.fmin_powell(min_func, x0=[-18855, 0], full_output=True, retall=True)
    
    full_ab = np.asarray(res[-1])
    sigma = np.std(full_ab, axis=0)
    params = res[0]
    chi2 = res[1]
          
    print('chi^2 = ', chi2)
    print('Reduced chi^2 =', chi2/(2*len(C_ij[0,:])-len(params)))
    print('ax + b = {}x + {} [uK_cmb]'.format(params[0], params[1]))
    print('ax + b = {}x + {} [MJy/sr]'.\
          format(params[0]*287.45*1e-6, params[1]*287.45*1e-6))
    print('sigma_a, sigma_b =',sigma*287.45*1e-6, '[MJy/sr]')
    
    return(params, sigma, chi2)


def Read_H5(filename, name):
    """
    Function to read in a HDF file. Handles multicolumn data files, either 1 or
    3 columns.

    Parameters:
    -----------
    - filename, string. The name of the file to read.
    - name, string.     The column names of the data.

    Return:
    -----------
    - maps, array.      An array with the map is returned.

    """
    print(filename)
    f = h5py.File(filename, 'r')
    maps = np.asarray(f[name])
    f.close()
    print(np.shape(maps))
    return(maps)

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
