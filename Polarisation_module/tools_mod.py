"""
Module containing analytic functions and helping functions.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import h5py
import sys, time, glob, os
import scipy.optimize as spo
from scipy.optimize import minimize
import scipy.integrate as integrate
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
                            (NCP) coord: (123., 27.4) if use of longitude and
                            latitude in galactic coordinates for the objects

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
    Y = np.tan(dG)*np.cos(dec) - np.sin(dec)*np.cos(aG - ra)

    diff = np.arctan2(np.sin(aG - ra),\
                    np.tan(dG)*np.cos(dec) - np.sin(dec)*np.cos(aG - ra))

    theta_eq = polang*np.pi/180.
    theta_gal = theta_eq + diff  # + or - ??
    theta_gal[theta_gal<0.] += np.pi
    theta_gal[theta_gal>=np.pi] += -np.pi
    return(theta_gal, diff)#+45*np.pi/180)


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
    

    # method of RoboPol: (Raphael) I use this! 
    ln = 122.93200023  # vincent
    bn = 27.12843      # vincent
    theta_gal, diff = get_theta_gal(l, b, polang, aG=ln, dG=bn) # 2. best corr

    q_gal_raf = p*np.cos(2.*theta_gal)
    u_gal_raf = p*np.sin(2.*theta_gal)


    a = 2.*diff #+ 0.02             # give anti correlation
    
    a1 = -2*diff1
    q_gal = q*np.cos(a) - u*np.sin(a)
    u_gal = (q*np.sin(a) + u*np.cos(a))
    q_gal1 = q*np.cos(a1) - u*np.sin(a1)
    u_gal1 = (q*np.sin(a1) + u*np.cos(a1))
    
    return(q_gal, u_gal, theta_gal)

def error_polangle(p, e_p, ang, e_ang):
    """
    Compute the uncertainty of q and u in a new ref.frame. using p and
    polarisation angle with uncertainties.
    The angles must be in radians.
    """
    e_q = np.sqrt((e_p*np.cos(2*ang))**2 + (2*p*e_ang*np.sin(2*ang))**2)
    e_u = np.sqrt((e_p*np.sin(2*ang))**2 + (2*p*e_ang*np.cos(2*ang))**2)
    return(e_q, e_u)
    

def get_qu_polangerr(p, sp, evpa, e_evpa):
    """
    Calculate q, u with uncertainties and uncertainty of evpa.
    """
    
    # define params:
    N = len(p)
    eq = np.zeros(N)
    eu = np.zeros(N)
    evpa_err = np.zeros(N)
    # calculation:
    for i in range(N):
        snr = p/sp
        res = minimize(int_eq, [np.pi/50], args=(snr,), method='Nelder-Mead',\
                       tol=1e-5)
    
        evpa_err[i] = res.x[0]
        eq[i], eu[i] = error_polangle(p[i], sp[i], evpa[i], evpa_err[i])
    #
    return(eq, eu, evpa_err)
              
def MAS(p, sp):
    """Use the MAS estimator on the polarisation"""
    mas = p - sp**2/(2*p)*(1 - np.exp(-(p/sp)**2))
    return(mas)

def int_eq(sigma,snr):
    """ This is the integral of EVPA probability density from -sigma to sigma """
    integ = integrate.quad(lambda x: EVPA_pdf(x,snr),-sigma,sigma)
    return abs(integ[0] - 0.68268949)

    

def delta_psi(Qs, qv, Us, uv, plot=False, name='smooth0'):
    """
    Compute the difference in polarisation angles between submillimeter and visual.
    
    Parameters:
    - Qs, array/seq. The Q Stokes parameter for submillimeter polarisation
    - qv, array/seg. Same length as Qs, fractional Q Stokes parameter for 
    visula polarisation
    - Us, same as Qs only U Stokes parameter
    - uv, same as qv
    Returns:
    - psi: difference between pol.ang in submm and vis, psi_v, psi_s
    """
    X = (Us*qv - Qs*uv)
    Y = (Qs*qv + Us*uv)  # + or - Y ??
    psi = 0.5*np.arctan2(X, -Y)
    #psi[psi>0.] -= np.pi  # for a=2diff1
    #psi[psi>=np.pi] -= np.pi
    print('-------')

    print('Delta psi:', np.mean(psi)*180/np.pi, np.median(psi)*180/np.pi)
    print('+/-', np.std(psi)*180/np.pi)

    psi_s = 0.5*np.arctan2(Us, Qs)
    #psi_s[psi_s<0.] += np.pi/2
    #psi_s[psi_s>=np.pi] -= np.pi/2
    print(np.mean(psi_s)*180/np.pi,np.min(psi_s)*180/np.pi, np.max(psi_s)*180/np.pi)
    psi_v = 0.5*np.arctan2(uv, qv)
    #psi_v[psi_v<0.] += np.pi/2
    #psi_v[psi_v>=np.pi] -= np.pi/2
    print(np.mean(psi_v)*180/np.pi,np.min(psi_v)*180/np.pi, np.max(psi_v)*180/np.pi)
    dpsi = (psi_s + np.pi/2.) - psi_v 
    dpsi[dpsi > np.pi/2] -= np.pi

    print(np.mean(dpsi)*180/np.pi, np.median(dpsi)*180/np.pi)
    print('+/-', np.std(dpsi)*180/np.pi)
    mean_psi = np.mean(psi)*180/np.pi
    sig = np.std(psi)*180/np.pi
    print('dpsi:', mean_psi, 'sigma/n:', sig/np.sqrt(len(psi)))
    if plot is True:
        #c, b = np.histogram(psi*180/np.pi, bins=50)
        plt.figure()
        plt.hist(psi*180/np.pi, bins=10, histtype='step', color='k',\
                 density=True, stacked=True)
        #plt.hist(psi_v*np.pi/180, bins=50, histtype='step', color='r',\
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

def weightedmean(vals, valerrs, method='best', Niter=500):
    """
    Calculate the weighted mean and the standard error on weighted mean. 
    Use the method of Cochran 1977 to calculate the error on weighted mean, 
    Faster than bootstraping, almost as accurate. (Can choose Bootstraping)
    Returns the weighted mean and error on weighted mean.
    """
    weights = 1./valerrs**2
    wmerr2 = 1/np.sum(weights)
    wm = wmerr2*np.sum(vals*weights)

    if method == 'best':
        n = len(vals)
        if n == 1:
            wmerr = valerrs
        else:
            meanweight = np.mean(weights)
            A = np.sum((weights*vals-meanweight*wm)**2)
            B = -2*wm*np.sum((weights - meanweight) * (weights*vals - meanweight*wm))
            C = wm**2 * np.sum((weights - meanweight)**2)
            wmerr = np.sqrt(n/(n-1) * wmerr2**2 * (A + B + C))

    elif method == 'bootstrap':
        xmeans = np.zeros(Niter)
        sxm = np.zeros(Niter)
        for i in range(Niter):
            # resample te measurements
            resample_inds = bootstrap_resample(vals)
            
            # weigthed mean
            a, b = vals[resample_inds], valerrs[resample_inds]
            xmeans[i], sxm[i] = weightedmean(a,b, 'bootstrap')

        wmerr = np.std(xmeans)
    #
    return(wm, wmerr)

def bootstrap_resample(X, n=None):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns  indices to use for resampling-----(used to be X_resamples)
    """
    if n is None:
        n = len(X)
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    return(resample_i)

def IVC_cut(pixels, dist, distcut, Nside=256, clouds=None):
    """
    Find where the IVC is strong and in these pixels only use stars 
    located behind the IVC. For where the IVC is insignificant,
    use stars as before with distancecut=360 pc.
    
    Parameters:
    -----------
    - pixels, array:   The pixel of each star.
    - dist, array:     Array with the lower distances to the stars
    - distcut, scalar: The distance to the far side of the IVC
    - Nside, integer:  Resolution of the maps.
    
    return:
    -----------
    - cut, boolean sequence: True for stars behind the IVC and 
    stars not affected by the IVC, else False.
    """
    
    NH_IVC = np.load('Data/NH_IVC_from_spectrum.fits.npy')
    NH_IVC = hp.ud_grade(NH_IVC, nside_out=Nside, order_in='RING',\
                         order_out='RING')

    if len(distcut) == 1:
        distcut = distcut[0]
    else:
        distcut = distcut[1]
    NH_cond = 3.53e20 # cm^-2?
    cut = []
    if (clouds == 'all') or (clouds is None):
        print('Use LVC and IVC stars at distance larger than {} pc'.\
              format(distcut))
        for i, pix in enumerate(pixels):
            # Loop over the pixels of the stars
            if (NH_IVC[pix] > NH_cond) and (dist[i] > distcut):
                # find the stars located behind the IVC
                cut.append(True)

            elif (NH_IVC[pix] <= NH_cond):
                # keep the stars where the IVC is weak
                cut.append(True)

            else:
                # dont keep the stars infront of the IVC
                cut.append(False)
        #
        return(cut)
    elif clouds == 'LVC':
        print('Use LVC stars')
        # Only use LOS with 1 cloud, i.e. no IVC stars
        for i, pix in enumerate(pixels):
            # Loop over the pixels of the stars
            if (NH_IVC[pix] <= NH_cond):
                # keep the stars where the IVC is weak
                cut.append(True)

            else:
                # dont keep the stars infront of the IVC
                cut.append(False)
        #
        return(cut)

    elif clouds == 'IVC':
        print('Use only IVC stars at distance larger than {} pc'.\
              format(distcut))
        # Only use stars affected by the IVC (+ LVC)
        for i, pix in enumerate(pixels):
            # Loop over the pixels of the stars
            if (NH_IVC[pix] > NH_cond) and (dist[i] > distcut):
                # keep the stars where the IVC is weak
                cut.append(True)

            else:
                # dont keep the stars infront of the IVC
                cut.append(False)
        #
        return(cut)

    else:
        print('Clouds must be either None or 1.')
        sys.exit()


def sample_error(params, qu, model_func):
    """
    Compute the uncertainty of sampling model, star model and background.

    Input:
    - params, ndarray (2, Niter, Nparams=Npix)
    - qu ndarray (2, Npix=Nparams)
    - model_func, function that returns the model
    """  
    
    star = np.zeros((2, len(params[0,:-2]), len(params[:,0])))
    model = np.zeros(np.shape(star))
    for i in range(len(params[:,0])):
        star[:,:,i] = model_func(params[i,:], qu, star=True)
        model[:,:,i] = model_func(params[i,:], qu, star=True)
    
    bkgr_err = np.std(params[:,-2:], axis=0)
    star_err = np.std(star, axis=2)
    model_err = np.std(model, axis=2)
    return(model_err, star_err, bkgr_err)

def extinction_correction(l, b, dist):
    """
    Correction for extinction, using extinction data from Green19
    Work in galactic coordinated.
    """
    N = len(l)
    correction = np.zeros(N)
    
    # Get extinction data
    file1 = np.load('Data/Av_los_RoboPol.npz')
    
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
    Compute the polarisation fraction ratio, to check the efficentcy of 
    producing polarised submillimeter emission. Units is same as 
    IQU_planck [K_cmb]

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
    - maplist, list/seq. List of maps to be down/up graded.
    Return:
    -----------
    - out_maps, list of resolution adjusted maps 
    """
    Npix = hp.nside2npix(Nside_in)
    pixels = np.arange(Npix, dtype=int)
    if mask is None:
        ind0 = []
    else:
        ind0 = np.isin(pixels, mask, invert=True)
        print(len(mask), Npix, len(ind0))

    new_maps = []
    # work with the one map in map list
    for i in range(len(maplist)):

        map = maplist[i]
        map[ind0] = hp.UNSEEN
        new_maps.append(map)

    # Downgrade map
    out_maps = hp.ud_grade(new_maps, new_Nside, order_in='RING',\
                           order_out='RING')
    return(out_maps)
    
def Chi2(Q, U, q, u, C_ij, sq, su, sampler=False):
    """
    Compute the chi^2 of the Stokes parameters for Planck and star 
    polarisation. Follow the chi^2 computation of Planck XII 2015. The 
    data arrays must be masked.
    
    Parameters:
    -----------
    - Q,U, arrays.   Stokes parameters from Planck. In units uKcmb
    - q,u, arrays.   Fractional stokes parameters from stellar data
                     like RoboPol.
    - C_ij, ndarray. The covariance elements of Planck. 
                     Only use ij = QQ, UU and QU. In units Kcmb^2
    - sq,su. arrays. Uncertainty of polarisation fractions of stellar data.
    - sampler, bool. If true the covariance matrix is estimated from sampling,
                     else use Planck covariance matrix. Default is False.

    Returns:
    ----------
    params, sigmas, chi2: The best fit parameters with uncertainties and 
                          chi2 value
    """
    
    unit = (287.45*1e-6) # convertion factor between uKcmb and MJy/sr
    if sampler is False:
        Q = Q 
        U = U
        q = q
        u = u
        C_qu = C_ij[1,:] *(1e6)**2 # to uKcmb^2
        C_qq = C_ij[0,:] *(1e6)**2
        C_uu = C_ij[2,:] *(1e6)**2
        sq = sq
        su = su

    else:
        C_qq = C_ij[0,:] *(1e6)**2
        C_uu = C_ij[1,:] *(1e6)**2
        C_qu = C_ij[2,:] *(1e6)**2
        

    def min_func(param, Q=Q, U=U, q=q, u=u, C_qu=C_qu, C_qq=C_qq,\
                 C_uu=C_uu, sq=sq, su=su):
        """
        Minimizing function for the chi^2 test:

        V.shape = 2,K, V.T: K,2
        M.shape = 2,2,K
        indexes: i=2, j=50, k=2
        """
        a = param[0]
        b = param[1]

        if Q is None:
            # estimate chi^2 for Uu
            V = U - a*u - b
            M = C_uu + a**2*su**2
            Minv = 1./M
            
            d = V*Minv*V.T

        elif U is None:
            # estimate chi^2 for Qq
            V = Q - a*q - b
            M = C_qq + a**2*sq**2
            Minv = 1./M
            
            d = V*Minv*V.T
            
        else: 
            # joint chi^2 estimate
            V = np.array([Q - a*q - b, U - a*u - b])
            M = np.array([[C_qq + a**2*sq**2, C_qu], [C_qu, C_uu + a**2*su**2]])
            
            Minv = np.linalg.inv(M.T).T
            c = np.einsum('ikj,jk->ij', Minv, V.T)
            d = (np.einsum('ij,ij->j', V, c))

        return(np.sum(d))
    
    res = spo.fmin_powell(min_func, x0=[-18855, 0], full_output=True,\
                          retall=True)
    
    full_ab = np.asarray(res[-1])
    sigma = np.std(full_ab, axis=0)
    params = res[0]
    chi2 = res[1]
    print(2*len(C_ij[0,:])-len(params)) # degrees of freedom
    print('chi^2 = ', chi2)
    print('Reduced chi^2 =', chi2/(2*len(C_ij[0,:])-len(params)))
    print('ax + b = {}x + {} [uK_cmb]'.format(params[0], params[1]))
    print('ax + b = {}x + {} [MJy/sr]'.\
          format(params[0]*287.45*1e-6, params[1]*287.45*1e-6))
    print('sigma_a, sigma_b =',sigma*287.45*1e-6, '[MJy/sr]')
    
    return(params, sigma, chi2)

def get_P(q, u):
    """
    Calculate P or p from the Stokes parameters
    """
    return(np.sqrt(q**2 + u**2))

def get_P_err(q, u, sq, su):
    """
    Estimate the uncertainty of P
    """
    return(np.sqrt((sq*q)**2 + (su*u)**2)/get_P(q, u))

def get_p_dust(Ps, I_dust):
    """
    Estimate p_dust.
    """
    return(Ps/I_dust)

def get_p_star(p_v, p_d):
    """
    Estimate the polarisation from the stars. Using minimisation 
    and fmin_powell method
    """
    
    def min_func(fac, p_v=p_v, p_d=p_d):
        return(np.sum(((p_d - fac*p_v)/(np.abs(p_d))))**2)
    
    factor = spo.fmin_powell(min_func, x0=1)
    print(factor)

    p_star = factor*p_v
    #mask = p_star > p_d # if traces full los of dust
    #p_star[mask] = p_d[mask]
    return(p_star)

def get_pol_star(pol_vis, pol_submm):
    """
    Estimate the polarization (Q and U) contribution from stellar space. 
    Convertin visual polarisation to submillimeter polarisation
    """
    def min_func(fac, pol_vis=pol_vis, pol_submm=pol_submm):
        return(np.sum(((pol_submm - fac*pol_vis)/(np.abs(pol_submm)))**2))
    
    factor = spo.fmin_powell(min_func, x0=1)
    print(factor, factor*287.45e-6)
    
    pol_star = factor*pol_vis
    #mask = pol_star >= pol_submm
    #pol_star[mask] = pol_submm[mask]
    return(pol_star)


def get_pol_bkgr(pol_dust, pol_star, x=None):
    """
    Estimate background polarisation. p_bkgr = p_dust - p_stars
    Q_bkgr = Q_s - Q_star, U_bkgr = U_s - U_star
    """

    pol_bkgr = pol_dust - pol_star
    mask = pol_star > pol_dust
    #print(pol_dust - pol_star)
    #pol_bkgr[mask] = 0
    #q_bkgr = p_bkgr*np.cos(2*x)
    #u_bkgr = p_bkgr*np.sin(2*x)
    return(pol_bkgr, mask) #(p_bkgr, q_bkgr, u_bkgr, mask)


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


