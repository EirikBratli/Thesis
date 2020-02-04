"""
Main program in the Tomography module.
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
import smoothing_mod as smooth
import plotting_mod as plotting
import load_data_mod as load


####################################

def main(tomofile, colnames, planckfile, dustfile, Ppol=False, Qpol=False,\
        Upol=False, write=False, read=False, ud_grade=True, newNside=512, res=15):
    """
    The main function of the program. Do all the calling to the functions used
    to calculate the comparison between the Tomography data and Planck
    polarisation data. Want to smooth to uK_cmb.

    Parameters:
    -----------
    - tomofile, string.     The name of the tomography file.
    - colnames, list.       List of the column names of the tomography file.
    - planckfile, string.   Name of the planck file to compare with.
    - dustfile, string.     Name of the dust intensity file.

    Return:
    -------
    """

    data = load.load_tomographydata(tomofile, colnames)
    print(data[0,:])

    p_map, q_map, u_map, sigma, r_map, pix = load.tomo_map(data)

    u_map = -u_map
    mask = np.unique(pix)
    #print(q_map[mask]**2 + u_map[mask]**2)
    l, b = tools.convert2galactic(data[:,0], data[:,1])

    lon = np.mean(l)
    lat = np.mean(b)
    print(lon, lat)
    print('pixel size, arcmin:', hp.pixelfunc.nside2resol(2048, arcmin=True))
    print('pixel size, radian:', hp.pixelfunc.nside2resol(2048))
    #sys.exit()

    names = ['pqu_tomo', 'IQU_planck', 'I_dust']
    # write smoothed maps
    if write is True:
        # load planck
        print('load planck 353GHz data')
        T, P, Q, U = load.load_planck_map(planckfile, p=True)
        d353 = load.load_planck_map(dustfile)
        dust353 = tools.Krj2Kcmb(d353)
        T = T*1e6
        P = P*1e6
        Q = Q*1e6
        U = U*1e6
        # if ud_grade is true, need new_nside, smoothing resloution
        if ud_grade is True:
            Nside = newNside
            #p_map, q_map, u_map, sigma, r_map, pix = load.tomo_map(data, Nside=Nside)
            print(newNside, 'Down grade maps')
            new_maps = tools.ud_grade_maps([U, u_map], mask, new_Nside=newNside)

            hp.mollview(new_maps[0])
            hp.mollview(new_maps[1])
            plt.show()
        else:
            pass

        print('Write smoothed maps to file')
        sys.exit()
        smooth.Write_smooth_map([p_map, q_map, u_map, T, Q, U, dust353],\
                                ['tomo','IQU', 'dust'], Nside=Nside, res=res)
        # ['pqu_tomo', 'IQU_planck', 'I_dust'] used to be
        #Write_smooth_map([p_map], ['p_tomo2'], iterations=3)

        sys.exit()
    #
    # Read in  smoothed maps
    if read is True:
        print('Load smoothed maps')
        smaps = smooth.read_smooth_maps(names)
        if smaps[-1] == 'all':
            T_smap = smaps[0] * 1e6
            Q_smap = smaps[1] * 1e6
            U_smap = smaps[2] * 1e6
            p_smap_sph, q_smap_sph, u_smap_sph = smaps[3:6] # spherical harmonic
            dust_smap = smaps[-2] * 1e6

        elif smaps[-1] == 'planck':
            T_smap, Q_smap, U_smap = smaps[0:3]

        elif smaps[-1] == 'tomo':
            p_smap_sph, q_smap_sph, u_smap_sph = smaps[0:3] # spherical harmonic

        elif smaps[-1] == 'dust':
            dust_smap = smaps[0] * 1e6

        print('smoothed maps loaded')
        #full_smaps = [T_smap, Q_smap, U_smap]
        #print(np.shape(smaps))
        print(np.mean(dust_smap[mask]), np.mean(U_smap[mask]))

    else:
        print('Use non smoothed maps')
        # load planck
        print('load planck 353GHz data')
        T, P, Q, U = load.load_planck_map(planckfile, p=True)
        d353 = load.load_planck_map(dustfile)
        dust353 = tools.Krj2Kcmb(d353) * 1e6
        T = T*1e6
        P = P*1e6
        Q = Q*1e6
        U = U*1e6

        #hp.mollview(p_map)

    #
    """
    Work with smoothed maps reduced to given area of Tomography data
    """

    if Ppol == True:
        print('-- P polarisation --')
        p_smap = smooth.smooth_tomo_map(p_map, mask)
        if read is True:
            full_IQU = [T_smap, Q_smap, U_smap]
            tot_res, frac_res, dust = tools.map_analysis_function(p_smap, T_smap,\
                                                            dust_smap, mask)

        else:
            full_IQU = [T, Q, U]
            tot_res, frac_res, dust = tools.map_analysis_function(p_map, T,\
                                                            dust353, mask)

        return(tot_res, frac_res, dust, lon, lat, full_IQU, mask, r_map)

    elif Qpol == True:
        print('-- Q polarisation --')
        q_smap = smooth.smooth_tomo_map(q_map, mask)
        if read is True:
            full_IQU = [T_smap, Q_smap, U_smap]
            tot_res, frac_res, dust = tools.map_analysis_function(q_smap, Q_smap,\
                                                            dust_smap, mask)
        else:
            full_IQU = [T, Q, U]
            tot_res, frac_res, dust = tools.map_analysis_function(q_map, Q,\
                                                            dust353, mask)

        return(tot_res, frac_res, dust, lon, lat, full_IQU, mask, r_map)

    elif Upol == True:
        print('-- U polarisation --')
        u_smap = smooth.smooth_tomo_map(u_map, mask)
        if read is True:
            full_IQU = [T_smap, Q_smap, U_smap]
            tot_res, frac_res, dust = tools.map_analysis_function(u_smap, U_smap,\
                                                            dust_smap, mask)

        else:
            full_IQU = [T, Q, U]
            tot_res, frac_res, dust = tools.map_analysis_function(u_map, U,\
                                                            dust353, mask)
        #hp.mollview(tot_res[1])
        #plt.show()
        return(tot_res, frac_res, dust, lon, lat, full_IQU, mask, r_map)



########################
tomofile = 'Data/total_tomography.csv'
planckfile = 'Data/HFI_SkyMap_353-psb-field-IQU_2048_R3.00_full.fits'
dustfile = 'Data/dust_353_commander_temp_n2048_7.5arc.fits'

colnames = ['ra', 'dec', 'p', 'p_er', 'q', 'q_er', 'u', 'u_er', 'dist',\
            'dist_low', 'dist_up', 'Rmag1']

########################


#load_planck_map(planckfile)


#main(tomofile, colnames, planckfile, dustfile, Ppol=True)
#main(tomofile, colnames, planckfile, dustfile, Qpol=True, write=True)

# input args from commando line:

if len(sys.argv) == 1:
    print('Need input arguments:')
    print('(write/read/unsmooth), (Upol/Qpol/Ppol/ud) and (plot/newNside)')
    sys.exit()

elif len(sys.argv) > 1:
    print('----------------')
    if sys.argv[1] == 'write':
        # Write smoothed maps:
        if (sys.argv[2] == 'ud'):
            if (len(sys.argv) == 4) and (sys.argv[3] != 'plot'):
                main(tomofile, colnames, planckfile, dustfile, write=True,\
                        ud_grade=True, newNside=int(sys.argv[3]))
                #print('lol')

            else:
                print('Need a valid Nside, meaning a integer')
                sys.exit()
        else:
            main(tomofile, colnames, planckfile, dustfile, write=True)

    elif sys.argv[1] == 'read':
        # Read in smoothed maps:
        if sys.argv[2] == 'Upol':
            Tot_res, frac_res, dust, lon, lat, IQU, mask, dist = main(tomofile,\
                        colnames, planckfile, dustfile, Upol=True, read=True)

        elif sys.argv[2] == 'Qpol':
            Tot_res, frac_res, dust, lon, lat, IQU, mask, dist = main(tomofile,\
                        colnames,planckfile, dustfile, Qpol=True, read=True)

        elif sys.argv[2] == 'Ppol':
            Tot_res, frac_res, dust, lon, lat, IQU, mask, dist = main(tomofile,\
                        colnames,planckfile, dustfile, Ppol=True, read=True)

    elif sys.argv[1] == 'unsmooth':
        # Read in smoothed maps:
        if sys.argv[2] == 'Upol':
            Tot_res, frac_res, dust, lon, lat, IQU, mask, dist = main(tomofile,\
                                    colnames, planckfile, dustfile, Upol=True)

        elif sys.argv[2] == 'Qpol':
            Tot_res, frac_res, dust, lon, lat, IQU, mask, dist = main(tomofile,\
                                    colnames,planckfile, dustfile, Qpol=True)

        elif sys.argv[2] == 'Ppol':
            Tot_res, frac_res, dust, lon, lat, IQU, mask, dist = main(tomofile,\
                                    colnames,planckfile, dustfile, Ppol=True)

        # the returned lists are sorted like:
        # [tomo_map, planck_map, difference_map, correlation_map]

    else:
        print('First input argument should be "write", "read" or "unsmooth"')
        sys.exit()
    print('----------------')

if (len(sys.argv) == 4) and (sys.argv[3] == 'plot'):
    arg = sys.argv[2].split('p')[0]
    print('Plotting for {} polarisation'.format(arg))
    if arg == 'P':
        ind = 0
    elif arg == 'Q':
        ind = 1
    elif arg == 'U':
        ind = 2
    print(ind)
    # Plotting:
    if sys.argv[3] == 'plot':
        # ratios:
        plotting.plot_ratio(frac_res[0], Tot_res[1], mask)
        sys.exit()

        # Correlation:
        plotting.plot_corr(Tot_res[0], Tot_res[1], 'corr_{}'.format(arg), mask,\
                    dist, xlab=r'Tomography ${}_{{frac}}\times I_d$'.format(arg),\
                    ylab=r'353 ${}$'.format(arg), title='{}'.format(arg))
        plotting.plot_corr(frac_res[0], frac_res[1], 'corr_{}_frac'.format(arg),\
                    mask, dist, xlab=r'Tomography ${}$'.format(arg),\
                    ylab=r'353 ${}_{{frac}}/I_d$'.format(arg),\
                    title='{}_frac'.format(arg))
        # not scaled:
        plotting.plot_corr(frac_res[0], Tot_res[1], 'corr_{}_unscaled'.format(arg),\
                    mask, dist, xlab=r'Tomography ${}_{{frac}}$'.format(arg),\
                    ylab=r'353 ${}$'.format(arg),\
                    title='{}_unscaled'.format(arg))
        #plot_gnom(Tot_res[3], lon,lat, 'test/corr_U_map1')
        #plot_gnom(frac_res[3], lon, lat, 'test/curr_u_frac_map1')

        # maps:
        plotting.plot_gnom(Tot_res[0], lon, lat, '{}'.format(arg), mask,\
                            unit=r'$K_{{cmb}}$', project='tomo')
        plotting.plot_gnom(frac_res[0], lon, lat, '{}_frac'.format(arg), mask,\
                            unit=r'$K_{{cmb}}$', project='tomo')
        plotting.plot_gnom(Tot_res[1], lon, lat, '{}'.format(arg), mask,\
                            unit=r'$K_{{cmb}}$', project='planck')
        plotting.plot_gnom(frac_res[1], lon, lat, '{}_frac'.format(arg), mask,\
                            unit=r'$K_{{cmb}}$', project='planck')

        ### Full plots
        plotting.plot_gnom(IQU[ind], lon, lat, 'full_{}'.format(arg),\
                            unit=r'$K_{{cmb}}$', project='planck')
        #plotting.plot_gnom(IQU[2], lon, lat, 'test/full_U_planck1')
        #plot_gnom(Tot_res[2], lon, lat, 'test/diff_U1')
        #plot_gnom(frac_res[2], lon, lat, 'test/diff_U_frac1')

    else:
        sys.exit()
