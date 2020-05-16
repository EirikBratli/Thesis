"""
Main program in the Tomography module, 2D analysis.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import h5py, argparse
import sys, time, glob, os
import scipy.optimize as spo

from astropy import units as u_
from astropy.coordinates import SkyCoord

import convert_units as cu
import tools_mod as tools
import smoothing_mod as smooth
import plotting_mod as plotting
import load_data_mod as load
import template_mod as template

####################################


def main(planckfile, dustfile, tomofile, colnames, names, pol, res,\
         part='all'):
    """
    The main function of the program. Do all the calling to the functions used
    to calculate the comparison between the Tomography data and Planck
    polarisation data. Want to smooth to uK_cmb.

    Parameters:
    -----------
    - planckfile, string.   Name of the planck file to compare with.
    - dustfile, string.     Name of the dust intensity file.
    - tomofile, string.     The name of the tomography file.
    - colnames, list.       List of the column names of the tomography file.
    - names, list.          List with the column names in the smoothed planck
                            maps, with polarisation first then dust intensity.
    - pol, bool.            Which Stokes parameter to evaluate.

    Return:
    -------
    """
    if (pol == 'P') or (pol == 'Q') or (pol == 'U'):
        polarisation = True
    elif (pol == 'p') or (pol == 'q') or (pol == 'u') or (pol == 'qu'):
        polarisation = True
    else:
        polarisation = False

    print(pol, polarisation)


    if (polarisation is True):
        # read smoothed planck maps.
        print('load planck 353GHz data')
        # read_smooth_maps(filename, name, shape)
        IQU_smaps = smooth.read_smooth_maps(planckfile, names[0], 3)
        dust_smap = smooth.read_smooth_maps(dustfile, names[1], 1)[0]
        T_smap = IQU_smaps[0]
        Q_smap = IQU_smaps[1]
        U_smap = IQU_smaps[2]
        
        Nside = hp.get_nside(T_smap)
        print('Using Nside={}'.format(Nside))

        #sys.exit()
        # load tomography data:
        data = load.load_tomographydata(tomofile, colnames)
        print('Data loaded, using Nside={}'.format(Nside))

        p_map, q_map, u_map, sigma, r_map, pix = load.tomo_map(data,\
                                                               Nside, part)
        u_map = -u_map # to Healpix convention
        mask = np.unique(pix)
        print(len(mask))
        u_smap = smooth.smooth_tomo_map(u_map, mask, Nside, res)
        q_smap = smooth.smooth_tomo_map(q_map, mask, Nside, res)
        p_smap = smooth.smooth_tomo_map(p_map, mask, Nside, res)
        print('Tomography maps smoothed')
        print(np.mean(q_smap[mask]), np.mean(dust_smap[mask]), np.mean(Q_smap[mask]))
        dPsi = np.full(len(u_map), hp.UNSEEN)
        #sys.exit()

        l, b = tools.convert2galactic(data[:,0], data[:,1])
        lon = np.mean(l)
        lat = np.mean(b)
        print(lon, lat)

        x = 0.5*np.arctan2(U_smap[mask], Q_smap[mask])
        #x[x<0.] += np.pi
        #x[x>=np.pi] -= np.pi

        x_v = 0.5*np.arctan2(u_smap[mask], q_smap[mask])
        #psi_v[psi_v<0] += np.pi
        #psi_v[psi_v>=np.pi] -= np.pi 
        print('Polarization angles of planck (mean, min, max) [deg]:')
        print(np.mean(x)*180/np.pi,np.min(x)*180/np.pi, np.max(x)*180/np.pi)
        print(np.mean(x_v)*180/np.pi,np.min(x_v)*180/np.pi,np.max(x_v)*180/np.pi)
        #print(np.mean(x+np.pi/2-psi_v))
        if (pol == 'P') or (pol == 'p'):
            print('-- P polarisation --')

            psi, psi_v, psi_s = tools.delta_psi(Q_smap[mask], q_smap[mask],\
                                                U_smap[mask],u_smap[mask])\
            #, plot=True, name='smooth2')

            dPsi[mask] = psi
            full_IQU = [T_smap, Q_smap, U_smap]
            tot_res, frac_res, dust = tools.map_analysis_function(p_smap, T_smap,\
                                                            dust_smap, mask, Nside)

            return(tot_res, frac_res, dust, [lon, lat], full_IQU, mask, r_map, dPsi)

        elif (pol == 'Q') or (pol == 'q'):
            print('-- Q polarisation --')
            psi, psi_v, psi_s = tools.delta_psi(Q_smap[mask], q_smap[mask], U_smap[mask],\
                                    u_smap[mask], plot=True)

            dPsi[mask] = psi
            full_IQU = [T_smap, Q_smap, U_smap]
            tot_res, frac_res, dust = tools.map_analysis_function(q_smap, Q_smap,\
                                                            dust_smap, mask, Nside)
            return(tot_res, frac_res, dust, [lon, lat], full_IQU, mask, r_map, dPsi)

        elif (pol == 'U') or (pol == 'u'):
            print('-- U polarisation --')
            print(len(u_smap))
            psi, psi_v, psi_s = tools.delta_psi(Q_smap[mask], q_smap[mask],\
                                                U_smap[mask],u_smap[mask], plot=True)

            dPsi[mask] = psi
            full_IQU = [T_smap, Q_smap, U_smap]
            tot_res, frac_res, dust = tools.map_analysis_function(u_smap, U_smap,\
                                                            dust_smap, mask, Nside)

            return(tot_res, frac_res, dust, [lon, lat], full_IQU, mask, r_map, dPsi)

        elif (pol == 'QU') or (pol == 'qu'):
            print('-- Q,U polarisation --')
            print('Return: tomo, planck, dust, mask, dpsi, fullIQU, [lon,lat], r')
            psi, psi_v, psi_s = tools.delta_psi(Q_smap[mask], q_smap[mask],\
                                                U_smap[mask], u_smap[mask])
            #, plot=True, name=Nside)

            dPsi[mask] = psi
            full_IQU = [T_smap, Q_smap, U_smap]
            tomo = [q_smap, u_smap, sigma[1], sigma[2]]
            planck = [Q_smap, U_smap]
            coord = [lon, lat]
            angles = [dPsi[mask], psi_v, psi_s, sigma[3]]
            return(tomo, planck, dust_smap, coord, full_IQU, mask, r_map, angles)


    else:
        # use unsmoothe maps
        print('Use non smoothed maps')
        # load planck
        print('load planck 353GHz data')

        #T, P, Q, U = load.load_planck_map(planckfile, p=True)
        data = load.load_planck_map(planckfile, p=True)
        d353 = load.load_planck_map(dustfile)
        sys.exit()
        dust353 = tools.Krj2Kcmb(d353) * 1e6
        T = T*1e6
        P = P*1e6
        Q = Q*1e6
        U = U*1e6
        Nside = hp.get_nside(T_smap)

        data = load.load_tomographydata(tomofile, colnames)
        p_map, q_map, u_map, sigma, r_map, pix = load.tomo_map(data, Nside)
        u_map = -u_map # to Healpix convention
        mask = np.unique(pix)

        l, b = tools.convert2galactic(data[:,0], data[:,1])
        lon = np.mean(l)
        lat = np.mean(b)

        dPsi = np.full(len(u_map), hp.UNSEEN)

        if Ppol == True:
            print('-- P polarisation --')
            psi = tools.delta_psi(Q[mask], q_map[mask], U[mask],\
                                    u_map[mask], plot=True)
            dPsi[mask] = psi
            full_IQU = [T, Q, U]
            tot_res, frac_res, dust = tools.map_analysis_function(p_map, T,\
                                                            dust353, mask)
            return(tot_res, frac_res, dust, [lon, lat], full_IQU, mask, r_map, dPsi)

        elif Qpol == True:
            print('-- Q polarisation --')
            psi = tools.delta_psi(Q[mask], q_map[mask], U[mask],\
                                    u_map[mask], plot=True)
            dPsi[mask] = psi
            full_IQU = [T, Q, U]
            tot_res, frac_res, dust = tools.map_analysis_function(q_map, Q,\
                                                            dust353, mask)
            return(tot_res, frac_res, dust, [lon, lat], full_IQU, mask, r_map, dPsi)

        if Upol == True:
            print('-- U polarisation --')
            psi = tools.delta_psi(Q[mask], q_map[mask], U[mask],\
                                    u_map[mask], plot=True)
            dPsi[mask] = psi
            full_IQU = [T, Q, U]
            tot_res, frac_res, dust = tools.map_analysis_function(u_map, U,\
                                                            dust353, mask)
            return(tot_res, frac_res, dust, [lon, lat], full_IQU, mask, r_map, dPsi)


########################
path = 'Data/'
tomofile = 'Data/total_tomography.csv'
#planckfile = 'Data/HFI_SkyMap_353-psb-field-IQU_2048_R3.00_full.fits'
#dustfile = 'Data/dust_353_commander_temp_n2048_7.5arc.fits'

colnames = ['ra', 'dec', 'p', 'p_er', 'evpa', 'evpa_er', 'q', 'q_er',\
            'u', 'u_er', 'dist', 'dist_low', 'dist_up', 'Rmag1']

########################
# Input arguments from command line
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument('-v', '--verbose', action='store_true')
group.add_argument('-q', '--quite', action='store_true')

parser.add_argument('planckfile', type=str,\
                    help='The filename of the planck IQU file to read, with path.')
parser.add_argument('dustfile', type=str,\
                    help='The planck dust intensity filename to read, with path.')
parser.add_argument('dtnames', type=str,\
                    choices=['IQU+dust', 'IQU_planck+I_dust'],\
                    help='The column names of the data files.')
parser.add_argument('pol', type=str,\
                    choices=['unsmooth', 'P', 'p', 'Q', 'q', 'U', 'u', 'qu'],\
                    help='Which Stokes parameter to evaluate')
parser.add_argument('Nside', type=int, choices=[256, 512, 2048],
                    help='The resolution of the maps')
parser.add_argument('res', type=str, choices=['7.5', '15'],
                    help='The resolution of the maps')
parser.add_argument('plot', type=str, nargs='?', default=None, const=True,\
                    choices=['map', 'plot', 'corr', 'temp'],\
                    help='If make plots of results or not.')
parser.add_argument('save', type=str, nargs='?', default='',\
                    const=True, help='The saving path for figures')
parser.add_argument('part', type=str, nargs='?', default='',\
                    const=True, choices=['all', '1cloud','2cloud'],\
                    help='The region of the sky to look at')

args = parser.parse_args()

planckfile = args.planckfile
dustfile = args.dustfile
dtnames = args.dtnames
pol = args.pol
Nside = args.Nside
res = args.res
plot = args.plot
save = args.save
part = args.part

print(planckfile)
print(dustfile)
print(dtnames)
print(pol)
print(Nside)
print(res)
print(plot)
print(save)
print(part)
print('----')

dtnames = dtnames.split('+')
print(dtnames)
#part = 'all'
#part = '1cloud'
#part = '2cloud'

##################################
#         Function calls         #
##################################

if pol == 'qu':

    tomo, planck, dust, coord, full_IQU, mask, dist, dPsi = main(planckfile,\
                                dustfile, tomofile, colnames, dtnames, pol,\
                                res, part=part) 

    q = tomo[0]
    u = tomo[1]
    sq = tomo[2]
    su = tomo[3]
    Q = planck[0]
    U = planck[1]
    #sys.exit()
    if plot == 'temp':
        qu = [q[mask], sq[mask], u[mask], su[mask]]
        delta_psi, psi_v, psi_s, err_psi = dPsi[:]
        print('Create template for Q and U')
        # the template stuff
        template.template(psi_v, err_psi, delta_psi, Q[mask], U[mask], qu, mask)
        pass

    if plot == 'map':
        print('Plotting for {} polarisation'.format(pol))
        if (pol == 'P') or (pol == 'p'):
            ind = 0
        elif (pol == 'Q') or (pol == 'q'):
            ind = 1
        elif (pol == 'U') or (pol == 'u'):
            ind = 2

        # Plotting:
        lon = coord[0]
        lat = coord[1]
        Ppl = np.sqrt(Q**2 + U**2)
        ptomo = np.sqrt(q**2 + u**2)
        Qmin = -166.
        Qmax = 23.
        Umin = 56.3
        Umax = 391.
        # Plot maps:
        # planck
        plotting.plot_gnom(Q, lon, lat, 'Q', Nside=Nside,\
                           unit=r'$\mu K_{{CMB}}$',\
                           project='Planck', save='1', range=[Qmin,Qmax])
        plotting.plot_gnom(Q, lon, lat, 'Q', mask=mask, Nside=Nside,\
                           unit=r'$\mu K_{{CMB}}$', project='Planck',\
                           save='masked1', range=[Qmin,Qmax])
        plotting.plot_gnom(Q/dust, lon, lat, 'q', Nside=Nside, unit=None,\
                           project='Planck', save='2')
        plotting.plot_gnom(Q/dust, lon, lat, 'q', mask=mask, Nside=Nside,\
                           unit=None, project='Planck', save='masked2')
        plotting.plot_gnom(U, lon, lat, 'U', Nside=Nside,\
                           unit=r'$\mu K_{{CMB}}$',\
                           project='Planck', save='1', range=[Umin,Umax])
        plotting.plot_gnom(U, lon, lat, 'U', mask=mask, Nside=Nside,\
                           unit=r'$\mu K_{{CMB}}$', project='Planck',\
                           save='masked1', range=[Umin,Umax])
        plotting.plot_gnom(U/dust, lon, lat, 'u', Nside=Nside, unit=None,\
                           project='Planck', save='2')
        plotting.plot_gnom(U/dust, lon, lat, 'u', mask=mask,  Nside=Nside,\
                           unit=None, project='Planck', save='masked2')

        # robopol
        plotting.plot_gnom(q, lon, lat, 'q', mask=mask, Nside=Nside,\
                           unit=None, project='RoboPol', save='2')
        plotting.plot_gnom(q*dust, lon, lat, 'Q', mask=mask, Nside=Nside,\
                           unit=r'$\mu K_{{CMB}}$', project='RoboPol',\
                           save='1')
        plotting.plot_gnom(u, lon, lat, 'u', mask=mask, Nside=Nside,\
                           unit=None, project='RoboPol', save='2')
        plotting.plot_gnom(u*dust, lon, lat, 'U', mask=mask, Nside=Nside,\
                           unit=r'$\mu K_{{CMB}}$', project='RoboPol',\
                           save='1')
        """
        # P pol
        plotting.plot_gnom(Ppl, lon, lat, 'P', mask=mask, Nside=Nside,\
                           unit=r'$K_{{CMB}}$', project='Planck', save='masked1')
        plotting.plot_gnom(Ppl, lon, lat, 'P', Nside=Nside, unit=r'$K_{{CMB}}$',\
                           project='Planck', save='1')
        plotting.plot_gnom(Ppl/dust, lon, lat, 'p', Nside=Nside,\
                           unit=None, project='Planck', save='2')
        plotting.plot_gnom(Ppl/dust, lon, lat, 'p', mask=mask, Nside=Nside,\
                           unit=None, project='Planck', save='masked2')

        plotting.plot_gnom(ptomo, lon, lat, 'p', mask=mask, Nside=Nside,\
                           unit=None, project='RoboPol', save='2')
        plotting.plot_gnom(ptomo*dust, lon, lat, 'P', mask=mask, Nside=Nside,\
                           unit=r'$K_{{CMB}}$', project='RoboPol', save='1')
        """
        plt.show()

    elif plot == 'corr':
        print('Plotting for {} polarisation'.format(pol))
        if (pol == 'P') or (pol == 'p'):
            ind = 0
        elif (pol == 'Q') or (pol == 'q'):
            ind = 1
        elif (pol == 'U') or (pol == 'u'):
            ind = 2

        # plot correlation:
        print('')
        print('Plot joint correlation')
        plotting.plot_corr2(Q, U, q, u, sq, su, mask, dist, \
                            Nside=Nside, xlab=r'$q,u$', \
                            ylab=r'$Q,U_{{353}}$', title='QU-qu',\
                            save=save, part=part)

        # plot tomography and planck stars in same plot:
        print('')
        #print('Plot comparing with Planck XII 2018 reproduction')
        #plotting.Tomo2Star_pl(Q, U, q, u, sq, su, dist, mask, planckfile,\
        #                      Nside=Nside,\
        #                      xlab=r'$q_v, u_v$', ylab=r'$Q_s,U_s$',\
        #                      save=save)

        #plotting.plot_corr2norm(Q,U, q, u, sq, su, 'QU-qu', mask, dist, dust, tau,\
        #    Nside=Nside, xlab=r'$(q,u)/\tau$', ylab=r'$(Q, U)/I_{{353}}$',\
        #    title='QU-qu', save=save)

    #

else:
    Totres, fracres, dust, coord, full_IQU, mask, dist, dPsi = main(planckfile,\
                                dustfile, tomofile, colnames, dtnames, pol, res)
    # data list comes as [tomo, planck, diff, corr]
    if plot == 'plot':
        print('Plotting for {} polarisation'.format(pol))
        if (pol == 'P') or (pol == 'p'):
            ind = 0
        elif (pol == 'Q') or (pol == 'q'):
            ind = 1
        elif (pol == 'U') or (pol == 'u'):
            ind = 2

        # Plotting:
        lon = coord[0]
        lat = coord[1]
        # Correlation plots:
        print('{}corr_{}_{}arcmin'.format(save, pol, res))
        """
        plotting.plot_corr(Totres[0], Totres[1], 'corr_{}'.format(pol), mask,\
                    dist, xlab=r'Tomography ${}_{{frac}}\times I_d$'.format(pol),\
                    ylab=r'353GHz ${}$'.format(pol), title='{}'.format(pol),\
                    save='{}corr_{}_{}arcmin'.format(save, pol, res), Nside=Nside)
        plotting.plot_corr(fracres[0], fracres[1], 'corr_{}_frac'.format(pol),\
                    mask, dist, xlab=r'Tomography ${}_{{frac}}$'.format(pol),\
                    ylab=r'353 ${}/I_d$'.format(pol), Nside=Nside,\
                    title='{}_frac'.format(pol),\
                    save='{}corr_{}_frac_{}arcmin'.format(save, pol, res))
        # not scaled:
        plotting.plot_corr(fracres[0], Totres[1], 'corr_{}_unscaled'.format(pol),\
                    mask, dist, xlab=r'Tomography ${}_{{frac}}$'.format(pol),\
                    ylab=r'353 ${}$'.format(pol), title='{}_unscaled'.format(pol),\
                    save='{}corr_{}_{}ercmin_unscaled'.format(save, pol, res),\
                    Nside=Nside)

        #"""
        # Map plots in gnomview:
        plotting.plot_gnom(Totres[0], lon, lat,\
                            '{}_{}_{}arcmin'.format(save,pol, res), mask,\
                            unit=r'$K_{{cmb}}$', project='tomo', Nside=Nside)
        plotting.plot_gnom(fracres[0], lon, lat,\
                            '{}_{}_frac_{}arcmin'.format(save, pol, res), mask,\
                            unit=r'$K_{{cmb}}$', project='tomo', Nside=Nside)
        plotting.plot_gnom(Totres[1], lon, lat,\
                            '{}_{}_{}arcmin'.format(save, pol, res),mask,\
                            unit=r'$K_{{cmb}}$', project='planck', Nside=Nside)
        plotting.plot_gnom(fracres[1], lon, lat,\
                            '{}_{}_frac_{}arcmin'.format(save, pol, res), mask,\
                            unit=r'$K_{{cmb}}$', project='planck', Nside=Nside)

        #plt.show()
        # Plot delta psi:

        #plotting.plot_DeltaPsi(dPsi, mask, Nside=Nside)

        #sys.exit()
        # plot Rations:
        Ebv_file = raw_input('Type reddening file name: ')
        Ebv = 'Data/' + Ebv_file
        print(Ebv)
        plotting.plot_ratios(fracres[0], Totres[1], dust, Ebv, mask, Nside=Nside)

sys.exit()
########################

########################
