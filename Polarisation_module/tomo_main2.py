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
from scipy import stats

import convert_units as cu
import tools_mod as tools
import smoothing_mod as smooth
import plotting_mod as plotting
import load_data_mod as load
import template_mod as template
import pol_sampler_mod as sampler

####################################


def main(planckfile, dustfile, tomofile, colnames, names, pol, res,\
         part='all', distcut=None):
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

    if distcut is None:
        distcut = 900

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
        print(planckfile)
        band = planckfile.split('_')[2]
        if len(band) > 3:
            band = band[:3]
            if band == '15a':
                band = '353'
        print(band)

        if int(band) < 353:
            # load cmb intensity and subtract form polarization maps
            cmbfile = 'Data/IQU_Nside{}_CMB_10arcmin.h5'.format(Nside)
            cmbmaps = tools.Read_H5(cmbfile, 'IQU')*1e6
            Q_cmb = cmbmaps[1,:]
            U_cmb = cmbmaps[1,:]
            Q_smap = Q_smap - Q_cmb
            U_smap = U_smap - U_cmb
            
        print(np.mean(Q_smap), np.mean(U_smap))
        #sys.exit()
        # load tomography data:
        data = load.load_tomographydata(tomofile, colnames)
        print('Data loaded, using Nside={}'.format(Nside))

        p_map, q_map, u_map, sigma, r_map, pix =\
                    load.tomo_map(data, Nside, part=part, distcut=distcut)
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
        theta, phi = hp.pix2ang(Nside, pix)        
        lon = np.mean(phi)*180/np.pi
        lat = 90 - np.mean(theta)*180/np.pi
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
            tomo = [q_smap, u_smap, p_smap, sigma[1], sigma[2], sigma[0]]
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
#tomofile = 'Data/total_tomography.csv'
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
parser.add_argument('tomofile', type=str,\
                    choices=['tomo1','tomo2', 'tomo_new'],\
                    help='Which tomography file to read, -1 for old data, -2 for total data and -new for only new data.')
parser.add_argument('dtnames', type=str,\
                    choices=['IQU+dust', 'IQU_planck+I_dust', 'IQU'],\
                    help='The column names of the data files.')
parser.add_argument('pol', type=str,\
                    choices=['unsmooth', 'P', 'p', 'Q', 'q', 'U', 'u', 'qu'],\
                    help='Which Stokes parameter to evaluate')
parser.add_argument('Nside', type=int, choices=[256, 512, 2048],
                    help='The resolution of the maps')
parser.add_argument('res', type=str, choices=['7', '10', '15'],
                    help='The resolution of the maps')
parser.add_argument('plot', type=str, nargs='?', default=None, const=True,\
                    choices=['map', 'plot', 'corr', 'temp', 'test1',\
                             'test2', 'test3', 'mcmc'],\
                    help='If make plots of results or not.')
parser.add_argument('save', type=str, nargs='?', default='',\
                    const=True, help='The saving path for figures')
parser.add_argument('part', type=str, nargs='?', default='',\
                    const=True, choices=['all', 'LVC','IVC', 'none'],\
                    help='The region of the sky to look at')
parser.add_argument('--distcut', nargs='+', type=int,\
                    help='The distance cut to IVC')

args = parser.parse_args()

planckfile = args.planckfile
tomofile = args.tomofile
dtnames = args.dtnames
pol = args.pol
Nside = args.Nside
res = args.res
plot = args.plot
save = args.save
part = args.part
distcut = args.distcut
dustfile = 'Data/dust_Nside{}_15arcmin.h5'.format(Nside)#args.dustfile

if tomofile == 'tomo1':
    tomofile = 'Data/total_tomography.csv'
elif tomofile == 'tomo_new':
    tomofile = 'Data/total_tomography_new.csv'
else:
    tomofile = 'Data/total_tomography_2.csv'

print(planckfile)
print(tomofile)
print(dustfile)
print(dtnames)
print(pol)
print(Nside)
print(res)
print(plot)
print(save)
print(part)
print(distcut)
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
                                res, part=part, distcut=distcut) 

    q = tomo[0]
    u = tomo[1]
    p = tomo[2]
    sq = tomo[3]
    su = tomo[4]
    Q = planck[0]
    U = planck[1]
    T = full_IQU[0]
    #print(np.abs(q[mask]/sq[mask]))
    #print(np.abs(u[mask]/su[mask]))
    a = planckfile.split('_')[2]
    b = a.split('3')[-1]
    print(b)
    if (b == 'npipe'):
        Q = Q*1e-6
        U = U*1e-6
    #

    #sys.exit()
    if plot == 'temp':
        qu = [q[mask], sq[mask], u[mask], su[mask]]
        delta_psi, psi_v, psi_s, err_psi = dPsi[:]
        print('Create template for Q and U')
        # the template stuff
        #template.plot_map(U, mask, 'U353', Nside)
        template.template(psi_v, err_psi, delta_psi, Q[mask], U[mask], qu,\
                          mask, Nside, res)
        pass

    if plot == 'mcmc':
        print('Sampling contribution to submm polarization')
        # Load C_ij from Planck:    
        Cfile = 'Data/Planck_Cij_353_2048_full.h5'
        C_ij = load.C_ij(Cfile, Nside)
        C_II = C_ij[0,:] * 1e12
        C_IQ = C_ij[1,:] * 1e12
        C_IU = C_ij[2,:]* 1e12
        C_QQ = C_ij[3,:]* 1e12
        C_QU = C_ij[4,:]* 1e12
        C_UU = C_ij[5,:]* 1e12
              
        # input params to sampler: Ps, ps, psi_v, mask
        Ps = np.sqrt(Q**2 + U**2)
        err_P = np.sqrt(C_QQ*Q**2 + C_UU*U**2)/Ps
        Ps = tools.MAS(Ps, err_P)
        ps = Ps/T
        lon = coord[0]
        lat = coord[1]
        ps_err = err_P/T

        # Convert from uK_cmb to MJy/sr: 
        unit = 287.45*1e-6
        Ps *= unit
        
        QU = np.array([Q, U])*unit
        qu = np.array([q, u])
        print(QU[:,mask], np.shape(QU))

        par, std, chi2 = tools.Chi2(Q[mask], U[mask], q[mask], u[mask],\
                                    C_ij[:,mask], sq[mask], su[mask])
        par_q, st_qd, chi2_q = tools.Chi2(Q[mask], None, q[mask], None,\
                                          C_ij[:,mask], sq[mask], None)
        par_u, std_u, chi2_u = tools.Chi2(None, U[mask], None, u[mask],\
                                          C_ij[:,mask], None, su[mask])
        
        R_Pp = Ps[mask]/p[mask]
        err_R = err_P[mask]/p[mask] - Ps[mask]*tomo[-1][mask]/p[mask]**2
        print(R_Pp)
        print(err_R*unit)

        mean_R = np.mean(R_Pp[R_Pp < 5.2])
        R_err = np.std(R_Pp)
        print(mean_R, R_err)
        Q0 = 0.18*Q
        U0 = 0.18*U
        params_mean = np.array([5, 0, 0])
        params_err = np.array([0.5, 0.005, 0.005]) 
        data_mean = np.array([mean_R, -0.005, 0.025]) 

        models, params, sigmas = sampler.QU_sampler(QU, qu, params_mean,\
                                                    params_err, mask,\
                                                    data_mean, Nside=Nside,\
                                                    Niter=2000, R_Pp=R_Pp)
        maxL_params, samples = params[:]
        QU_model, QU_star, QU_bkgr = models[:]
        QU_error, params_error = sigmas[:]
        
        Q_star_err = sampler.star_error(maxL_params[0,mask],\
                                        params_error[0,mask],\
                                        q[mask], sq[mask])
        U_star_err = sampler.star_error(maxL_params[1,mask],\
                                        params_error[0,mask],\
                                        u[mask], su[mask])
        Q_bkgr_err = params_error[1,mask]
        U_bkgr_err = params_error[2,mask]
        
        # Plot correlation plots of models vs visual
        print('--> model')
        sampler.correlation(qu[:,mask], QU_model[:,mask], sq[mask], su[mask],\
                            QU_error[:,mask], mask, lab='model', save=save,\
                            R_Pp=[mean_R, np.mean(maxL_params[0, mask])],\
                            R_err=[R_err, np.mean(params_error[0,mask])])
        print('--> star')
        sampler.correlation(qu[:,mask], QU_star[:,mask], sq[mask], su[mask],\
                            np.array([Q_star_err, U_star_err]), mask, \
                            lab='star', save=save,\
                            R_Pp=[mean_R, np.mean(maxL_params[0, mask])],\
                            R_err=[R_err, np.mean(params_error[0,mask])])
        print('--> bkgr')
        sampler.correlation(qu[:,mask], QU_bkgr[:,mask], sq[mask], su[mask],\
                            np.array([Q_bkgr_err, U_bkgr_err]), mask,\
                            lab='bkgr', save=save)
        
        print('--> Data')
        plotting.plot_corr2(Q, U, q, u, sq, su, mask, dist, \
                            Nside=Nside, xlab=r'$q,u$', \
                            ylab=r'$Q,U_{{353}}$', title='QU-qu',\
                            save=save, part=part)

        """
        sampler.plot_model_vs_data(q[mask], u[mask], Q[mask]*unit,\
                                   U[mask]*unit, c=['k', 'b'], m='^')
        sampler.plot_model_vs_data(q[mask], u[mask], QU_model[0,mask],\
                                   QU_model[1,mask],\
                                   c=['dimgray', 'cornflowerblue'],\
                                   m='*')
        sampler.plot_model_vs_data(q[mask], u[mask], QU_star[0,mask],\
                                   QU_star[1,mask],\
                                   c=['silver', 'skyblue'], m='+')
        sampler.plot_model_vs_data(q[mask], u[mask], QU_bkgr[0,mask],\
                                   QU_bkgr[1,mask], c=['gray', 'c'], m='x',\
                                   lab=['data','model','star','bkgr'])
        """
        plt.show()
        sys.exit()

        Rmin = 3.8
        Rmax = 5.2
        Rcut = np.logical_and(R_Pp < Rmax, R_Pp > Rmin)
        print(len(mask))
        #mask = mask[Rcut]
        #print(len(mask))
        delta_psi, psi_v, psi_s, err_psi = dPsi[:]
        #plotting.plot_Rpp_psi(R_Pp, psi_v, psi_s, delta_psi, save=save)
        plotting.plot_corr_Rpp(Q[mask], U[mask], q[mask], u[mask], R_Pp,\
                               C_ij[:,mask]*1e12, sq[mask], su[mask],\
                               title='correlation', save=save)
        
        """
        plt.figure()
        plt.scatter(u[mask], U[mask]*unit, marker='*', c=R_Pp, cmap='jet',\
                    label=r'Uu', vmin=Rmin, vmax=Rmax)
        plt.scatter(q[mask], Q[mask]*unit, marker='^', c=R_Pp, cmap='jet',\
                    label='Qq', vmin=Rmin, vmax=Rmax)
        cbar = plt.colorbar()
        cbar.set_label(r'$R_{{P/p}}$ [MJy/sr]')
        plt.xlabel(r'$q_v, u_v$')
        plt.ylabel(r'$Q_s, U_s$ [MJy/sr]')
        plt.legend()
        plt.savefig('Figures/pol_Rpp_{}.png'.format(save))
        plt.figure()
        plt.scatter(u[mask], R_Pp, marker='*', c=U[mask]*unit, cmap='jet')
        cbar = plt.colorbar()
        cbar.set_label(r'$U_s$ [MJy/sr]')
        plt.xlabel(r'$u_v$')
        plt.ylabel(r'$R_{{Pp}}$ [MJy/sr]')
        plt.figure()
        plt.scatter(q[mask], R_Pp, marker='^', c=Q[mask]*unit, cmap='jet')
        cbar = plt.colorbar()
        cbar.set_label(r'$Q_s$ [MJy/sr]')
        plt.xlabel(r'$q_v$')
        plt.ylabel(r'$R_{{Pp}}$ [MJy/sr]')
        """
        R_map = np.full(hp.nside2npix(Nside), hp.UNSEEN)
        R_map[mask] = R_Pp
        
        hp.gnomview(R_map, title=r'$R_{{P/p}}$', cmap='jet', min=Rmin,\
                    max=Rmax, rot=[lon,lat], unit='MJy/sr', xsize=100)
        hp.graticule()
        plt.savefig('Figures/R_pp_map_{}.png'.format(save))
        hp.gnomview(Q*unit/u, title=r'$Q_s/q_v$', cmap='jet',\
                    rot=[lon,lat], unit='MJy/sr', xsize=100)
        hp.graticule()
        plt.savefig('Figures/Qq_map_{}.png'.format(save))
        hp.gnomview(U*unit/u, title=r'$U_s/u_v$', cmap='jet',\
                    rot=[lon,lat], unit='MJy/sr', xsize=100)
        hp.graticule()
        plt.savefig('Figures/Uu_map_{}.png'.format(save))
        hp.gnomview(T*unit, title=r'$I_{{353}}$', cmap='jet',\
                    rot=[lon,lat], unit='MJy/sr', xsize=100)
        hp.graticule()
        plt.savefig('Figures/I353_map_{}.png'.format(save))
        #"""
        plt.show()

    if plot[:4] == 'test':
        """
        Estimate polarisation contribution from behind the stars. 
        Assume: p_star = factor*p_v, uniform b-field for 1cloud region. 
        """
        # Load C_ij from Planck:    
        Cfile = 'Data/Planck_Cij_353_2048_full.h5'
        C_ij = load.C_ij(Cfile, Nside)
        C_II = C_ij[0,:]
        C_IQ = C_ij[1,:]
        C_IU = C_ij[2,:]
        C_QQ = C_ij[3,:]
        C_QU = C_ij[4,:]
        C_UU = C_ij[5,:]
        #mask = mask[np.arange(len(mask))!=5]
        psi_s = dPsi[2]
        #psi_s = psi_s[np.arange(len(psi_s))!=5]
        psi_v = dPsi[1]
        #psi_v = psi_v[np.arange(len(psi_v))!=5]
        Ps = np.sqrt(Q[mask]**2 + U[mask]**2)
        err_P = np.sqrt(C_QQ[mask]*Q[mask]**2 + C_UU[mask]*U[mask]**2)/Ps
        Ps = tools.MAS(Ps, err_P)
        print(np.mean(Ps/T[mask]), np.mean(Ps))
        pv = tomo[2]; p_err = tomo[-1]


        p_dust = tools.get_p_dust(Ps, T[mask])
    
        # Test 1:
        if plot == 'test1':
            p_star = tools.get_p_star(pv[mask], p_dust) 
            p_bkgr, Pmask_bkgr =\
                            tools.get_pol_bkgr(p_dust,p_star,np.mean(psi_s))
            q_bkgr, Qmask_bkgr =\
                            tools.get_pol_bkgr(p_dust,p_star,np.mean(psi_s)) 
            u_bkgr, Umask_bkgr =\
                            tools.get_pol_bkgr(p_dust,p_star,np.mean(psi_s))
            P_bkgr = p_bkgr*T[mask[Pmask_bkgr]]#Ps[mask_bkgr]
            P_star = p_star*T[mask]#Ps
            Q_bkgr = q_bkgr*T[mask[Qmask_bkgr]]#Ps[mask_bkgr]
            Q_star = p_star*np.cos(2*psi_s)*T[mask]#Ps
            U_bkgr = u_bkgr*T[mask[Umask_bkgr]]#Ps[mask_bkgr]
            U_star = p_star*np.sin(2*psi_s)*T[mask]#Ps
            
            print(np.mean(P_bkgr), np.mean(P_star), np.mean(Ps))
            print('P_bkgr/P=', np.mean(P_bkgr/Ps[Pmask_bkgr]), np.mean(P_star/Ps))
            print('Q_bkgr/Q=', np.mean(Q_bkgr/Q[mask[Qmask_bkgr]]), np.mean(Q_star/Q[mask]))
            print('U_bkgr/U=', np.mean(U_bkgr/U[mask[Umask_bkgr]]), np.mean(U_star/U[mask]))

        elif plot == 'test2':
            # Test 2, use as default
            Q_star = tools.get_pol_star(q[mask], Q[mask])
            Q_bkgr, Qmask_bkgr = tools.get_pol_bkgr(Q[mask], Q_star)
            U_star = tools.get_pol_star(u[mask], U[mask])
            U_bkgr, Umask_bkgr = tools.get_pol_bkgr(U[mask], U_star)
            P_star = tools.get_pol_star(pv[mask], Ps)
            P_bkgr, Pmask_bkgr = tools.get_pol_bkgr(Ps, P_star)
            
            print(np.mean(P_bkgr), np.mean(P_star), np.mean(Ps))
            print('P_bkgr/P=', np.mean(P_bkgr/Ps), np.mean(P_star/Ps))
            print('Q_bkgr/Q=', np.mean(Q_bkgr/Q[mask]),np.mean(Q_star/Q[mask]))
            print('U_bkgr/U=', np.mean(U_bkgr/U[mask]),np.mean(U_star/U[mask]))
            
            # test chi^2 and compare:
            params, std, chi2 = tools.Chi2(Q, U, q, u, C_ij, sq, su)
            Q_star2 = params[0]*q[mask]
            U_star2 = params[0]*u[mask]
            Q_bkgr2 = params[1]
            U_bkgr2 = params[1]
            print('Compare chi^2 test to test 2')
            print('Star:', np.mean(Q_star), np.mean(Q_star2))
            print((Q_star-Q_star2)/Q_star)
            print('     ', np.mean(U_star), np.mean(U_star2))
            print((U_star-U_star2)/U_star)
            #print('Full:', (Q_star+Q_bkgr)/Q[mask], (U_star+U_bkgr)/U[mask])
            print('Full:', np.mean((Q_star2+Q_bkgr2)/Q[mask]), np.mean((U_star2+U_bkgr2)/U[mask]))
            print('-',(U_star2+U_bkgr2)/U[mask], (U_star2+U_bkgr2)/U[mask])
            


        elif plot == 'test3':
            # Check R_Pp*[q,u], R as array and scalar. Find residuals:
            unit = 287.45*1e-6
            
            R_Pp = Ps/pv[mask]*unit
            R_mean = np.mean(R_Pp)
            sR = np.std(R_Pp)
            print(R_Pp, R_mean, sR)

            Q_mod = -R_Pp*q[mask]
            U_mod = -R_Pp*u[mask]
            Q_mod2 = -R_mean*q[mask]
            U_mod2 = -R_mean*u[mask]
            
            x = 0.5*np.arctan2(U_mod, Q_mod)
            print(x*180/np.pi, np.mean(x)*180/np.pi, np.std(x)*180/np.pi)
            # residual for array analysis:
            Q_res = Q[mask]*unit - Q_mod
            U_res = U[mask]*unit - U_mod
            #print(Q_res)
            #print(U_res)
            x2 = 0.5*np.arctan2(U_res, Q_res)
            print(x2*180/np.pi, np.mean(x2)*180/np.pi, np.std(x2)*180/np.pi)
            
            dx, x_v, x_s = tools.delta_psi(Q_mod, q[mask],\
                                                U_mod, u[mask])
            dx2, x2_v, x2_s = tools.delta_psi(Q_res, q[mask],\
                                                U_res, u[mask])
            R_star = np.sqrt(Q_mod**2 + U_mod**2)/pv[mask]
            R_res = np.sqrt(Q_res**2 + U_res**2)/pv[mask]
            print(np.mean(R_star), np.std(R_star))
            print(np.mean(R_res), np.std(R_res))
            print(R_star/R_Pp, np.mean(R_star/R_Pp))
            print(R_res/R_Pp, np.mean(R_res/R_Pp))
            print(psi_v)
            print(psi_s)
            print('Residuals in polarisation angle')
            psi_res_star = (psi_s - x)/psi_s
            psi_res_bkgr = (psi_s - x2)/psi_s
            print(psi_res_star)
            print(psi_res_bkgr)
            
            
            plotting.plot_corr_Rpp(Q[mask], U[mask], q[mask], u[mask],\
                                   R_Pp, C_ij[:,mask]*1e12,\
                                   sq[mask], su[mask], title='correlation',\
                                   save=save, part='full_dpsi', Rmin=-10, Rmax=10)
            plotting.plot_corr_Rpp(Q_mod/unit, U_mod/unit, q[mask], u[mask],\
                                   R_star, C_ij[:,mask]*1e12, sq[mask],\
                                   su[mask], title='correlation', save=save,\
                                   part='star', Rmin=3.8, Rmax=5.2)
            plotting.plot_corr_Rpp(Q_res/unit, U_res/unit, q[mask], u[mask],\
                                   R_res, C_ij[:,mask]*1e12, sq[mask],\
                                   su[mask], title='correlation', save=save,\
                                   part='res', Rmin=0, Rmax=1.4)
            """
            plt.figure()
            plt.scatter(u[mask], U[mask]*unit, c=psi_s*180/np.pi, cmap='jet')
            plt.scatter(q[mask], Q[mask]*unit, c=psi_s*180/np.pi, cmap='jet')
            plt.colorbar()
            plt.figure()
            plt.scatter(u[mask], U[mask]*unit, c=psi_v*180/np.pi, cmap='jet')
            plt.scatter(q[mask], Q[mask]*unit, c=psi_v*180/np.pi, cmap='jet')
            plt.colorbar()
            
            plt.figure()
            plt.scatter(psi_v, psi_s, c=dPsi[0]*180/np.pi, cmap='jet')
            #plt.scatter(x_v, x_s, c=dPsi[0]*180/np.pi, cmap='brg', marker='.')
            plt.colorbar()
            """
            plt.figure()
            plt.scatter(psi_s*180/np.pi, x*180/np.pi, c=psi_res_star, cmap='jet')
            plt.colorbar()
            plt.figure()
            plt.scatter(psi_s*180/np.pi, x2*180/np.pi, c=psi_res_bkgr, cmap='brg')
            plt.colorbar()
        
            plt.show()
            
            sys.exit()
        #
        unit = 287.45*1e-6
        R_Pp_star = P_star*unit/pv[mask]
        R_Pp = Ps*unit/pv[mask]
        R_Pp_bkgr = P_bkgr*unit/pv[mask]
        print(np.mean(R_Pp), np.std(R_Pp))
        print(np.mean(R_Pp_star), np.std(R_Pp_star))
        print(np.mean(R_Pp_bkgr), np.std(R_Pp_bkgr))
        
        # plotting:
        plotting.plot_corr_Rpp(Q[mask], U[mask], q[mask], u[mask], R_Pp,\
                               C_ij[:,mask]*1e12, sq[mask], su[mask],\
                               title='correlation', save=save, part='full',\
                               Rmin=3.8, Rmax=5.2)
        plotting.plot_corr_Rpp(Q_star, U_star, q[mask], u[mask], R_Pp_star,\
                               C_ij[:,mask]*1e12, sq[mask], su[mask],\
                               title='correlation', save=save, part='Star',\
                               Rmin=4.0, Rmax=5.0)
        
        plotting.plot_corr_Rpp(Q_bkgr, U_bkgr,q[mask], u[mask], R_Pp_bkgr,\
                               C_ij[:,mask]*1e12, sq[mask], su[mask],\
                               title='correlation', save=save, part='bkgr',\
                               Rmin=0, Rmax=1)
        

        #print((p_star + p_bkgr) - p_dust)
        #plt.plot(u[mask], U[mask]*278.45*1e-6, 'b.')
        #plt.plot(u[mask], U_bkgr*278.45*1e-6,'xg')
        #plt.plot(u[mask], U_star*278.45*1e-6, 'm^')
        #plt.plot(u[mask],(U_bkgr+U_star)*278.45e-6, '+k')
        #plt.ylabel(r'$U_s$ [MJy/sr]')
        #plt.xlabel(r'$u_v$')
        
        """
        print('-- full plot --')        
        plotting.plot_bkgr(Q[mask], U[mask], q[mask], u[mask],\
                           C_ij[:,mask], sq[mask], su[mask], mask,\
                           title='Full', save=save)

        #plotting.plot_bkgr(Q_bkgr+Q_star, U_bkgr+U_star, q[mask], u[mask],\
        #                   C_ij[:,mask], sq[mask], su[mask], mask,\
        #                   title='Full', save=save)
        
        print('background')
        plotting.plot_bkgr(Q_bkgr, U_bkgr, q[mask], u[mask],\
                           C_ij[:,mask], sq[mask], su[mask], mask,\
                           title='Background', save=save)
        print('star')
        plotting.plot_bkgr(Q_star, U_star, q[mask], u[mask],\
                           C_ij[:,mask], sq[mask], su[mask], mask,\
                           title='Star', save=save)
        
        """
        #plt.show()
        sys.exit()

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
        print(coord)
        Ppl = np.sqrt(Q**2 + U**2)
        ptomo = np.sqrt(q**2 + u**2)
        Qmin = -166.
        Qmax = 23.
        Umin = 56.3
        Umax = 391.
        # Plot maps:
        # planck
        #plotting.plot_gnom(Q, lon, lat, 'Q', Nside=Nside,\
        #                   unit=r'$\mu K_{{CMB}}$',\
        #                   project='Planck', save='1', range=[Qmin,Qmax])
        plotting.plot_gnom(Q, lon, lat, 'Q', mask=mask, Nside=Nside,\
                           unit=r'$\mu K_{{CMB}}$', project='Planck',\
                           save=save, range=[-100,300], xsize=100)
        #plotting.plot_gnom(Q/dust, lon, lat, 'q', Nside=Nside, unit=None,\
        #                   project='Planck', save='2')
        #plotting.plot_gnom(Q/dust, lon, lat, 'q', mask=mask, Nside=Nside,\
        #                   unit=None, project='Planck', save='masked2')
        #plotting.plot_gnom(U, lon, lat, 'U', Nside=Nside,\
        #                   unit=r'$\mu K_{{CMB}}$',\
        #                   project='Planck', save='1', range=[Umin,Umax])
        plotting.plot_gnom(U, lon, lat, 'U', mask=mask, Nside=Nside,\
                           unit=r'$\mu K_{{CMB}}$', project='Planck',\
                           save=save, range=[-100,300], xsize=100)
        #plotting.plot_gnom(U/dust, lon, lat, 'u', Nside=Nside, unit=None,\
        #                   project='Planck', save='2')
        #plotting.plot_gnom(U/dust, lon, lat, 'u', mask=mask,  Nside=Nside,\
        #                   unit=None, project='Planck', save='masked2')

        # robopol
        """
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
        #plotting.plot_Pol_NH(Q, U, q, u, sq, su, mask, save=save)
        #plotting.plot_data_vs_template(Q, U, q, u, sq, su, dist, dust, mask,\
        #                               Nside=Nside, save=save)
        #plotting.plot_UfromQ(Q, U, q, u, sq, su, mask, T)
        
        plotting.plot_corr2(Q, U, q, u, sq, su, mask, dist, \
                            Nside=Nside, xlab=r'$q,u$', \
                            ylab=r'$Q,U_{{353}}$', title='QU-qu',\
                            save=save, part=part)
        #plotting.plot_corr2(Q, Q, q, q, sq, sq, mask, dist, \
        #                    Nside=Nside, xlab=r'$q,q$', \
        #                    ylab=r'$Q,Q_{{353}}$', title='Q-q',\
        #                    save=save, part=part)
        # q/u (q): 0.9863, q/u (u): 0.9934
        
        # plot tomography and planck stars in same plot:
        print('')
        print('Plot comparing with Planck XII 2018 reproduction')
        #plotting.Tomo2Star_pl(Q, U, q, u, sq, su, dist, mask, planckfile,\
        #                      Nside=Nside, xlab=r'$q_v, u_v$',\
        #                      ylab=r'$Q_s,U_s$', save=save)

        #plotting.plot_corr2norm(Q,U, q, u, sq, su, 'QU-qu', mask, dist, \
        #                        dust, tau, Nside=Nside, xlab=r'$(q,u)/\tau$',\
        #                        ylab=r'$(Q, U)/I_{{353}}$', title='QU-qu',\
        #                        save=save)

    #
    plt.show()
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
