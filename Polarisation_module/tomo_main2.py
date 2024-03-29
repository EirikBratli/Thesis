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
#import pol_sampler_mod as sampler
from sampler2_mod import sampler as sampler2
from sampler3_mod import sampler as sampler3
from sampler2_mod import Error_estimation as Err_est2
from sampler3_mod import Error_estimation as Err_est3
from sampler4_mod import sampler as gibbs_sampler
import sampling_mod as sm
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
        # read smoothed planck maps. Units = uKcmb
        print('load planck 353GHz data')
        # read_smooth_maps(filename, name, shape)
        if len(names) == 3:
            IQU_smaps = smooth.read_smooth_maps(planckfile, names, 3)
        else:
            IQU_smaps = smooth.read_smooth_maps(planckfile, names[0], 3)
        #dust_smap = smooth.read_smooth_maps(dustfile, names[1], 1)[0]
        T_smap = IQU_smaps[0] 
        Q_smap = IQU_smaps[1]
        U_smap = IQU_smaps[2]
        Nside = hp.get_nside(T_smap)
        
        band = planckfile.split('_')[2]
        if len(band) > 3:
            band = band[:3]
            if band == '15a':
                band = '353'

        if int(band) < 353:
            # load cmb intensity and subtract form polarization maps
            cmbfile = 'Data/IQU_Nside{}_CMB_10arcmin.h5'.format(Nside)
            cmbmaps = tools.Read_H5(cmbfile, 'IQU')*1e6
            Q_cmb = cmbmaps[1,:]
            U_cmb = cmbmaps[1,:]
            Q_smap = Q_smap - Q_cmb
            U_smap = U_smap - U_cmb
        
        # load tomography data:
        data = load.load_tomographydata(tomofile, colnames)
        print('Data loaded, using Nside={}'.format(Nside))

        p_map, q_map, u_map, sigma, r_map, pix =\
                    load.tomo_map(data, Nside, part=part, distcut=distcut)
        u_map = -u_map # to Healpix convention
        mask = np.unique(pix)
        
        u_smap = smooth.smooth_tomo_map(u_map, mask, Nside, res)
        q_smap = smooth.smooth_tomo_map(q_map, mask, Nside, res)
        p_smap = smooth.smooth_tomo_map(p_map, mask, Nside, res)
        print('Tomography maps smoothed')
        print(np.mean(q_smap[mask]), np.mean(Q_smap[mask]))
        dPsi = np.full(len(u_map), hp.UNSEEN)
        #sys.exit()

        l, b = tools.convert2galactic(data[:,0], data[:,1])
        theta, phi = hp.pix2ang(Nside, pix)        
        lon = np.mean(phi)*180/np.pi
        lat = 90 - np.mean(theta)*180/np.pi
        print(lon, lat)

        x = 0.5*np.arctan2(U_smap[mask], Q_smap[mask])
        x_v = 0.5*np.arctan2(u_smap[mask], q_smap[mask])

        print('-- Q,U polarisation --')
        print('Return: tomo, planck, dust, mask, dpsi, fullIQU, [lon,lat], r')
        psi, psi_v, psi_s = tools.delta_psi(Q_smap[mask], q_smap[mask],\
                                            U_smap[mask], u_smap[mask])
        #, plot=True, name=Nside
        dPsi[mask] = psi # deg
        full_IQU = [T_smap, Q_smap, U_smap] # uKcmb
        tomo = [q_smap, u_smap, p_smap, sigma[1], sigma[2], sigma[0]] # .
        planck = [Q_smap, U_smap] # uKcmb
        coord = [lon, lat, l, b] # [deg,rad]
        angles = [dPsi[mask], psi_v, psi_s, sigma[3]] # deg
        return(tomo, planck, coord, full_IQU, mask, r_map, angles)

    

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
                    choices=['IQU'],\
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
tomofile_in = args.tomofile
dtnames = args.dtnames
pol = args.pol
Nside = args.Nside
res = args.res
plot = args.plot
save = args.save
part = args.part
distcut = args.distcut
dustfile = 'Data/dust_Nside{}_15arcmin.h5'.format(Nside)#args.dustfile

if tomofile_in == 'tomo1':
    tomofile = 'Data/total_tomography.csv'
elif tomofile_in == 'tomo_new':
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
#if dtnames == :
#dtnames = dtnames.split('+')
#print(dtnames)


##################################
#         Function calls         #
##################################

if pol == 'qu':

    tomo, planck, coord, full_IQU, mask, dist, dPsi = main(planckfile,\
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
    params0 = plotting.data_lines(q[mask], u[mask], Q[mask]*287.45e-6,\
                                  U[mask]*287.45e-6)
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
        unit = 287.45*1e-6
        print('Sampling contribution to submm polarization')
        # Load C_ij from Planck:    
        Cfile = 'Data/Planck_Cij_353_2048_full.h5'
        C_ij = load.C_ij(Cfile, Nside) # Kcmb^2
        C_II = C_ij[0,:] * 1e12 #uKcmb
        C_IQ = C_ij[1,:] * 1e12
        C_IU = C_ij[2,:]* 1e12
        C_QQ = C_ij[3,:]* 1e12
        C_QU = C_ij[4,:]* 1e12
        C_UU = C_ij[5,:]* 1e12

        QU = np.array([Q, U])*unit
        qu = np.array([q, u])


        # Sample visual polarisation
        params_mean, data_mean, data_err =\
                                    tools.sampler_prep(Q, U, q, u, p,\
                                                       C_ij[3:,:], sq, su,\
                                                       tomo[-1], mask,\
                                                       unit=287.45*1e-6,\
                                                       submm=False)

        qu_model, samples, mod_err, samp_err =\
                                    gibbs_sampler(QU, qu, C_ij[3:,:]*1e12,\
                                                  params_mean, data_err,\
                                                  mask, data_mean, p=p,\
                                                  sp=tomo[-1], sq=sq, su=su)
        #
        best_guess_pix = np.array([0.24027794, 0.25232319, 0.28067511,\
                                   0.22065324, 0.27751509, 0.21551718,\
                                   0.28457226, 0.2577072, 0.26976363,\
                                   0.31075199, 0.33175059, 0.22250278,\
                                   0.2449938, 0.24938268, 0.25435339,\
                                   0.21490348, 0.2120685, 0.25049663,\
                                   -0.00206514, 0.00212248])


        sys.exit()

        # sample for submm polarisation
        params_mean, data_mean, data_err =\
                                    tools.sampler_prep(Q, U, q, u, p,\
                                                       C_ij[3:,:], sq, su,\
                                                       tomo[-1], mask,\
                                                       unit=287.45*1e-6,\
                                                       submm=True)
       
        # sampling for all in one: sampler2_mod
        models2, err2 = sampler2(QU, qu, params_mean,\
                              data_err, mask, data_mean)
        QU_model2, QU_star2, QU_bkgr2 = models2[:-2]
        model_err2, star_err2, bkgr_err2 = err2[:-1]
        err_mod2, err_star2, err_bkgr2 = Err_est2(models2[-2], qu[:,mask],\
                                                  qu_err=np.array([sq[mask],\
                                                                   su[mask]]),\
                                                  samples=models2[-1])


        err_model2 = np.full(np.shape(model_err2), np.nan)
        err_model2[:,mask] = err_mod2

        P_b = np.abs(params0[1,:])
        psi_b = 0.5*np.arctan2(par_u[1], par_q[1])
        
        bkgr_params = np.append(P_b, psi_b) 
        params_mean = np.append(np.ones(len(R_Pp))*5, np.zeros(len(R_Pp)+1))
        params_mean[-1] = 1
        data_mean = np.append(R_Pp, bkgr_params)
        data_err = np.append(np.ones(len(R_Pp))*R_err,\
                             np.append(np.std(P_b)*np.ones(len(R_Pp)), 0.1))
        print(np.shape(params_mean), np.shape(data_mean), np.shape(data_err))
        print(Q[mask]*unit/q[mask], U[mask]*unit/u[mask])
        print(np.mean(QU[:,mask]/qu[:,mask], axis=0))
        
        models3, err3 = sampler3(QU, qu, params_mean, data_err,\
                                 mask, data_mean)
        QU_model3, QU_star3, QU_bkgr3 = models3[:-2]
        model_err3, star_err3, bkgr_err3 = err3[:-1]
        
        err_mod3, err_star3, err_bkgr3 = Err_est3(models3[-1],\
                                                p_maxL=models3[-2],\
                                                qu=qu[:,mask],\
                                                qu_err=np.array([sq[mask],\
                                                                 su[mask]]))

        """
        # plot the models; compare with data and with each other
        plotting.plot_models(q, u, sq, su, mask,\
                             data=[Q, U, C_ij],\
                             model1=[QU_model2[0,:], QU_model2[1,:],\
                                     model_err2],\
                             model2=[QU_model3[0,:], QU_model3[1,:],\
                                     model_err3],\
                             part=part, save=save, Nside=Nside)
        """
        #"""
        #plt.show()
        #sys.exit()

        print('')
        #print(model_err3[:,mask])
        #print(err_mod1)
        print(err_mod3[0,:]/(np.sqrt(C_QQ[mask])*unit),\
              err_mod3[1,:]/(np.sqrt(C_UU[mask])*unit),\
              err_mod3[2,:]/(np.sqrt(C_QU[mask])*unit))
        err_model3 = np.full(np.shape(model_err3), np.nan)
        err_model3[:, mask] = err_mod3

        print('Residual polarisation:')
        P_bkgr = np.sqrt(QU_bkgr2[0]**2 + QU_bkgr2[1]**2)
        P_bkgr_err = np.sqrt((bkgr_err2[0]*QU_bkgr2[0])**2\
                             + (bkgr_err2[1]*QU_bkgr2[1])**2)/P_bkgr
        P_bkgr = tools.MAS(P_bkgr, P_bkgr_err)
        #print(P_bkgr, P_bkgr_err)
        #print(P_bkgr/Ps[mask])
        print(P_bkgr/np.mean(Ps[mask]), np.std(P_bkgr/Ps[mask]))
        # plot correlation with slopes:
        plotting.plot_corr2(QU_model3[0,:]/unit, QU_model3[1,:]/unit, q, u,\
                            sq, su, mask, dist, \
                            Nside=Nside, xlab=r'$q,u$', \
                            ylab=r'$Q,U_{{353}}$', title='QU-qu',\
                            save=save+'_model3', part=part, C_ij=err_model3)

        plotting.plot_corr2(QU_model2[0,:]/unit, QU_model2[1,:]/unit, q, u,\
                            sq, su, mask, dist, \
                            Nside=Nside, xlab=r'$q,u$', \
                            ylab=r'$Q,U_{{353}}$', title='QU-qu',\
                            save=save+'_model2', part=part, C_ij=err_model2)

        print('--> Data')
        plotting.plot_corr2(Q, U, q, u, sq, su, mask, dist, \
                            Nside=Nside, xlab=r'$q,u$', \
                            ylab=r'$Q,U_{{353}}$', title='QU-qu',\
                            save=save+'_data', part=part, C_ij=C_ij)        
        plt.show()
        """
        # pol_sampler_mod:
        params_mean = np.array([5, 0, 0])
        params_err = np.array([R_err, std_q[1]*unit, std_u[1]*unit])
        data_mean = np.array([mean_R, par_q[1]*unit, par_u[1]*unit])

        models, params, sigmas = sampler.QU_sampler(QU, qu, params_mean,\
                                                    params_err, mask,\
                                                    data_mean, Nside=Nside,\
                                                    Niter=2000, R_Pp=R_Pp)
        maxL_params, samples = params[:]
        QU_model, QU_star, QU_bkgr = models[:]
        QU_error, params_error = sigmas[:] # (MJy/sr)
        print(mean_R, R_err, params_err)
        print(par_q*unit, par_u*unit, std_q*unit, std_u*unit)
        print(np.mean(QU[:,mask], axis=1), np.std(QU[:,mask], axis=1))
        print(np.mean(QU_star[:,mask], axis=1), np.std(QU_star[:,mask], axis=1))
        print(np.mean(QU_model[:,mask],axis=1),np.std(QU_model[:,mask],axis=1))
        print(np.mean(maxL_params[:,mask], axis=1), np.std(maxL_params[:,mask], axis=1))
        Q_star_err = sampler.star_error(maxL_params[0,mask],\
                                        params_error[0,mask],\
                                        q[mask], sq[mask]/2.)
        U_star_err = sampler.star_error(maxL_params[0,mask],\
                                        params_error[0,mask],\
                                        u[mask], su[mask]/2.)
        
        Q_bkgr_err = params_error[1,mask]
        U_bkgr_err = params_error[2,mask]
        
        #plotting.data_lines(q[mask], u[mask], QU_model[0,mask],QU_model[1,mask])
        #plotting.data_lines(q[mask], u[mask], QU_star[0,mask],QU_star[1,mask])
        #plotting.data_lines(q[mask], u[mask], QU_bkgr[0,mask],QU_bkgr[1,mask])
        #plt.show()
        #sys.exit()
        # Plot correlation plots of models vs visual
        print('--> model')
        
        sampler.correlation(qu[:,mask], QU_model[:,mask], sq[mask], su[mask],\
                            QU_error[:,mask], mask, lab='model', save=save,\
                            R_Pp=[mean_R, np.mean(maxL_params[0, mask])],\
                            R_err=[R_err, np.mean(params_error[0,mask])])
        # data in correlation plotting in sampler module:
        #sampler.correlation(qu[:,mask], np.array([Q[mask]*unit, U[mask]*unit]),\
        #                    sq[mask], su[mask],\
        #                    np.array([np.sqrt(C_QQ[mask])*unit,\
        #                              np.sqrt(C_UU[mask])*unit,\
        #                              np.sqrt(C_QU[mask])*unit]),\
        #                    mask, lab='model', save=save,\
        #                    R_Pp=[mean_R, np.mean(maxL_params[0, mask])],\
        #                    R_err=[R_err, np.mean(params_error[0,mask])])
        
        
        print('--> star')
        sampler.correlation(qu[:,mask], QU_star[:,mask], sq[mask], su[mask],\
                            np.array([Q_star_err, U_star_err,\
                                      np.sqrt(Q_star_err*U_star_err)]), mask, \
                            lab='star', save=save,\
                            R_Pp=[mean_R, np.mean(maxL_params[0, mask])],\
                            R_err=[R_err, np.mean(params_error[0,mask])])
        print('--> bkgr')
        sampler.correlation(qu[:,mask], QU_bkgr[:,mask], sq[mask], su[mask],\
                            np.array([Q_bkgr_err, U_bkgr_err,\
                                      np.sqrt(Q_bkgr_err*U_bkgr_err)]), mask,\
                            lab='bkgr', save=save)
        
        #hp.gnomview(QU_bkgr[0,:], title='Q', rot=[104,22.225])
        #hp.gnomview(QU_bkgr[1,:], title='U', rot=[104,22.225])
        print(np.mean(u[mask]), np.mean(q[mask]), mean_R)

        """
        

        """
        plotting.plot_corr2(QU_model[0,:]/unit, QU_model[1,:]/unit, q, u,\
                            sq, su, mask, dist, \
                            Nside=Nside, xlab=r'$q,u$', \
                            ylab=r'$Q,U_{{353}}$', title='QU-qu',\
                            save=save+'_model', part=part, C_ij=QU_error) 
        """

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
