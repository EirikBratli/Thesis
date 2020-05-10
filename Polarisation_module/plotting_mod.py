"""
Plotting module for the tomography checking with planck polarisation.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import h5py
import sys, time, glob, os
import scipy.optimize as spo
from scipy import stats

from astropy import units as u_
from astropy.coordinates import SkyCoord

import convert_units as cu
import tools_mod as tools
import load_data_mod as load

#####################################

def plot_corr(tomo_map, planck_map, name, mask, dist, Nside=2048, y_lim=None,\
                x_lim=None, xlab=None, ylab=None, title=None, save=None):
    """
    Plot the correlation between tomography and planck353.

    Parameters:
    -----------

    Return:
    -----------
    """
    path = 'Figures/tomography/correlations/'
    print(save)

    R = np.corrcoef(tomo_map[mask], planck_map[mask])
    print('Correlation coefficient of {}:'.format(title))
    print(R)
    print(np.sign(R[1,0]))
    x = np.linspace(np.min(tomo_map[mask]), np.max(tomo_map[mask]), 10)
    y = x * 5.42 / (287.45*1e-6) * (np.sign(R[1,0]))# to uK_cmb,
    #5.42MJy/sr / (287.45 MJy/sr/\mu K_{{cmb}})
    # numbers form planck18, XXI and planck13,IX

    plt.figure('{}'.format(name))
    plt.plot(tomo_map[mask], planck_map[mask], '.k', label='Pearson $R={}$'.\
            format(round(R[1,0], 3)))
    plt.plot(x, y, '-r', label=r'$y=x*18555.45 \mu K_{{cmb}}$')
    plt.title('Correlation of {}'.format(title))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if np.sign(R[1,0]) == -1:
        plt.legend(loc=1)
    else:
        plt.legend(loc=2)
    if y_lim is not None:
        plt.ylim(y_lim)
    if x_lim is not None:
        plt.xlim(x_lim)
    plt.savefig(path + '{}_{}.png'.format(save, Nside))

    plt.figure()

    c_y = np.max(planck_map[mask])*0.1
    plt.hexbin(tomo_map[mask], planck_map[mask], C=dist[mask], bins='log')#,\
    #            label=r'Pearson $R={}$'.format(R[1,0]))

    #plt.plot(x, y, '-r', label='y=x*5.42 MJy/sr * (287.45 uK_{{cmb}}/MJy/sr)')
    cbar = plt.colorbar(pad=0.01)
    cbar.set_label(r'$log(r)$ [pc]')
    plt.title('Correlation of {}, Preason R={}'.format(title, round(R[1,0], 3)))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    """
    if np.sign(R[1,0]) == -1:
        c_x = np.min(tomo_map[mask])*0.9
        plt.text(c_x, c_y, s=r'Pearson $R={}$'.format(round(R[1,0], 3)))
    else:
        c_x = np.min(tomo_map[mask])*0.1
        plt.text(c_x, c_y, s=r'Pearson $R={}$'.format(round(R[1,0], 3)))
    """
    plt.tight_layout()
    plt.savefig(path + '{}_dist_color_{}.png'.format(save, Nside))
    #

def plot_corr2(Q, U, q, u, sq, su, mask, dist, Nside=2048, y_lim=None,\
               x_lim=None, xlab=None, ylab=None, title=None,\
               save=None, part=None):
    """
    Plot Q,U vs q,u correlation with line fitting and errorbars. 
    Also compute the chi^2 
    of the line fitting.
    """
    
    unit1 = 287.45
    unit2 = unit1*1e-6
    ii = np.where(u[mask] == np.max(u[mask]))[0]
    jj = np.where(q[mask] == np.max(q[mask]))[0]
    print('Mask index outliers:', ii, jj)
    #mask = np.delete(mask, jj)
    #mask = np.delete(mask, ii)
    
    #print(U[mask], len(mask))
    path = 'Figures/correlations/'
    QU = np.concatenate((Q[mask], U[mask]), axis=0)
    qu = np.concatenate((q[mask], u[mask]), axis=0)
    #print(np.shape(QU), np.shape(qu))
    R = np.corrcoef(qu, QU)
    print('Correlation coefficient between Q,U and q,u:')
    print(R)
    #print(np.mean(QU*(287.45*1e-6)/qu))
    #print(np.mean(Q[mask]*287.45*1e-6/q[mask]), np.mean(U[mask]*287.45*1e-6/u[mask]))

    # load uncertainties:
    #Cfile = 'Data/Sroll_Cij_353_2048_full.h5'
    Cfile = 'Data/Planck_Cij_353_2048_full.h5'
    C_ij = load.C_ij(Cfile, Nside)
    C_II = C_ij[0,:]
    C_IQ = C_ij[1,:]
    C_IU = C_ij[2,:]
    C_QQ = C_ij[3,:]
    C_QU = C_ij[4,:]
    C_UU = C_ij[5,:]

    print(np.shape(C_ij[:,mask]))
    print('Calculate chi^2')
    print('QU vs qu')
    params, std_params, chi2 = tools.Chi2(Q[mask], U[mask], q[mask], u[mask],\
                                          C_ij[:,mask], sq[mask], su[mask])
    print('Q vs q')
    params_q, std_q, chi2_u = tools.Chi2(Q[mask], None, q[mask], None,\
                                       C_ij[:,mask], sq[mask], None)
    print(np.corrcoef(q[mask], Q[mask]))
    print('U vs u')
    params_u, std_u, chi2_u = tools.Chi2(None, U[mask], None, u[mask],\
                                       C_ij[:,mask], None,su[mask])
    print(np.corrcoef(u[mask], U[mask])) 

    params = params*unit2
    params_q = params_q*unit2
    params_u = params_u*unit2
    std = std_params*unit2
    std_q = std_q*unit2
    std_u = std_u*unit2
    P = np.sqrt(Q[mask]**2 + U[mask]**2)
    p = np.sqrt(q[mask]**2 + u[mask]**2)
    print('R_P/p =', np.mean(P/p)*unit2, np.std(P/p)*unit2)
    R_Pp = np.corrcoef(p, P)
    print('Correlation coefficient between Ps and pv:')
    print(R_Pp)
    print('---------')
    x = np.linspace(-0.1, 0.1, 10)
    y = x * 5.42 * (np.sign(R[1,0])) / (287.45*1e-6) # to MJy/sr,
    #5.42MJy/sr / (287.45 MJy/sr/\mu K_{{cmb}})
    # numbers from planck18, XXI and planck13,IX
    slope, intercept, r_value, p_value, std_err = stats.linregress(qu, QU)
    print('r, p, std (q,u):', r_value, p_value, std_err)
    aq, bq, rq, pq, std_err_q = stats.linregress(q[mask], Q[mask])
    print('r, p, std (q):', rq, pq, std_err_q)
    au, bu, ru, pu, std_err_u = stats.linregress(u[mask], U[mask])
    print('r, p, std (u):', ru, pq, std_err_u)

    print('Slope of R_P/p is {} MJy/sr'.format(slope*unit2), intercept*unit2)
    print(r_value, p_value, std_err)
    print('R_Q/q: {}'.format(aq*unit2), bq*unit2)
    print('R_U/u: {}'.format(au*unit2), bu*287.45*1e-6)

    plt.figure()
    plt.plot(q[mask], Q[mask]*unit2, '^k', label=r'$Q_{s}, q_{v}$')
    plt.plot(u[mask], U[mask]*unit2, 'vb', label=r'$U_{s}, u_{v}$')
    #plt.plot(x, y*(287.45*1e-6), '-r', label=r'(Planck XII 2018) -5.42 MJy/sr')
    #plt.plot(x, intercept*unit2 + slope*x*unit2, '-g',\
    #        label='lin.reg. slope={} MJy/sr'.format(round(slope*unit2, 3)))
    #plt.plot(x, bq*unit2 + aq*x*unit2, '-m',\
    #         label=r'$q=ax+b$: $a={}$ MJy/sr'.format(round(aq*unit2, 3)))
    #plt.plot(x, bu*unit2 + au*x*unit2, '-y',\
    #         label=r'$u=ax+b$: $a={}$ MJy/sr'.format(round(au*287.45*1e-6, 3)))
    #plt.plot(x, params[0]*x + params[1], 'lime',\
    #         label=r'$\chi^2/n$ fit: $a={}$ MJy/sr'.format(round(params[0], 3)))
    l1, = plt.plot(x, y*unit2, color='orange', linestyle='-',\
                   label=r'(Planck XII 2018) -5.42 MJy/sr')
    l2, = plt.plot(x, params[1] + params[0]*x, '-r',\
             label=r'$a_{{QU,qu}}={}\pm{}$ MJy/sr'.\
             format(round(params[0], 3), round(std[0], 3)))
    l3, = plt.plot(x, params_q[1] + params_q[0]*x, '-k',\
             label=r'$a_{{Q,q}}={}\pm{}$ MJy/sr'.\
             format(round(params_q[0], 3),round(std_q[0],3)))
    l4, = plt.plot(x, params_u[1] + params_u[0]*x, '-b',\
             label=r'$a_{{U,u}}={}\pm{}$ MJy/sr'.\
             format(round(params_u[0], 3), round(std_u[0], 3)))

    plt.title('Correlation of {}, Preason R={}'.format(title, round(R[1,0], 3)))
    plt.xlabel(xlab)
    plt.ylabel(ylab + ' [MJy/sr]')
    plt.legend(loc=3)
    plt.xlim(np.min(qu)*1.2, np.max(qu)+0.002)
    plt.ylim(np.min(QU)*unit2-0.005, np.max(QU)*unit2 + 0.005)    
    plt.savefig(path + '{}_corr_QU_qu_{}.png'.format(save, Nside))

    print(save, save[:-1])
    plt.figure()
    e1 = plt.errorbar(q[mask], Q[mask]*unit2, xerr=sq[mask],\
                      yerr=np.sqrt(C_QQ[mask])*unit1,
                      fmt='none', ecolor='k', label=r'$Q_s, q_v$', capthick=2)
    e2 = plt.errorbar(u[mask], U[mask]*unit2, xerr=su[mask],\
                      yerr=np.sqrt(C_UU[mask])*unit1,
                      fmt='none', ecolor='b', label=r'$U_s, u_v$', capthick=2)
    legend1 = plt.legend(handles=[e1, e2], loc=1)
    ax = plt.gca().add_artist(legend1)
    
    l1, = plt.plot(x, y*unit2, color='orange', linestyle='-',\
                   label=r'(Planck XII 2018) -5.42 MJy/sr')
    l2, = plt.plot(x, params[1] + params[0]*x, '-r',\
             label=r'$a_{{QU,qu}}={}\pm{}$ MJy/sr'.\
             format(round(params[0], 3), round(std[0], 3)))
    l3, = plt.plot(x, params_q[1] + params_q[0]*x, '-k',\
             label=r'$a_{{Q,q}}={}\pm{}$ MJy/sr'.\
             format(round(params_q[0], 3),round(std_q[0],3)))
    l4, = plt.plot(x, params_u[1] + params_u[0]*x, '-b',\
             label=r'$a_{{U,u}}={}\pm{}$ MJy/sr'.\
             format(round(params_u[0], 3), round(std_u[0], 3)))
    #plt.plot(x, params[0]*x + params[1], 'lime', 
    #         label=r'$\chi^2/n$ fit: {}MJy/sr'.format(round(params[0], 3)))

    plt.title('Correlation of {}, Pearson R={}, {} stars'.\
              format(title, round(R[1,0], 3), part))
    plt.xlabel(xlab)
    plt.ylabel(ylab + ' [MJy/sr]')
    plt.legend(handles=[l1,l2,l3,l4], loc=3)
    plt.xlim(np.min(qu)*1.2, np.max(qu)+0.002)
    plt.ylim(np.min(QU)*unit2-0.005, np.max(QU)*unit2 + 0.005)
    plt.savefig(path + '{}_corr_QU_qu_{}_ebar.png'.format(save, Nside))

    # P
    Q = Q[mask]; q = q[mask]
    U = U[mask]; u = u[mask]
    C_QQ = C_QQ[mask]; sq = sq[mask]
    C_UU = C_UU[mask]; su = su[mask]
    C_QU = C_QU[mask]
    C_P = (Q**2*C_QQ + U**2*C_UU + 2*Q*U*C_QU)/(P**2)
    sp = np.sqrt(((q*sq)**2 + (u*su)**2)/(q**2 + u**2))

    plt.figure('P/p')
    plt.scatter(p, P*unit2, marker='*', c='k', s=50)
    plt.plot(x, unit2*np.mean(P/p)*x, 'b', label=r'$P/p={}\pm{}$MJy/sr'.\
             format(round(np.mean(P/p)*unit2, 3), round(np.std(P/p)*unit2, 3)))
    plt.plot(x, -params[0]*x, 'r', label=r'$|a_{{QU,qu}}|={}\pm{}$MJy/sr'.\
             format(round(np.abs(params[0]), 3), round(std[0], 3)))
    plt.xlabel(r'$p_v$')
    plt.ylabel(r'$P_s = \sqrt{Q_s^2 + U_s^2}$ [MJy/sr]')
    plt.title(r'Correlation $P_s$ vs $p_v$, R={}, {} stars'.\
              format(round(R_Pp[0,1],3),part))
    plt.legend(loc=2)
    plt.xlim(np.min(p)*0.9, np.max(p)*1.1)
    plt.ylim(np.min(P*unit2)*0.9, np.max(P*unit2)*1.1)
    plt.savefig(path + '{}_corr_Pp_{}.png'.format(save, Nside))

    plt.figure('P ebar')
    plt.errorbar(p, P*unit2, xerr=sp, yerr=np.sqrt(C_P)*unit1,\
                 fmt='none', ecolor='k')
    plt.plot(x, unit2*np.mean(P/p)*x, 'b', label=r'$P/p={}\pm{}$MJy/sr'.\
             format(round(np.mean(P/p)*unit2, 3), round(np.std(P/p)*unit2, 3)))
    plt.plot(x, -params[0]*x, 'r', label=r'$|a_{{QU,qu}}|={}\pm{}$MJy/sr'.\
             format(round(np.abs(params[0]), 3), round(std[0], 3)))
    plt.xlabel(r'$p_v$')
    plt.ylabel(r'$P_s = \sqrt{Q_s^2 + U_s^2}$ [MJy/sr]')
    plt.title(r'Correlation $P_s$ vs $p_v$, R={}, {} stars'.\
              format(round(R_Pp[0,1],3),part))
    plt.legend(loc=2)
    plt.xlim(np.min(p)*0.9, np.max(p)*1.1)
    plt.ylim(np.min(P*unit2)*0.9, np.max(P*unit2)*1.1)
    plt.savefig(path + '{}_ebar_Pp_{}.png'.format(save, Nside))
                 
    plt.show()

def Tomo2Star_pl(Qtomo, Utomo, qtomo, utomo, sq_tomo, su_tomo,\
                 dist_tomo, mask_tomo, planckfile, Nside=2048,\
                 xlab=None, ylab=None, save=None, part=None):
    """
    Plotting function to compare the correlation between Tomography data of RoboPol 
    and the star polarisation data used in Planck XII 2018.
    """
    
    # Get the planck star polarisation data in the given Nside
    # The arrays are masked and ready to be plotted, exept for the C_ij array
    # units: uK_cmb and K_cmb^2
    
    unit1 = 287.45
    unit2 = unit1*1e-6
    print('Comparing polarisation data between Planck XII2018 and Tomography data of RoboPol')
    import planck_star_pol_mod as psp
    Cij_file = 'Data/Planck_Cij_353_2048_full.h5'
    Q_pl, U_pl, q_pl, u_pl, C_ij, sigma_pl, mask_pl = psp.main_pl('Data/vizier_star_table.tsv',\
                                                        planckfile=planckfile,\
                                                        Cij_file=Cij_file,\
                                                        Nside=Nside, return_data=True)
    #
    #print(np.sqrt(C_ij[3,mask_tomo])*unit1)
    #print(sq_tomo[mask_tomo])
    sq_pl = sigma_pl[1]
    su_pl = sigma_pl[-1]
    #print(np.shape(q_pl), np.shape(u_pl), np.shape(mask_pl))
    #print(np.shape(C_ij)) # k_cmb^2
    QU_tomo = np.concatenate((Qtomo[mask_tomo],Utomo[mask_tomo]), axis=0) # uK_cmb
    qu_tomo = np.concatenate((qtomo[mask_tomo],utomo[mask_tomo]), axis=0)
    QU_pl = np.concatenate((Q_pl[mask_pl],U_pl[mask_pl]), axis=0)
    qu_pl = np.concatenate((q_pl[mask_pl],u_pl[mask_pl]), axis=0)

    # Get the linear regression slopes:
    a_tomo, b_tomo, r_tomo, p_tomo, std_tomo = stats.linregress(qu_tomo, QU_tomo*unit2)
    a_pl, b_pl, r_pl, p_pl, std_pl = stats.linregress(qu_pl, QU_pl*unit2)
    
    # Correlation coefficient
    R_pl = np.corrcoef(qu_pl, QU_pl)
    R_tomo = np.corrcoef(qu_tomo, QU_tomo)

    # printing:
    print('........')
    print('-> Planck:')
    print('Pearson correlation R =:')
    print(R_pl)
    print('lin.reg values: (a, b, std)')
    print('ax+b = {}x + {} MJy/sr (+/- {}MJy/sr)'.\
          format(round(a_pl,3), round(b_pl,3), round(std_pl,3)))
    print('r_val = {}, p_val = {}'.format(r_pl, p_pl))
    params_pl, std_pl2, chi2_pl = tools.Chi2(Q_pl[mask_pl], U_pl[mask_pl],\
                                            q_pl[mask_pl], u_pl[mask_pl],\
                                            C_ij[:,mask_pl], sq_pl[mask_pl],\
                                            su_pl[mask_pl])
    PP_pl = Q_pl[mask_pl]**2 + U_pl[mask_pl]**2
    pp_pl = q_pl[mask_pl]**2 + u_pl[mask_pl]**2
    P2p_pl = np.sqrt(PP_pl)*unit2/np.sqrt(pp_pl)
    print('R_P/p (mean, median) =', np.mean(P2p_pl), np.median(P2p_pl))
    print('sigma R_P/p=', np.std(P2p_pl))

    print('-> Tomography:')
    print('Pearson correlation R =:')
    print(R_tomo)
    print('lin.reg values: (a, b, std)')
    print('ax+b = {}x + {} MJy/sr (+/- {}MJy/sr)'.\
          format(round(a_tomo,3), round(b_tomo,3), round(std_tomo,3)))
    print('r_val = {}, p_val = {}'.format(r_tomo, p_tomo))    
    params_tomo, std_tomo2, chi2_tomo = tools.Chi2(Qtomo[mask_tomo],\
                                                  Utomo[mask_tomo],\
                                                  qtomo[mask_tomo],\
                                                  utomo[mask_tomo],\
                                                  C_ij[:,mask_tomo],\
                                                  sq_tomo[mask_tomo],\
                                                  su_tomo[mask_tomo])
    
    PP_t = Qtomo[mask_tomo]**2 + Utomo[mask_tomo]**2
    pp_t = qtomo[mask_tomo]**2 + utomo[mask_tomo]**2
    P2p_t = np.sqrt(PP_t)*unit2/np.sqrt(pp_t)
    print('R_P/p (mean, median) =', np.mean(P2p_t), np.median(P2p_t))
    print('sigma R_P/p=', np.std(P2p_t))
    print('......')
    
    x = np.linspace(-0.03, 0.03, 10)
    params_tomo = params_tomo*unit2
    std_tomo2 = std_tomo2*unit2
    params_pl = params_pl*unit2
    std_pl2 = std_pl2*unit2
    print(params_tomo, params_pl)
    print(std_tomo, std_pl)
    # Plotting: simple correlation map, correlation with error and 2d hist

    plt.figure('Correlation simple')
    s1, = plt.plot(q_pl[mask_pl], Q_pl[mask_pl]*unit2, '*r',\
             label=r'$q_v^{{pl}}, Q_s^{{pl}}$')
    s2, = plt.plot(u_pl[mask_pl], U_pl[mask_pl]*unit2, '*g',\
             label=r'$u_v^{{pl}}, U_s^{{pl}}$')
    s3, = plt.plot(qtomo[mask_tomo], Qtomo[mask_tomo]*unit2, '.k',\
             label=r'$q_v^{{Robo}}, Q_s^{{Robo}}$')
    s4, = plt.plot(utomo[mask_tomo], Utomo[mask_tomo]*unit2, '.b',\
             label=r'$u_v^{{Robo}}, U_s^{{Robo}}$')
    legend1 = plt.legend(handles=[s1,s2,s3,s4], loc=1)
    ax = plt.gca().add_artist(legend1)
    #plt.plot(x, a_tomo*x + b_tomo, color='black', linestyle=':',\
    #         label=r'$a_{{tomo}}={}\pm{}MJy/sr$'.format(round(a_tomo, 2), round(std_tomo, 2)))
    #plt.plot(x, a_pl*x + b_pl, color='black', linestyle='--',\
    #         label=r'$a_{{pl}}={}\pm{}MJy/sr$'.format(round(a_pl, 2), round(std_pl, 2)))
    l3, = plt.plot(x, -5.42*x, linestyle='-', color='orange',\
                   label=r'(Planck XII 2018) -5.42 MJy/sr')
    l1, = plt.plot(x, params_tomo[0]*x + params_tomo[1], color='k', linestyle='-',\
                   label=r'$a_{{RoboPol}}={}\pm{}MJy/sr$'.\
             format(round(params_tomo[0], 2), round(std_tomo2[0],2)))
    l2, = plt.plot(x, params_pl[0]*x + params_pl[1], color='k', linestyle='--',\
             label=r'$a_{{pl}}={}\pm{}MJy/sr$'.\
             format(round(params_pl[0], 3), round(std_pl2[0], 2)))
    

    plt.ylabel(ylab + ' [MJy/sr]')
    plt.xlabel(xlab)
    plt.legend(handles=[l1,l2,l3], loc=3)
    plt.xlim(-0.02, 0.02)
    plt.ylim(-0.15, 0.1)
    #plt.tight_layout()
    plt.savefig('Figures/correlations/{}planck_vs_tomo_corr{}.png'.format(save,Nside))
    
    plt.figure('tomo corr')
    # find best fit for Uu and Qu
    aq, bq, r_q, p_q, std_q = stats.linregress(qtomo[mask_tomo], Qtomo[mask_tomo]*unit2)
    au, bu, r_u, p_u, std_u = stats.linregress(utomo[mask_tomo], Utomo[mask_tomo]*unit2)
    print('a_q x + b_q = {}x + {} (+/-{}) MJy/sr'.format(round(aq,3),round(bq,3),round(std_q,3)))
    print('a_u x + b_u = {}x + {} (+/-{}) MJy/sr'.format(round(au,3),round(bu,3),round(std_u,3)))
    
    params_q, std_q, chi2_q = tools.Chi2(Qtomo[mask_tomo], None,\
                                         qtomo[mask_tomo], None,\
                                         C_ij[:,mask_tomo],\
                                         sq_tomo[mask_tomo], None)
    params_u, std_u, chi2_u = tools.Chi2(None, Utomo[mask_tomo],\
                                         None, utomo[mask_tomo],\
                                         C_ij[:,mask_tomo],\
                                         None, su_tomo[mask_tomo])
    
    
    print('only q:', params_q*unit2)
    print('only u:', params_u*unit2)
    params_q = params_q*unit2
    params_u = params_u*unit2
    std_q = std_q*unit2
    std_u = std_u*unit2
    q = qtomo[mask_tomo]
    u = utomo[mask_tomo]
    Q = Qtomo[mask_tomo]
    U = Utomo[mask_tomo]
    Rq = np.corrcoef(q, Q)
    Ru = np.corrcoef(u, U)
    pv = np.sqrt(q**2 + u**2)
    Ps = np.sqrt(Q**2 + U**2)
    Rp = np.corrcoef(pv, Ps)
    print('R_q:', Rq)
    print('R_u:', Ru)
    print('R_p:', Rp)
    #print(dist_tomo[mask_tomo])
    #ind = np.where(dist_tomo[mask_tomo] <= 360)[0]
    #ind2 = np.where(dist_tomo[mask_tomo] > 360)[0]
    #print(len(ind), len(ind2))
    plt.plot(q, Q*unit2, 'vk', label=r'$q_v$, $Q_s$') # LVC
    plt.plot(u, U*unit2, '^b', label=r'$u_v$, $U_s$')
    #plt.plot(q[ind], Q[ind]*unit2, 'xg', label=r'$q_v$, $Q_s$')# no LVC
    #plt.plot(u[ind], U[ind]*unit2, 'xr', label=r'$u_v$, $U_s$')
    plt.plot(x, -5.42*x, '-g', label=r'(Planck XII 2018) -5.42 MJy/sr')
    plt.plot(x, params_q[0]*x + params_q[1], '-k', label=r'$a_{{Qq}}={}\pm{}$'.\
             format(round(params_q[0],2), round(std_q[0],2)))
    plt.plot(x, params_u[0]*x + params_u[1], '-b', label=r'$a_{{Uu}}={}\pm{}$'.\
             format(round(params_u[0],2), round(std_u[0],2)))
    plt.plot(x, params_tomo[0]*x + params_tomo[1], color='red', linestyle='-',\
             label=r'$a_{{QU,qu}}={}\pm{}$MJy/sr'.format(round(params_tomo[0],2),\
                                                        round(std_tomo2[0],2)))
    
    
    plt.ylabel(ylab + ' [MJy/sr]')
    plt.xlabel(xlab)
    plt.legend(loc=3)
    plt.xlim(-0.02, 0.02)
    plt.ylim(-0.15, 0.1)
    plt.savefig('Figures/correlations/{}tomo_corr_{}.png'.format(save, Nside))

    #
    #print(np.sqrt(np.mean(C_ij[3, mask_pl]))*unit1, np.sqrt(np.mean(C_ij[3, mask_tomo]))*unit1)
    #print(np.mean(sq_pl[mask_pl]), np.mean(sq_tomo[mask_tomo]))
    #Cqq = np.sqrt(C_ij[3,:])*unit1
    #Cqq[mask_pl] = 0.25
    #Cqq[mask_tomo] = 0.25
    #hp.mollview(Cqq, title=r'$C_{{QQ}}^{{1/2}}$')#, min=0, max=0.02)
    #plt.savefig('Figures/correlations/test/C_qq2.png')
    #hp.mollview(su_tomo, title=r'$\sigma_{u}$', min=0, max=0.002)
    #plt.savefig('Figures/correlations/test/su.png')



    plt.figure('Correlation errorbars')
    p1 = plt.errorbar(q_pl[mask_pl], Q_pl[mask_pl]*unit2, xerr=sq_pl[mask_pl],\
                 yerr=np.sqrt(C_ij[3,mask_pl])*unit1, fmt='none', ecolor='r',\
                 label=r'$q_v^{{pl}}, Q_s^{{pl}}$')
    p2 = plt.errorbar(u_pl[mask_pl], U_pl[mask_pl]*unit2, xerr=su_pl[mask_pl],\
                 yerr=np.sqrt(C_ij[5,mask_pl])*unit1, fmt='none', ecolor='g',\
                 label=r'$u_v^{{pl}}, U_s^{{pl}}$')
    p3 = plt.errorbar(qtomo[mask_tomo], Qtomo[mask_tomo]*unit2,\
                      xerr=sq_tomo[mask_tomo],\
                      yerr=np.sqrt(C_ij[3,mask_tomo])*unit1, fmt='none',\
                      ecolor='k', label=r'$q_v^{{Robo}}, Q_s^{{Robo}}$')
    p4 = plt.errorbar(utomo[mask_tomo], Utomo[mask_tomo]*unit2,\
                      xerr=su_tomo[mask_tomo],\
                      yerr=np.sqrt(C_ij[5,mask_tomo])*unit1,fmt='none',\
                      ecolor='b', label=r'$u_v^{{Robo}}, U_s^{{Robo}}$')
    legend1 = plt.legend(handles=[p1,p2,p3,p4], loc=1)
    ax = plt.gca().add_artist(legend1)

    #plt.plot(x, a_tomo*x + b_tomo, ':k', label=r'$a_{{tomo}}={}\pm{}MJy/sr$'.\
    #         format(round(a_tomo, 2), round(std_tomo, 2)))
    #plt.plot(x, a_pl*x + b_pl, '--k', label=r'$a_{{pl}}={}\pm{}MJy/sr$'.\
    #         format(round(a_pl, 2), round(std_pl, 2)))
    l3, = plt.plot(x, -5.42*x, linestyle='-', color='orange',\
                   label=r'(Planck XII 2018) -5.42 MJy/sr')
    p5, = plt.plot(x, params_tomo[0]*x + params_tomo[1], color='k', linestyle='-',\
                   label=r'$a_{{Robo}}^{{\chi^2}}={}\pm{}MJy/sr$'.\
                   format(round(params_tomo[0], 2), round(std_tomo2[0], 2)))
    p6, = plt.plot(x, params_pl[0]*x + params_pl[1], color='k', linestyle='--',\
                   label=r'$a_{{pl}}^{{\chi^2}}={}\pm{}MJy/sr$'.\
                   format(round(params_pl[0], 2), round(std_pl2[0],2)))

    plt.xlabel(ylab + ' [MJy/sr]')
    plt.xlabel(xlab)
    plt.legend(handles=[p5,p6,l3], loc=3)
    plt.xlim(-0.02,0.02)
    plt.ylim(-0.15, 0.1)
    plt.savefig('Figures/correlations/{}planck_vs_tomo_ebar{}.png'.format(save,Nside))
   
    plt.figure('2D histogram 1')
    
    plt.hist2d(qu_pl, QU_pl*unit2, bins=50, cmap='brg', cmin=0.1,\
               label='Planck')
    cbar1 = plt.colorbar(pad=0.01)
    cbar1.set_label('counts, Planck XII 2018')
    plt.plot(x, -5.42*x, 'orange', label=r'(Planck XII 2018) -5.42 MJy/sr')
    plt.plot(x, params_tomo[0]*x + params_tomo[1], color='k', linestyle='-',\
             label=r'$a_{{RoboPol}}={}\pm{}MJy/sr$'.\
             format(round(params_tomo[0], 2), round(std_tomo2[0])))
    plt.plot(x, params_pl[0]*x + params_pl[1], color='k', linestyle='--',\
             label=r'$a_{{pl}}={}\pm{}MJy/sr$'.\
             format(round(params_pl[0], 2), round(std_pl2[0], 2)))


    plt.ylabel(ylab + ' [MJy/sr]')
    plt.xlabel(xlab)
    plt.legend(loc=3)
    plt.xlim(-0.02, 0.02)
    plt.ylim(-0.15, 0.1)
    #plt.tight_layout()
    plt.savefig('Figures/correlations/{}planck_2dhist{}.png'.format(save,Nside))
     
    plt.figure('2D histogram 2')
    plt.hist2d(qu_tomo, QU_tomo*unit2, bins=50, cmap='brg', cmin=0.1,\
               label='Tomography')
    cbar = plt.colorbar(pad=0.01)
    #plt.plot(x, a_tomo*x + b_tomo, '-r', label=r'$a_{{tomo}}={}\pm{}MJy/sr$'.\
    #         format(round(a_tomo, 2), round(std_tomo, 2)))
    #plt.plot(x, a_pl*x + b_pl, '-g', label=r'$a_{{planck}}={}\pm{}MJy/sr$'.\
    #         format(round(a_pl, 2), round(std_pl, 2)))
    plt.plot(x, -5.42*x, 'orange', label=r'(Planck XII 2018) -5.42 MJy/sr')    
    plt.plot(x, params_tomo[0]*x + params_tomo[1], color='k', linestyle='-',\
             label=r'$a_{{RoboPol}}={}\pm{}MJy/sr$'.\
             format(round(params_tomo[0], 2), round(std_tomo2[0])))
    plt.plot(x, params_pl[0]*x + params_pl[1], color='k', linestyle='--',\
             label=r'$a_{{pl}}={}\pm{}MJy/sr$'.\
             format(round(params_pl[0], 2), round(std_pl2[0], 2)))

    #plt.subplots_adjust(wspace=0, hspace=0.2)
    plt.ylabel(ylab + ' [MJy/sr]')
    plt.xlabel(xlab)
    cbar.set_label('counts, RoboPol')
    plt.legend(loc=3)
    plt.xlim(-0.02, 0.02)
    plt.ylim(-0.15, 0.1)
    #plt.tight_layout()
    plt.savefig('Figures/correlations/{}tomo_2dhist{}.png'.format(save,Nside))

    # For P:
    plt.figure('P polarisation')
    sq = sq_tomo[mask_tomo]
    su = su_tomo[mask_tomo]
    sp = np.sqrt(((q*sq)**2 + (u*su)**2)/(q**2 + u**2))
    sQ = C_ij[3,mask_tomo]
    sU = C_ij[5,mask_tomo]
    sP = np.sqrt(((Q*sQ)**2 + (U*sU)**2)/(Q**2 + U**2))
    q_p = q_pl[mask_pl]; u_p = u_pl[mask_pl]
    Q_p = Q_pl[mask_pl]; U_p = U_pl[mask_pl]
    sq_p = sq_pl[mask_pl]
    su_p = su_pl[mask_pl]
    sQ_p = C_ij[3,mask_pl]
    sU_p = C_ij[5,mask_pl]
    sP_p = np.sqrt(((Q_p*sQ_p)**2 + (U_p*sU_p)**2)/(Q_p**2 + U_p**2))
    sp_p = np.sqrt(((q_p*sq_p)**2 + (u_p*su_p)**2)/(q_p**2 + u_p**2))
    Ps_p = np.sqrt(Q_p**2 + U_p**2)
    pv_p = np.sqrt(q_p**2 + u_p**2)
    
    plt.scatter(pv_p, Ps_p*unit2, marker='^', color='g')
    plt.scatter(pv, Ps*unit2, marker='*', color='k')
    #plt.errorbar(pv, Ps*unit2, xerr=sp, yerr=np.sqrt(C_ij[4,mask_tomo])*unit1,\
    #             fmt=None, ecolor='k', label=r'$p_v^{{Robo}}, P_s^{{Planck}}$')
    plt.plot(x, -params_tomo[0]*x, '-b',\
             label=r'$a = {}\pm{} MJy/sr$'.\
             format(round(params_tomo[0],2),round(std_tomo2[0],2)))
    plt.plot(x, np.mean(P2p_t)*x, ':b',\
             label=r'$R_{{P/p}} = {}\pm{} MJy/sr$'.\
             format(round(np.mean(P2p_t),2),round(np.std(P2p_t),2)))

    plt.plot(x, -params_pl[0]*x, '-r',\
             label=r'$a_{{pl}} = {}\pm{} MJy/sr$'.\
             format(-round(params_pl[0],2),round(std_pl2[0], 2)))
    plt.plot(x, np.mean(P2p_pl)*x, ':r',\
             label=r'$R^{{pl}}_{{P/p}} = {}\pm{} MJy/sr$'.\
             format(round(np.mean(P2p_pl),2),round(np.std(P2p_pl),2)))

    plt.ylabel(ylab + ' [MJy/sr]')
    plt.xlabel(xlab)
    plt.legend(loc=4)
    plt.xlim(-0.0005, 0.02)
    plt.ylim(-0.005, 0.1)
    plt.savefig('Figures/{}P_pol_corr_{}.png'.format(save,Nside))

    #plt.show()
    #

def plot_corr_stars(Qs, Us, qv, uv, sq, su, r, mask, Nside=2048, y_lim=None,\
                    x_lim=None, x_lab=None, y_lab=None, title=None,\
                    save=None, part=None):
    """
    Function to plot correlation plot for stars with distance over 360 pc 
    against submillimeter polarisation.
    """

    unit1 = 287.45 # MJy/sr/Kcmb
    unit2 = unit1*1e-6

    path = 'Figures/correlations/Star/'
    QU = np.concatenate((Qs, Us), axis=0)
    qu = np.concatenate((qv, uv), axis=0)

    Ps = np.sqrt(Qs**2 + Us**2)
    pv = np.sqrt(qv**2 + uv**2)
    sp = np.sqrt(((sq*qv)**2 + (su*uv)**2)/(qv**2 + uv**2))
    #print(sp)
    # get covariance matrix of planck:
    C_ij = load.C_ij('Data/Planck_Cij_353_2048_full.h5', Nside)
    C_II = C_ij[0,mask]
    C_IQ = C_ij[1,mask]
    C_IU = C_ij[2,mask]
    C_QQ = C_ij[3,mask]
    C_QU = C_ij[4,mask]
    C_UU = C_ij[5,mask]
    print(len(C_QQ), len(Qs), len(qv))
    C_P = ((Qs**2*C_QQ) + (Us**2*C_UU))/(Qs**2 + Us**2)

    # Calculate Pearson correlation coeff:
    R_QUqu = np.corrcoef(qu, QU)
    R_Qq = np.corrcoef(qv, Qs)
    R_Uu = np.corrcoef(uv, Us)
    R_Pp = np.corrcoef(pv, Ps)
    print('Correlation coefficient between Q,U and q,u:')
    print(R_QUqu)
    print('Correlation coefficient between Q and q:')
    print(R_Qq)
    print('Correlation coefficient between U and u:')
    print(R_Uu)
    print('Correlation coefficient between Ps and pv:')
    print(R_Pp)
    
    # chi^2:
    print('Calculate chi^2')
    print('QU vs qu')
    params, std_params, chi2 = tools.Chi2(Qs, Us, qv, uv,\
                                          C_ij[:,mask], sq, su)
    print('Q vs q')
    params_q, std_q, chi2_u = tools.Chi2(Qs, None, qv, None,\
                                       C_ij[:,mask], sq, None)
    print('U vs u')
    params_u, std_u, chi2_u = tools.Chi2(None, Us, None, uv,\
                                       C_ij[:,mask], None,su)
    
    # Convert to MJy/sr:
    params = params*unit2
    params_q = params_q*unit2
    params_u = params_u*unit2
    std_params = std_params*unit2
    std_q = std_q*unit2
    std_u = std_u*unit2
    Ps = Ps*unit2
    
    # 
    x = np.linspace(-0.1, 0.1, 10)
    plt.figure('corr')
    plt.scatter(qv, Qs*unit2, c='k', vmin=0, vmax=2000, marker='*',\
                s=15)
    plt.scatter(uv, Us*unit2, c='b', vmin=0, vmax=2000, marker='*',\
                s=15)
    #cbar = plt.colorbar()
    #cbar.set_label('distance [pc]')
    plt.plot(x, params[0]*x + params[1], 'k', label=r'$a={}\pm{}$MJy/sr'.\
             format(round(params[0],2),round(std_params[0],2)))
    plt.plot(x, params_q[0]*x + params_q[1], 'r',\
             label=r'$a_Q={}\pm{}$MJy/sr'.\
             format(round(params_q[0],2),round(std_q[0],2)))
    plt.plot(x, params_u[0]*x + params_u[1], 'b',\
             label=r'$a_U={}\pm{}$MJy/sr'.\
             format(round(params_u[0],2),round(std_u[0],2)))
    plt.plot(x, -np.mean(Ps/pv)*x, 'g', label=r'$-P/p=-{}\pm{}$MJy/sr'.\
             format(round(np.mean(Ps/pv),2),round(np.std(Ps/pv),2)))
    
    plt.xlabel(x_lab)
    plt.ylabel(y_lab + ' [MJy/sr]')
    plt.title('Correlation, Pearson R={}, {} stars'.\
              format(round(R_QUqu[0,1],3), part))
    plt.legend(loc=3)
    plt.xlim(-0.025, 0.005)
    plt.ylim(-0.05, 0.1)
    plt.savefig(path + 'corr_QUqu_star_{}.png'.format(save))

    #
    plt.figure('ebar')
    plt.errorbar(qv, Qs*unit2, xerr=sq, yerr=np.sqrt(C_QQ)*unit1,\
                 fmt='none', ecolor='k')
    plt.errorbar(uv, Us*unit2, xerr=su, yerr=np.sqrt(C_UU)*unit1,\
                 fmt='none', ecolor='b')
    
    plt.plot(x, params[0]*x + params[1], 'r', label=r'$a={}\pm{}$MJy/sr'.\
             format(round(params[0],2),round(std_params[0],2)))
    plt.plot(x, params_q[0]*x + params_q[1], 'k',\
             label=r'$a_Q={}\pm{}$MJy/sr'.\
             format(round(params_q[0],2),round(std_q[0],2)))
    plt.plot(x, params_u[0]*x + params_u[1], 'b',\
             label=r'$a_U={}\pm{}$MJy/sr'.\
             format(round(params_u[0],2),round(std_u[0],2)))
    plt.plot(x, -np.mean(Ps/pv)*x, 'g', label=r'$-P/p=-{}\pm{}$MJy/sr'.\
             format(round(np.mean(Ps/pv),2),round(np.std(Ps/pv),2)))
    
    plt.xlabel(x_lab)
    plt.ylabel(y_lab + ' [MJy/sr]')
    plt.title('Correlation, Pearson R={}, {} stars'.\
              format(round(R_QUqu[0,1],3), part))
    plt.legend(loc=1)
    plt.xlim(-0.03, 0.015)
    plt.ylim(-0.1, 0.15)
    plt.savefig(path + 'ebar_QUqu_star_{}.png'.format(save))
    #
    
    plt.figure('P')
    plt.scatter(pv, Ps, marker='*', color='g')
    #if (part != 'all') or (part is not None):
    #    plt.errorbar(pv, Ps, xerr=sp, yerr=np.sqrt(C_P)*unit1,\
    #                 fmt='none', ecolor='g')

    plt.plot(x, -params[0]*x + params[1], 'b',\
             label=r'$|a|={}\pm{}$MJy/sr'.\
             format(round(abs(params[0]),2),round(std_params[0],2)))
    plt.plot(x, np.mean(Ps/pv)*x, 'k', label=r'$P/p={}\pm{}$MJy/sr'.\
             format(round(np.mean(Ps/pv),2),round(np.std(Ps/pv),2)))

    plt.xlabel(r'$p_v$ [$\%$]')
    plt.ylabel(r'$P_s$ [MJy/sr]')
    plt.title('Correlation, Pearson R={}, {} stars'.\
              format(round(R_Pp[0,1],3), part))
    plt.legend(loc=4)
    plt.xlim(np.min(pv)*0.9, np.max(pv)*1.1)
    plt.ylim(np.min(Ps)*0.9, np.max(Ps)*1.1)
    plt.savefig(path + 'corr_Pp_star_{}.png'.format(save))

    #
    
    
    #

def plot_gnom(map, lon, lat, label, mask=None, Nside=2048, unit=None,\
              project=None, save=None, range=None):
    """
    Plotting function viewing in gnomonic projection.

    Parameters:
    -----------
    - map, array. The map to plot.
    - lon, scalar. mean position in longitude
    - lat, scalar. mean position in latitude
    - label, string.  one of Q, U, P, q, u, p.
    -
    -


    Return:
    -------
    """
    
    #sys.exit()

    path = 'Figures/Smooth_figs/'
    print(label, project)
    res = hp.nside2resol(Nside)*(180/np.pi)*60
    m = np.full(len(map), hp.UNSEEN)
    if mask is None:
        m = map
    else:
        m[mask] = map[mask]
    print(m)
    if range is not None:
        min = range[0]
        max = range[1]
        hp.gnomview(m, title=r'${}$ polarisation {}'.format(label,project),\
                    rot=[lon, lat], xsize=100, unit=unit, cmap='jet',\
                    min=min, max=max)
    
    else:
        hp.gnomview(m, title=r'${}$ polarisation {}'.format(label,project),\
                    rot=[lon, lat], xsize=100, unit=unit, cmap='jet')
    
        #hp.gnomview(m, title=r'${}$ polarisation {}'.\
        #            format(label,project), rot=[lon, lat],\
        #            xsize=100, unit=unit, cmap='jet', reso=res)
    hp.graticule()
    plt.savefig(path + '{}{}_{}_{}.png'.format(save,label, project, Nside))
    
    ###
def plot_DeltaPsi(Psi, mask, name=None, Nside=2048):
    """
    Plot difference in polarisation angle between submillimeter and visual.
    """
    if Nside == 256:
        b = 10
    else:
        b = int(len(mask)/10)

    m = np.median(Psi[mask])*180/np.pi
    plt.figure()
    plt.hist(Psi[mask]*180/np.pi, bins=b, histtype='step', color='k',\
            normed=True)
    plt.axvline(m, c='r', ls=':', label=r'median($\Delta\psi_{{s/v}}$)={}'.\
                format(m))
    plt.xlabel(r'$\Delta \psi_{{s/v}}$ [deg]')
    plt.ylabel('Probability density')
    plt.legend(loc=2)
    # apply sigma
    plt.savefig('Figures/tomography/Delta_psi_sv_Ns{}.png'.format(Nside))



def plot_ratios(tomo, planck, dust, Ebv_file, mask, Nside=2048, save=None,\
                label=None, name=None):
    """
    Something wrong in the values??? try tomo_main.py??
    """


    # Do P/v
    fac = (287.45*1e-6) # to uK_cmb, 1/MJy/sr/uK_cmb
    R_P2p, mean_P2p = tools.ratio_P2p(tomo, planck, mask, Nside)
    R_P2p *= fac
    mean_P2p *= fac
    for i in (mask):
        print(i, R_P2p[i], planck[i], tomo[i])
    p = np.arange(hp.nside2npix(Nside))
    ii = np.isin(p, mask, invert=True)
    R_P2p[ii] = hp.UNSEEN
    print(mean_P2p, np.median(R_P2p[mask]))
    R1 = (R_P2p[mask])
    print(np.where(R_P2p[mask] > 500))
    R1 = R1[np.where(np.abs(R_P2p[mask]) < 50)]
    print(np.mean(R1))
    #hp.gnomview(R_P2p, rot=[104, 22.2], xsize=100)
    plt.figure()
    plt.plot(R1, '.k')
    #plt.plot(R_P2p[mask], '.k')
    #plt.axhline(mean_P2p, c='r', ls='--', label=r'mean $R_{{P/p}}={}$ MJy/sr'.\
    #            format(round(mean_P2p, 3)))
    plt.axhline(np.mean(R1), c='g', ls='--',\
                label=r'$R_{{P/p}}={}$ MJy/sr'.format(np.mean(R1)))
    plt.legend(loc=4)
    plt.ylabel(r'$R_{P/p}$ [MJy/sr]')
    plt.show()
    sys.exit()
    # Do S/V
    # Read reddening file
    Ebv_star = tools.Read_H5(Ebv_file, 'Ebv')
    R_S2v, mean_S2v = tools.ratio_S2V(tomo, planck, Ebv_star, mask)
    print(mean_S2v/fac)
    R2 = (R_S2v[mask]/fac)
    R2 = R2[np.where(R_S2v[mask]/fac < 500)]
    print(np.where(R_S2v[mask]/fac > 500))
    plt.figure()
    #plt.plot(R_S2v[mask]/fac, '.k')
    plt.plot(R2, '.k')
    #plt.axhline(mean_S2v/fac, c='r', ls='--', label=r'mean $R_{{s/v}}={}$ MJy/sr'.\
    #            format(mean_S2v/fac))
    plt.axhline(np.mean(R2), c='g', ls='--',\
                label=r'$R_{{P/p}}=5.42$ MJy/sr (Planck Collab. 2018 XII)')
    plt.legend(loc=4)
    plt.ylabel(r'$R_{P/p}$ [MJy/sr]')
    plt.show()
    sys.exit()
    pass




#
