"""
Module to analyse sampling results.
"""
import numpy as np
import matplotlib.pyplot as plt

import comp_intensity_mod as cim 
from stat_mod import logLikelihood, logPrior

def plot_model(Gibbs_steps, nu, data, model0, model, params, std_p):
    b, T, beta_d, A_cmb, A_s, beta_s = params[:]
    sb, sT, sbeta_d, sA_cmb, sA_s, sbeta_s = std_p[:]

    plt.figure('model components')
    plt.loglog(nu, data, 'xk', label='mean data')
    plt.plot(nu, cim.MBB(nu, b, T, beta_d), '-b', label='dust')
    plt.plot(nu, cim.I_CMB(nu, A_cmb), '-r', label='CMB')
    plt.plot(nu, cim.I_sync(nu, A_s, beta_s), '-g', label='sync')
    plt.plot(nu, model, '--k', label='model')
    plt.xlabel('Frequency [GHz]')
    plt.ylabel(r'Brightness temperature [$\mu K_{{RJ}}$]')
    plt.legend(loc=3)
    plt.ylim(0.01, 1000)
    plt.savefig('Figures/mean_data_intensity{}.png'.format(Gibbs_steps))
    #

def print_results(nu, data, params, Gibbs_steps, Npix):
    shape = Gibbs_steps*Npix
    b_array = np.reshape(params[:,:,0], shape)
    T_array = np.reshape(params[:,:,1], shape)
    beta_dust_array = np.reshape(params[:,:,2], shape)
    A_cmb_array = np.reshape(params[:,:,3], shape)
    A_sync_array = np.reshape(params[:,:,4], shape)
    beta_sync_array = np.reshape(params[:,:,5], shape)

    b = np.mean(b_array)
    T = np.mean(T_array)
    beta_dust = np.mean(beta_dust_array)
    A_cmb = np.mean(A_cmb_array)
    A_sync = np.mean(A_sync_array)
    beta_sync = np.mean(beta_sync_array)

    sb = np.std(b_array)
    sT = np.std(T_array)
    sbeta_dust = np.std(beta_dust_array)
    sA_cmb = np.std(A_cmb_array)
    sA_sync = np.std(A_sync_array)
    sbeta_sync = np.std(beta_sync_array)
    print('---------------------------')
    print('Mean and standard deviation')
    print('Mean b: {}+/-{}'.format(b, sb))
    print('Mean T_d: {}+/-{}'.format(T, sT))
    print('Mean beta_dust: {}+/-{}'.format(beta_dust, sbeta_dust))
    print('Mean A_cmb: {}+/-{}'.format(A_cmb, sA_cmb))
    print('Mean A_sync: {}+/-{}'.format(A_sync, sA_sync))
    print('Mean beta_sync: {}+/-{}'.format(beta_sync, sbeta_sync))
    print(' ')
    print('Likelihood, model + initial model')
    p = [b, T, beta_dust, A_cmb, A_sync, beta_sync] # need the values
    std_p = [sb, sT, sbeta_dust, sA_cmb, sA_sync, sbeta_sync]
    model = cim.Model(nu, b=b, T=T, beta_d=beta_dust, A_cmb=A_cmb,\
                            A_s=A_sync, beta_s=beta_sync)
    model0 = cim.MBB(nu) + cim.I_CMB(nu) + cim.I_sync(nu)

    like = logLikelihood(model, data)
    like0 = logLikelihood(model0, data)
    print(like)   # need this
    print(like0)
    
    #### Plotting ####
    plot_model(Gibbs_steps, nu, np.mean(data, axis=1), model0, model, p, std_p)
    plt.show()
