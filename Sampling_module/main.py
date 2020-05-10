"""
The main program for sampling and fitting models to the planck data.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import sys, time
#import h5py
from functools import partial
import convert_units as cu

# import the modules
from comp_intensity_mod import Model
from metropolis_mod import Initialize, MetropolisHastings
from stat_mod import logLikelihood, logPrior, Cov
import planck_map_mod as planck
import result_mod as res

def main(Nside, Gibbs_steps, pfiles, nu, mean, err, data_mean):
    """
    Main function to run sampling module. First load data, initial guess values,
    Run Gibbs sampling with MH, print and plot results
    """
    Npix = hp.nside2npix(Nside)
    t0 = time.time()

    # load Planck maps
    maps = planck.ChangeMapUnits(pfiles, nu)
    # fix resolution
    new_map = np.zeros((len(maps), Npix))
    for i, m in enumerate(maps):
        new_map[i,:] = planck.fix_resolution(m, Nside)
        print(i, nu[i], new_map[i,5],np.min(m), np.max(m))

    new_map, index, mask = planck.remove_badpixel(new_map, Npix=Npix)
    ind = index.values()
    ind = ind
    print(ind)
    data = new_map[:,mask] * 1e6 # convert to uK_rj

    t1 = time.time()
    print('Loading and preparing time: {}s'.format(t1-t0))
    print('=============================================')

    # initial guesss on params:
    mean_b = mean[:1]
    x1_mean = mean[1:]
    # uncertainty guesses
    err_x1 = err[1:]
    err_b = err[:1]

    cov0 = Cov(len(x1_mean))
    cov_b0 = Cov(len(mean_b))
    sigma = 10.

    # arrays to store values:
    params_array = np.zeros((Gibbs_steps, Npix, len(mean)))
    maxL_params_list = np.zeros((Gibbs_steps, len(mean)))
    for i in range(Gibbs_steps):
        #print(' ')
        print('-- Gibbs step: {} --'.format(i))
        t2 = time.time()
        print('Calculate "T, beta_d, A_cmb, A_s, beta_s", given "b"')
        for pix in range(len(data[0,:])):
            ii = np.where(data[:,pix] < -1e4)[0]
            if len(ii) > 0:
                print(ii, data[:,pix])
                continue
            # set up input functions
            log_like = partial(logLikelihood, data=data[:,pix])
            log_prior = partial(logPrior, mu=data_mean[1:], sigma=data_err[1:])
            Model_func = partial(Model, b=mean_b)

            # Initialize the parameters, model, etc.
            params0, model0, loglike0, logprior0 = Initialize(nu, log_like,\
                                                        log_prior, Model_func,\
                                                        x1_mean, cov0, mean_b)
            # test initial values, if init log like is less than -1e4, make new
            # initial values. because bad parameters.
            ll0 = loglike0
            c = 0
            while loglike0 < -1e4:
                #print('hei', pix)
                c += 1
                params0, model0, loglike0, logprior0 = Initialize(nu, log_like,\
                                                        log_prior, Model_func,\
                                                        x1_mean, cov0, mean_b)
                if c == 10:
                    break
                #
            # sample parameters:
            params, par_maxL = MetropolisHastings(nu, log_like, log_prior,\
                                        Model_func, sigma, params0, model0,\
                                        loglike0, logprior0, x1_mean, cov0,\
                                        len(x1_mean), mean_b)
            params_array[i, pix, 1:] = params
        # end pixel loop
        #print(params)
        maxL_params_list[i, 1:] = par_maxL
        x1_mean = par_maxL + np.random.normal(np.zeros(len(par_maxL)),\
                                                np.fabs(par_maxL)/30.)
        print('Calculate "b" given "T, beta_d, A_cmb, A_s, beta_s"')
        print(params_array[i,:,1:])

        # set up new input functions
        log_like = partial(logLikelihood, data=np.mean(data, axis=1)) #       ??
        log_prior = partial(logPrior, mu=data_mean[:1], sigma=data_err[:1])
        Model_func = partial(Model, T=params[0], beta_d=params[1],\
                            A_cmb=params[2], A_s=params[3], beta_s=params[4])

        # Initialize:
        params0, model0, loglike0, logprior0 = Initialize(nu, log_like,\
                                                    log_prior, Model_func,\
                                                    mean_b, cov_b0, x1_mean)
        # test initial values:
        c = 0
        while loglike0 < -1e4:
            c += 1
            params0, model0, loglike0, logprior0 = Initialize(nu, log_like,\
                                                        log_prior, Model_func,\
                                                        mean_b, cov_b0, x1_mean)
            if c > 10:
                break
            #
        # Sample b
        b, maxL_b = MetropolisHastings(nu, log_like, log_prior, Model_func,\
                                sigma, params0, model0, loglike0, logprior0,\
                                mean_b, cov_b0, len(mean_b), params)
        params_array[i,:,0] = b
        mean_b = maxL_b + np.random.normal(0, 0.25)
        maxL_params_list[i, 0] = maxL_b

        # update the covariace matrix for each 10th Gibbs step.
        #if (i+1)%10 == 0:
        #    cov = np.cov(maxL_params_list[:i, 1:].T)
        #    cov_b = np.std(maxL_params_list[:i, 0])
        #    print(cov, cov_b)
        t3 = time.time()
        print('Gibbs sample iteration time: {}s'.format(t3-t2))
    # end Gibbs loop
    t4 = time.time()
    print('*** Sampling time: {}s, {}min'.format(t4-t1, (t4-t1)/60.))

    ind = list(index.values())[0]
    ind.sort()
    for i in ind:
        params_array[:,i:,:] = params_array[:, i-1:-1,:]
        params_array[:,i,:] = 0
    print(params_array[0,:,1])
    print(nu)
    res.print_results(nu, data, params_array, Gibbs_steps, Npix)
    #res.plot_model(Gibbs_steps, nu, data, model0, model, params, std_p)

    pass


#####  Global/input parameters  #####
Nside = 1
nu_array = np.array([30.,60.,90.,100.,200.,300.,400.,500.,600.,700.,800.,900.])
nu_ref = np.array([30., 44., 70., 100., 143., 217., 353., 545., 857.])
params = np.array([20., 20., 1.])

data_err = np.array([0.2, 5., 0.3, 4., .5, 0.2])
mean = np.array([1., 10., 1., 10., 1., -1.])
data_mean = np.array([3., 25., 1.5, 12., 1., -3.]) # expectations

path = 'Data/'
pfiles = np.array(['Data/LFI_SkyMap_030-BPassCorrected-field-IQU_1024_R3.00_full.fits',\
            'Data/LFI_SkyMap_044-BPassCorrected-field-IQU_1024_R3.00_full.fits',\
            'Data/LFI_SkyMap_070-BPassCorrected-field-IQU_1024_R3.00_full.fits',\
            'Data/HFI_SkyMap_100-field-IQU_2048_R3.00_full.fits',\
            'Data/HFI_SkyMap_143-field-IQU_2048_R3.00_full.fits',\
            'Data/HFI_SkyMap_217-field-IQU_2048_R3.00_full.fits',\
            'Data/HFI_SkyMap_353-psb-field-IQU_2048_R3.00_full.fits',\
            'Data/HFI_SkyMap_545-field-Int_2048_R3.00_full.fits',\
            'Data/HFI_SkyMap_857-field-Int_2048_R3.00_full.fits'])


###############################
#        Function call        #
###############################

main(Nside, 10, pfiles, nu_ref, mean, data_err, data_mean)
