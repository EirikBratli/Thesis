"""
Sampler to create dust intensity map
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import sys, time
import h5py

from functools import partial

np.random.seed(1189)
#np.random.seed(11095)
#np.random.seed(249)

def Noise(sigma, shape):
    """
    Make the noise map, with variables frequency 'nu' for a map with N pixels.

    Parameters:
    -----------
    sigma, array/scalar.    uncertainty of the noise
    shape, scalar/tupel.    shape of the noise

    Return:
    -----------
    n, array. The noise map.
    """

    a = 1e-2   # need a more proper value?
    #sigma = 10 #nu*a
    n = np.random.normal(0, sigma, shape)
    return(n)

#n = Noise(Npix, 10)
#hp.mollview(n)

def Model_Intensity(nu, params=None, b=3., T=25., beta_d=1.5, A_cmb=12., A=10.,\
                    A_s=0.1, beta_s=-2., sigma=10., shape=None):
    #, nu, A=0, T=0, beta=0, nu_d=353):
    """
    Make the intensity model 'I_m = I_d + n' for a given frequency 'nu'.
    Intensity given as a modified blackbody spectrum. Make a case where Return
    an array and a case returning a scalar.

    Parameters:
    -----------
    - nu, array.        The frequencies to iterate over
    - params, array.    Parameters of the MBB, containing (A, T, beta) in the
                        given order. Can have first dimension
    - sigma, scalar.    The uncertainty of the map used in the noise part.
    - shape,            The shape of the noise array to construct. Must reflect
                        the first axis dimesion of params
    Return:
    -----------
    I_model, array. The simulated intensity maps for the different frequencies
    """

    #print(params)
    if shape is None:
        shape = len(nu)

    if params is None:
        I_dust = MBB(nu, b, T, beta_d, A)
        I_cmb = I_CMB(nu, A_cmb)
        I_s = I_sync(nu, A_s, beta_s)
    elif len(params) > 1:
        # sample over T, beta, A_cmb
        #print('T and beta')
        T = params[0]
        beta_d = params[1]
        A_cmb = params[2]
        A_s = params[3]
        beta_s = params[4]
        I_dust = MBB(nu, T=T, beta=beta_d)#MMB(nu, A, T, beta)
        I_cmb = I_CMB(nu, A_cmb)
        I_s = I_sync(nu, A_s, beta_s)
    else: # sample over b
        #print('b')
        I_dust = MBB(nu, b=params[0])#MMB(nu, A, T, beta)
        I_cmb = I_CMB(nu, A_cmb)
        I_s = I_sync(nu, A_s, beta_s)

    n = Noise(sigma, shape)
    #print(b, T, beta, A_cmb)
    I_model = I_dust + I_cmb + I_s + n
    return(I_model)

def Data_Intensity(nu, b=3., A_d=10., T=25., beta_d=1.5, A_cmb=12., A_s=0.1,\
                    beta_s=-2., shape=None):
    #, nu, A=0, T=0, beta=0, nu_d=353):
    """
    Make the intensity data in 'I_m = I_d + n' for a given frequency 'nu'.
    Intensity given as a modified blackbody spectrum.

    Parameters:
    -----------
    - nu, array.        The frequencies to iterate over
    - A, scalar.        The amplitude of the intensity, default is 30.
    - T, scalar.        The temperature of the radiation, default is 25.
    - beta, scalar.     The spectral index of the modified blackbody,
                        default is 1.5
    - shape, None/tupel/scalar. The shape of the parameters.

    Return:
    -----------
    I_model, array. The simulated intensity maps for the different frequencies
    """

    #A = 30.#np.random.normal(30, sigma[0], shape)
    #T = 25.#np.random.normal(25, sigma[1], shape)
    #beta = 1.5#np.random.normal(1.5, sigma[2], shape)

    I_dust = MBB(nu, b, T, beta_d, A_d)
    I_cmb = I_CMB(nu, A_cmb)
    I_s = I_sync(nu, A_s, beta_s)
    #print(b, T, beta_d, A_cmb, A_s, beta_s)
    #print(I_dust[0] + I_cmb[0] + I_s[0], I_dust[0], I_cmb[0], I_s[0])
    return(I_dust + I_cmb + I_s)

def I_CMB(nu, A_cmb=12., Tcmb=2.7255, nu0=100.):
    """
    Calculate the intensity of the CMB. Sample the amplitude

    Parameters:
    -----------
    Return:
    -----------
    """
    #print(nu)
    h = 6.62607004e-34  # m^2 kg / s
    kB = 1.38064852e-23 # m^2 kg s^-2 K^-1
    x = h*nu*1e9/(kB*Tcmb)
    x0 = h*nu0*1e9/(kB*Tcmb)
    norm = (np.exp(x0) - 1.)**2 / (x0**2*np.exp(x0))

    I = A_cmb * (x**2*np.exp(x)) / ((np.exp(x) - 1.)**2)
    #print(I*norm, norm)
    return(I * norm)

def I_sync(nu, A_s=0.1, beta_s=-2., nu_0=408.):
    """
    Calculate the intensity of synchrotron radiation, sampling the amplitude and
    power, nu_0 is set to the nu_0 value in table 4, Plack Collaboration 2015 X.
    """

    return(A_s*(nu/nu_0)**beta_s)


def MBB(nu, b=3., T=25., beta=1.5, A=10., nu_d=353.):
    """
    Make the modified Planck spectrum of eq.1 in Planck Collaboration 2013 XII.

    Parameters:
    -----------
    nu, scalar.     Frequency in GHz
    b, scalar.      scale factor from A_cloud to A_dust
    T, scalar.      Brightness temperature in K
    beta, scalar.   spectral Index
    A, scalar.      The amplitude of the clouds, default=10.
    nu_d, scalar    The filter frequency to evaluate at, default is 353 GHz.

    Return:
    -----------
    B, scalar.      The intensity of the modified blackbody
    """
    h = 6.62607004e-34  # m^2 kg / s
    kB = 1.38064852e-23 # m^2 kg s^-2 K^-1
    c = 299792458.       # m / s
    factor = h*1e9/kB
    #print(b, T, beta)
    freq = (nu/nu_d)**(beta+1.)         # shape of nu and beta
    expo1 = np.exp(factor*nu_d/T) - 1.  # shape of T
    expo2 = np.exp(factor*nu / T) - 1.  # shape of nu and T

    B = b*A*freq*expo1/expo2              # shape of A and nu
    return(B)

def run_sampler(Nside, Gibbs_steps, nu, sigma=10.):
    """
    Function to run the sampler algorithm. Read in the extinction data
    Parameters:
    -----------
    - Nside, integer.   The resolution of the map to simulate.
    - nu, array.        Array with the frequencies to evaluate at, should be
                        something similar to the frequencies of Planck.
    - sigma, scalar.    The error of the noise, used in defining the noise in
                        data and in the likelihood estimation. Default is 10.

    Return:
    -----------
    - None
    """
    print('=========================')
    print('Sampling intensity parameters, A, T, beta of thermal dust.')

    # read in amplitude data from (Panopoulou):
    path = 'Data/Eirik/'
    #pix_num = np.load(path+'nside64_index.npy')
    #cloud_amp = np.load(path+'NH_per_cloud.npy')
    #print(pix_num, len(pix_num))
    #print(cloud_amp, len(cloud_amp))
    #print(12*64**2)
    #plt.plot(pix_num, cloud_amp)
    #plt.show()
    #ys.exit()
    print(nu)

    Npix = hp.pixelfunc.nside2npix(Nside)
    print('Map with Nside: {} and Npix: {}'.format(Nside, Npix))

    # initial guess on mean values: b, T, beta_d, A_cmb, A_sync, Beta_sync
    mean = np.array([1., 10., 1., 10., 1., -1.]) # : initial guess
    mean_b = mean[:1]
    print(mean_b)
    x1_mean = mean[1:]

    #data_err = np.array([0.1, 5., 0.3])
    data_err = np.array([0.2, 5., 0.3, 5., 1.5, 0.3]) # initial guess on errors.
    err_b = np.array([0.2])

    cov = Cov(len(x1_mean))
    cov_b = Cov(len(mean_b))

    Npix = 1
    sigma = 10.
    params_array = np.zeros((Gibbs_steps, Npix, len(mean)))
    b_param = np.zeros((Gibbs_steps, Npix))
    print('==========================')
    t0 = time.time()

    # fix data and a "solution" model with the "solution" parameters
    data_mean = np.array([3., 25., 1.5, 12., 0.1, -2.]) # "solution"
    data = Model_Intensity(nu, b=3., T=25., beta_d=1.5, A_cmb=12., A_s=0.1,\
                            beta_s=-2., sigma=sigma)
    model = Data_Intensity(nu, b=3., T=25., beta_d=1.5, A_cmb=12., A_s=0.1,\
                            beta_s=-2.)

    loglike = logLikelihood(data, model, 10.)
    print('=> Model logLikelihood = {}'.format(loglike))
    loglikes = []
    #loglike_params = []
    maxL_params_list = np.zeros((Gibbs_steps, len(mean)))
    for i in range(Gibbs_steps):
        print(' ')
        print('Gibbs step: {}'.format(i))
        t2 = time.time()

        #data_mean = np.array([3., 25., 1.5, 12.]) # "solution"
        #data = Model_Intensity(nu, b=3., T=25., beta=1.5, A_cmb=1., sigma=sigma)
        #model = Data_Intensity(nu, b=3., T=25., beta=1.5, A_cmb=1.)

        print('Calculate "T, beta, A_cmb" per pixel, given "b"')
        for pix in range(Npix):
            # set up function to send into Metropolis algorithm
            log_like = partial(logLikelihood, data=data)
            log_prior = partial(logPrior, mu=data_mean[1:], sigma=data_err[1:])
            #Model_func = partial(Model_Intensity, b=mean_b)  # right function???
            Model_func = partial(Data_Intensity, b=mean_b)
            # Initialize the parameters, model, etc.
            params0, model0, loglike0, logprior0 = Initialize(nu, log_like,\
                                                        log_prior, Model_func,\
                                                        x1_mean, cov, mean_b)
            # test initial values, if init log likelihood is less than -1e4
            # make new initial values, cause bad parameters.
            ll0 = loglike0
            while loglike0 < -1e4:
                print('hei')
                params0, model0, loglike0, logprior0 = Initialize(nu, log_like,\
                                                        log_prior, Model_func,\
                                                        x1_mean, cov, mean_b)
                #

            # Sample parameters:
            params, par_maxL = MetropolisHastings(nu, log_like, log_prior,\
                                    Model_func, sigma, params0, model0,\
                                    loglike0, logprior0, x1_mean, cov,\
                                    len(x1_mean), mean_b)
            params_array[i, pix,1:] = params
            print(par_maxL)

        # end pixel loop
        print('Calculate "b" given "T, beta"')
        # set up functions to use in MetropolisHastings using the previous params
        maxL_params_list[i, 1:] = par_maxL
        x1_mean = par_maxL + np.random.normal(np.zeros(len(par_maxL)),\
                                            np.fabs(par_maxL)/30.)
        # last max like params?not mean
        log_like = partial(logLikelihood, data=data)
        log_prior = partial(logPrior, mu=data_mean[:1], sigma=data_err[:1])
        #Model_func = partial(Model_Intensity, T=params[0], beta_d=params[1],\
        #                    A_cmb=params[2], A_s=params[3], beta_s=params[4])
        Model_func = partial(Data_Intensity, T=params[0], beta_d=params[1],\
                            A_cmb=params[2], A_s=params[3], beta_s=params[4])

        # Initialize
        params0, model0, loglike0, logprior0 = Initialize(nu, log_like,\
                                                    log_prior, Model_func,\
                                                    mean_b, cov_b, x1_mean)
        #
        # test initial values
        while loglike0 < -1e4:
            # if loglike0 < -1e4, make new initial values
            params0, model0, loglike0, logprior0 = Initialize(nu, log_like,\
                                                        log_prior, Model_func,\
                                                        mean_b, cov_b, x1_mean)
            #
        # Sample b:
        b, maxL_b = MetropolisHastings(nu, log_like, log_prior, Model_func,\
                                sigma, params0, model0, loglike0, logprior0,\
                                mean_b, cov_b, len(mean_b), params)

        b_param[i,:] = b
        mean_b = maxL_b + np.random.normal(0, 0.25)#b
        maxL_params_list[i, 0] = maxL_b

        # Update the covariace matrix for each 10th gibb step
        if (i+1)%10 == 0:
            #mean_params = np.mean(params_array[:i,:,1:], axis=1)
            #cov = np.cov(mean_params[:,:].T)#Cov(len(x1_mean))
            #cov_b = np.std(b_param[:i,:])#np.cov(b_param[:i,:].T)

            cov = np.cov(maxL_params_list[:i,1:].T)
            cov_b = np.std(maxL_params_list[:i, 0])
            print(cov, cov_b)

        #sys.exit()
        t3 = time.time()
        print('Gibbs sample iteration time: {}s'.format(t3-t2))
        #print('--- --- ---')
        temp = Data_Intensity(nu, b=maxL_b, T=par_maxL[0], beta_d=par_maxL[1],\
                        A_cmb=par_maxL[2], A_s=par_maxL[3], beta_s=par_maxL[4])
        print(logLikelihood(data, temp, 10.))
        loglikes.append(logLikelihood(data, temp, 10.))

    # end gibbs loop
    t1 = time.time()

    print('------------------')
    print('Time used: {}.s'.format(t1-t0))
    print(max(loglikes), np.where(loglikes == max(loglikes))[0])
    maxloglike = maxL_params_list[np.where(loglikes == max(loglikes))[0][0], :]
    print(maxloglike, min(loglikes))
    params_array[:,:,0] = b_param
    #print(params_array[-1,:,:])
    #print(np.mean(params_array, axis=0))
    #print(np.median(params_array, axis=0))

    mean_params = np.mean(params_array, axis=0)
    b_array = np.reshape(b_param, Gibbs_steps*Npix)
    T_array = np.reshape(params_array[:,:,1], Gibbs_steps*Npix)
    beta_dust_array = np.reshape(params_array[:,:,2], Gibbs_steps*Npix)
    A_cmb_array = np.reshape(params_array[:,:,3], Gibbs_steps*Npix)
    A_sync_array = np.reshape(params_array[:,:,4], Gibbs_steps*Npix)
    beta_sync_array = np.reshape(params_array[:,:,5], Gibbs_steps*Npix)

    b = np.mean(b_array)
    T = np.mean(T_array)
    beta_dust = np.mean(beta_dust_array)
    A_cmb = np.mean(A_cmb_array)
    A_sync = np.mean(A_sync_array)
    beta_sync = np.mean(beta_sync_array)

    print('Mean b: {}'.format(b))
    print('Mean T_d: {}'.format(T))
    print('Mean beta_dust: {}'.format(beta_dust))
    print('Mean A_cmb: {}'.format(np.mean(A_cmb)))
    print('Mean A_sync: {}'.format(np.mean(A_sync)))
    print('Mean beta_sync: {}'.format(np.mean(beta_sync)))
    #"""
    modelnew = Data_Intensity(nu, b=b, T=T, beta_d=beta_dust, A_cmb=A_cmb,\
                                A_s=A_sync, beta_s=beta_sync)

    maxL_model = Data_Intensity(nu, b=maxloglike[0], T=maxloglike[1],\
                    beta_d=maxloglike[2], A_cmb=maxloglike[3], A_s=maxloglike[4],\
                    beta_s=maxloglike[5])

    loglike_sim = logLikelihood(data, modelnew, 10.)
    print('Simulated loglikelihood: {}'.format(loglike_sim))
    print('=> Model logLikelihood = {}'.format(loglike))

    plt.figure('Intensity')
    plt.plot(nu, data, 'xk', label='data')
    plt.plot(nu, model, '--b', label='origin model')
    plt.plot(nu, modelnew, '-b', label='simulated model')
    plt.plot(nu, maxL_model, 'purple', label='max loglike model')
    #plt.plot(nu, MBB(nu, b, T, beta_dust),'-r', label='MBB(b,T,beta)')
    #plt.plot(nu, MBB(nu), '--r',label='dust')
    #plt.plot(nu, I_CMB(nu), '--c', label='I_cmb')
    #plt.plot(nu, I_CMB(nu, A_cmb), '-c', label='I_cmb, A_cmb=mean')
    #plt.loglog(nu, I_sync(nu), '--g', label='sync')
    #plt.plot(nu, I_sync(nu, A_sync, beta_sync), '-g', label='I_sync, mean params')
    plt.errorbar(nu, data, yerr=sigma, ecolor='grey', ls='none')
    #mean = np.array([1., 10., 1., 10., 1., -1.])

    #plt.ylim(0.01,500)
    plt.xlabel('freq')
    plt.ylabel('intensity')
    plt.legend(loc='best')
    plt.savefig('Figures/Sampling_test/components_of_freq{}.png'.\
                format(Gibbs_steps))
    #"""
    #"""
    plt.figure('b')
    #plt.hist(mean_params[:,0], bins=30, color='b')
    plt.hist(b_array, bins=30, color='b')
    plt.axvline(x=3, linestyle=':', color='k')

    plt.figure('Temp')
    #plt.hist(mean_params[:,1], bins=30, color='r')
    plt.hist(T_array, bins=30, color='r')
    plt.axvline(x=25, linestyle=':', color='k')

    plt.figure('beta')
    #plt.hist(mean_params[:,2], bins=30, color='g')
    plt.hist(beta_dust_array, bins=30, color='g')
    plt.axvline(x=1.5, linestyle=':', color='k')

    plt.figure('A_cmb')
    #plt.hist(mean_params[:,2], bins=30, color='g')
    plt.hist(A_cmb_array, bins=30, color='m')
    plt.axvline(x=12, linestyle=':', color='k')

    plt.figure('A_sync')
    #plt.hist(mean_params[:,2], bins=30, color='g')
    plt.hist(A_sync_array, bins=30, color='y')
    plt.axvline(x=0.1, linestyle=':', color='k')

    plt.figure('beta_sync')
    #plt.hist(mean_params[:,2], bins=30, color='g')
    plt.hist(beta_sync_array, bins=30, color='c')
    plt.axvline(x=-2, linestyle=':', color='k')
    #"""
    plt.show()


def Initialize(nu, log_like, log_prior, Model_func, mean, cov, const):
    """
    Initialize the parameters, model, likelihood and prior for the sampling.
    Check also for negative parameters, not acceptable, use new mean for those
    parameters given as mu = mean_i - params_i

    Parameters:
    -----------

    Return:
    - curr_params, array.
    - curr_model, array.
    - curr_like, scalar.
    - curr_prior, scalar.
    """

    curr_params = proposal_rule(cov, mean)
    print(mean)
    print(curr_params)
    #print('-',const)
    if len(mean) > 1:
        c = len(curr_params)-1
        for i in range(len(curr_params)-1):
            if curr_params[i] < 0:
                c -= 1
                mu = mean[i] - curr_params[i]
                print(mu)
                curr_params[i] = np.random.normal(mu, np.sqrt(cov[i,i]))

        print(c, curr_params)
    else:
        if curr_params < 0:
            curr_params = proposal_rule(cov, mean)

    # make a model from the parameters
    if len(curr_params) == 1:
        curr_model = Model_func(nu, b=curr_params[0], T=const[0],\
                                beta_d=const[1], A_cmb=const[2],\
                                A_s=const[3], beta_s=const[4])
    else:
        curr_model = Model_func(nu, b=const[0], T=curr_params[0],\
                                beta_d=curr_params[1], A_cmb=curr_params[2],\
                                A_s=curr_params[3], beta_s=curr_params[4])
        #Model_Intensity(nu, curr_params)
    curr_like = log_like(curr_model)
    curr_prior = log_prior(curr_params)

    print(curr_params, curr_like, curr_prior)
    print(curr_model)
    print('----')
    return(curr_params, curr_model, curr_like, curr_prior)

def MetropolisHastings(nu, log_like, log_prior, Model_func, sigma, curr_params,\
                        curr_model, curr_like, curr_prior, mean, cov, Nparams,\
                        const, Niter=1000):
    """
    Do the samlping loop of the samling.
    Parameters:
    -----------
    Return:
    -----------
    """
    accept = np.zeros(Niter)
    params = np.zeros((Niter, Nparams))
    counter = 0
    steplength = 1.
    max_like = -50
    #print(mean)
    # initialize
    #curr_params, curr_model, curr_like, curr_prior = Initialize(nu, log_like,\
    #                                                    log_prior, mean, cov)


    #print(curr_params)
    #print((curr_model))
    #print(curr_like)
    #print(curr_prior)
    params_max_like = curr_params
    # sampling
    #print('-----')
    for i in range(Niter):
        #model = Model_Intensity(nu, curr_params)
        prop_params = proposal_rule(cov*steplength, mean)
        #print(prop_params, mean)
        accept[i], curr_params, max_like, params_max_like = mh_step(log_like,\
                                        log_prior, Model_func, prop_params, nu,\
                                        curr_like, curr_prior, curr_params,\
                                        max_like, params_max_like, const)

        #print(np.shape(params[i,:]), np.shape(curr_params))
        params[i,:] = curr_params
        #print(max_like, params_max_like)
        # update current likelihood and prior:
        if Nparams == 1:
            curr_model_new = Model_func(nu, b=curr_params[0], T=const[0],\
                                    beta_d=const[1], A_cmb=const[2],\
                                    A_s=const[3], beta_s=const[4])
        else:
            #if (i)%10 == 0:
            #    #print('--', curr_model[0])
            #    #print(accept[i], curr_like, curr_prior, curr_params)

            curr_model_new = Model_func(nu, b=const[0], T=curr_params[0],\
                                    beta_d=curr_params[1],A_cmb=curr_params[2],\
                                    A_s=curr_params[3], beta_s=curr_params[4])


        curr_like = log_like(curr_model_new)
        curr_prior = log_prior(curr_params)
        mean = curr_params
        #print('-', curr_like)
        # update the steplength in the cov-matrix
        if accept[i] == True:
            counter += 1


        if (i+1)%50==0:
            if counter/float(i+1) < 0.2:
                steplength /= 2.
            elif counter/float(i+1) > 0.5:
                steplength *= 2.
            else:
                pass
            #

        #if (i+1)%300==0:
        #    cov = np.cov(params[:i,:].T)

        if (i)%100 == 0:
            #if (i)%100 == 0:
            print(i, curr_like, curr_prior, curr_params)
        #

    #
    print(counter, counter/float(Niter), max_like)
    #print('Maximum loglikelihood fit: {}'.format(max_like))
    #print(params[-1,:], curr_params)
    #params = np.mean(params, axis=0)
    """
    plt.figure('.')
    plt.plot(params)
    plt.figure('1')
    plt.hist(params[:,0], bins=100)
    plt.figure('2')
    plt.hist(params[:,1], bins=100)
    plt.figure('3')
    plt.hist(params[:,2], bins=100)
    plt.show()
    #"""

    return(curr_params, params_max_like)
    #

def mh_step(log_like, log_prior, Model_func, prop_params, nu, curr_like,\
            curr_prior, curr_params, max_like, params_max_like, const):
    """
    Do the MH algorithm steps, with acceptance and stuff.
    Parameters:
    -----------
    - log_like, function.       function to calculate the log likelihood of the
                                model given the data. Takes in an assumed model.
    - log_prior, function.      Function to calculate the logarithm of the prior
                                by the drawn parameters. Takes in the assumed
                                parameters.
    - Model_func, function.     Function to calculate an assumed model using the
                                drawn parameters.
    - prop_params, array.       The proposed parameters
    -
    -
    -

    Return:
    -----------
    """
    # proposal
    #prop_params = proposal
    #print('lol', const, prop_params)
    if len(prop_params) == 1:
        prop_model = Model_func(nu, b=prop_params[0], T=const[0],\
                                beta_d=const[1], A_cmb=const[2], A_s=const[3],\
                                beta_s=const[4])
    else:
        prop_model = Model_func(nu, b=const[0], T=prop_params[0],\
                                beta_d=prop_params[1], A_cmb=prop_params[2],\
                                A_s=prop_params[3], beta_s=prop_params[4])

    prop_like = log_like(prop_model) #(prop_params, model)
    prop_prior = log_prior(prop_params) #(prop_params)

    # posterior:
    post_old = curr_like + curr_prior
    post_new = prop_like + prop_prior

    # acceptance testing
    a = np.exp(post_new - post_old)
    #print(a, prop_params, prop_like, curr_like)
    draw = np.random.uniform(0, 1)
    if (a > draw) and (a < np.inf):
        accept = True
        curr_params = prop_params
        #curr_like = prop_like
        if prop_like > max_like:
            max_like = prop_like
            params_max_like = curr_params
            #print(prop_like, curr_like, max_like, prop_prior)

    else:
        accept = False
        curr_params = curr_params

    #print(prop_like)
    #print('-',a, draw, post_old, post_new)

    # return statement
    return(accept, curr_params, max_like, params_max_like)


def proposal_rule(cov, mean=None):
    """
    Draw new parameters for proposal.
    """

    if (len(mean) >= 2) and (np.ndim(cov) >= 2):
        params = np.random.multivariate_normal(mean, cov)
    else:
        params = np.random.normal(mean, cov)
    return(params)

def logLikelihood(model, data, sigma=10.): # data should be optional
    """
    Compute the log likelihood of the data, P(d|m) for each sight line

    Parameters:
    -----------
    - model, array.             Contains the f(x) points
    - data, array, optional.    Contains the data points
    - sigma, scalar, optional.  The uncertainty of the data/model

    Return:
    -----------
    - L, scalar.        The log likelihood of the data fitting the model
    """
    L = 0
    #print(len(data), len(model))
    #print(data)
    #print(model)
    for i in range(len(data)):
        L += -0.5*((data[i] - model[i])/sigma)**2
        #L -= (np.log(sigma) + 0.5*np.log(2*np.pi))
    #print(L)
    return(L)

def logPrior(params, mu=None, sigma=None):# mu='data_mean', sigma='data_err'):
    """
    Compute the prior, p(m). The parameters must be positive

    Parameters:
    -----------
    - params, array.            Array with the parameters
    - mu, array, optional.      Array with the mean parameter values,
    - sigma, array, optional.   The uncertainties of the parameters

    Return:
    -----------
    - ln(P(model)), scalar. The logarithm of the prior value
    """

    pm = 0.
    c = 0
    #print(params, mu, sigma)

    for i in range(len(mu)):
        pm += -0.5*((params[i] - mu[i])/sigma[i])**2
        #pm -= (np.log(sigma[i]) + 0.5*np.log(2*np.pi))
        #print(params[i], (params[i] - mu[i])/sigma[i])

        if (params[i] <= 0) and (i < len(params)-1):
            c += 1
            #if i == len(params)-1:
            #    c = c
    #if len(params) > 1:
    #    if params[1] < 0:
    #        pm = -50
    #print(c, len(params))
    if c > 0:
        pm = -50
    else:
        pm = pm
    return(pm)

def sampler(data, data_mean, data_err, mean, cov, nu, sigma,\
            Niter=10000):
    """
    Sampler function to fit the parameters, {A, T, beta}

    Parameters:
    -----------
    - data, array.      An array contain the data points for a sight line
    - data_mean, array. list of the mean values of the parameters (goal params)
    - data_err, array.  List of the used errors for each parameter, used in prior
    - mean, array.      The initial mean values of the parameters.
    - cov, ndarray.     The covariace matric of the parameters.
    - nu, array.        The frequency bands.
    - Niter, integer.   The number of sampling iterations to preform.
    - sigma, scalar.    The uncertainty of the noise.
    Return:
    -----------
    args, array.        The most likely parameter values that describe the data.
    sigma_args, array.  The standard deviation of the mean parameter values
    par_maxL, array.    The maximum likelihood parameters.
    """

    Nparams = len(mean)
    accept = np.zeros((Niter, len(data_mean)))

    # initialize:
    curr_params = np.random.multivariate_normal(mean, cov) # A, T, beta
    curr_model = Model_Intensity(nu, curr_params)
    pd_old = logLikelihood(data, curr_model, sigma)
    pm_old = logPrior(curr_params, data_mean, np.asarray(data_err))
    posterior_curr = pd_old + pm_old

    # sampling:
    max_like = -20
    params_max_like = mean
    steplength = 1.
    counter = 0
    t0 = time.time()
    for i in range(Niter-1):

        # draw parameters and contruct proposed posterior
        prop_params = np.random.multivariate_normal(mean, cov)
        model_prop = Model_Intensity(nu, prop_params)
        like = logLikelihood(data, model_prop, sigma)
        posterior_prop = like + logPrior(prop_params, data_mean,\
                                         np.asarray(data_err))

        # check for acceptance:
        posterior_prev = pd_old + pm_old
        a = np.exp(posterior_prop - posterior_prev)
        draw = np.random.uniform(0,1)

        if (a > draw) and (a < np.inf):

            if prop_params.any() < 0:
                print(i, 'something wrong', prop_params)
            accept[i,:] = prop_params
            curr_params = prop_params
            counter += 1
            if like > max_like:
                max_like = like
                params_max_like = accept[i,:]
        else:
            accept[i,:] = curr_params
            curr_params = curr_params
        #
        pd_old = logLikelihood(data, Model_Intensity(nu, curr_params), sigma)
        pm_old = logPrior(curr_params, data_mean, np.asarray(data_err))
        mean = accept[i,:]

        # update the steplength in the cov-matrix
        if (i+1)%100==0:
            if counter/float(i+1) < 0.2:
                steplength /= 2.
            elif counter/float(i+1) > 0.5:
                steplength *= 2.
            else:
                pass
            #
            cov = steplength*cov
        if (i+1)%3000==0:
            cov = np.cov(accept[:i,:].T)

        #if (i)%100==0:
        #    print(i, params_max_like, max_like)


    #
    t1 = time.time()
    #print('-----', counter, counter/float(Niter), '------')
    #print('Sampling time: {}s'.format(t1-t0))
    #print(params_max_like, max_like)
    burnin = 3000
    args = np.mean(accept[burnin:,:], axis=0)
    sigma_args = np.std(accept[burnin:,:], axis=0)
    #print(args)
    #print(sigma_args)
    """
    plt.figure('A')
    plt.hist(accept[:,0], bins=100, color='b')

    plt.figure('T')
    plt.hist(accept[:,1], bins=100, color='r')

    plt.figure('beta')
    plt.hist(accept[:,2], bins=100, color='g')
    """
    return(args, sigma_args, params_max_like)



def Cov(N):
    # N is number of parameters
    if N > 1:
        C = np.eye(N)
        for i in range(N):
            for j in range(N):
                if i != j:
                    C[i,j] = 0.81
        #
    else:
        C = 0.81
    return(C)


def func(data):
    """
    Give the model to fit to the data. Could be any function like power law,
    step function, etc...

    Parameters:
    -----------
    Return:
    -----------
    """
    model = np.zeros(len(data))
    model[0] = 0.23 # half of gaia global A_g uncertainty
    for i in range(1,len(data)):
        if data[i] < 2 * model[i-1]:
            model[i] = model[i-1]
        else:
            model[i] = model[i-1] + data[i]
    return(model)

def Read_H5(file, name):
    """
    Read a .h5 file and return the data array

    Parameters:
    -----------
    - file, string. File name of the file to read.
    - name, string. Column name of the data

    Return:
    -----------
    - data, array. The array with data
    """
    f = h5py.File(file, 'r')
    data = np.asarray(f[name])
    f.close()
    return(data)


###################################
#         Global variables        #
###################################

Nside = 1
#Npix = hp.pixelfunc.nside2npix(Nside)
nu_array = np.array([30.,60.,90.,100.,200.,300.,400.,500.,600.,700.,800.,900.])
nu_d = np.array([30., 44., 70., 100., 143., 217., 353., 545., 857.])
params = np.array([20., 20., 1.])

####################################
#           Run code:              #
#         Function calls:          #
####################################

#Id = Data_Intensity(nu)
#Im = Model_Intensity(nu, params)
run_sampler(Nside, 100, nu_array)
#hp.mollview(Im + n)
#plt.show()
