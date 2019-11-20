"""
Sampler to create dust intensity map
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import sys, time
import h5py




def Noise(Npix, sigma):
    """
    Make the noise map, with variables frequency 'nu' for a map with N pixels.

    Parameters:
    -----------
    Npix, integer. Number of pixels in the map.
    sigma, array.

    Return:
    -----------
    n, array. The noise map.
    """

    a = 1e-2   # need a more proper value?
    #sigma = 10 #nu*a
    n = np.random.normal(0, sigma, Npix)
    return(n)

#n = Noise(Npix, 10)
#hp.mollview(n)

def Model_Intensity(Npix, nu, params, sigma=10.):#, nu, A=0, T=0, beta=0, nu_d=353):
    """
    Make the intensity model 'I_m = I_d + n' for a given frequency 'nu'.
    Intensity given as a modified blackbody spectrum.

    Parameters:
    -----------
    - Npix, integer.    The number of pixels in the map to create
    - nu, array.        The frequencies to iterate over
    - sigma, scalar.    The uncertainty of the map used in the noise part.

    Return:
    -----------
    I_model, array. The simulated intensity maps for the different frequencies
    """
    I_model = np.zeros((Npix, len(nu)))
    I_dust = np.zeros((Npix, len(nu)))#, len(nu_d)))
    beta = np.zeros((Npix, len(nu)))#, len(nu_d)))
    T = np.zeros((Npix, len(nu)))#, len(nu_d)))
    A = np.zeros((Npix, len(nu)))#, len(nu_d)))

    for i in range(len(nu)):
        A = params[0]      #30.#np.random.normal(30, 5, Npix)
        T = params[1]      #np.random.normal(25, 5)
        beta = params[2]   #np.random.normal(1.5, 0.3)

        I_dust[:,i] = MMB(nu[i], A, T, beta)

        n = Noise(Npix, sigma)

        I_model[:,i] = I_dust[:,i] + n
        #print(nu[i], np.mean(I_model[:,i]), np.max(n), np.min(n))

        #hp.mollview(I_dust[:,i])
        #hp.mollview(I_model[:,i], title=r'$I_m=I_d+n$, for $\nu={}$'.\
        #            format(nu[i]), unit=r'$\mu K_{{RJ}}$')

    #plt.plot(nu, (I_dust[3000,:]))

    return(I_model)

def Data_Intensity(Npix, nu, sigma=10.):#, nu, A=0, T=0, beta=0, nu_d=353):
    """
    Make the intensity model 'I_m = I_d + n' for a given frequency 'nu'.
    Intensity given as a modified blackbody spectrum.

    Parameters:
    -----------
    - Npix, integer.    The number of pixels in the map to create
    - nu, array.        The frequencies to iterate over
    - sigma, scalar.    The uncertainty of the map used in the noise part.

    Return:
    -----------
    I_model, array. The simulated intensity maps for the different frequencies
    """
    I_model = np.zeros((Npix, len(nu)))
    I_dust = np.zeros((Npix, len(nu)))#, len(nu_d)))
    beta = np.zeros((Npix, len(nu)))#, len(nu_d)))
    T = np.zeros((Npix, len(nu)))#, len(nu_d)))
    A = np.zeros((Npix, len(nu)))#, len(nu_d)))

    for i in range(len(nu)):
        A[:,i] = np.random.normal(30, sigma[0], Npix)
        T[:,i] = np.random.normal(25, sigma[1], Npix)
        beta[:,i] = np.random.normal(1.5, sigma[2], Npix)

        I_dust[:,i] = MMB(nu[i], A[:,i], T[:,i], beta[:,i], )

    return(I_dust)

def MMB(nu, A, T, beta, nu_d=353):
    """
    Make the modified Planck spectrum of eq.1 in Planck Collaboration 2013 XII.

    Parameters:
    -----------
    nu, scalar.     Frequency in GHz
    T, scalar.      Brightness temperature in K
    beta, scalar.   spectral Index
    A, scalar.      The amplitude
    nu_d, scalar    The filter frequency to look at, default is 353 GHz.

    Return:
    -----------
    B, scalar.      The intensity of the modified blackbody
    """
    h = 6.62607004e-34  # m^2 kg / s
    kB = 1.38064852e-23 # m^2 kg s^-2 K^-1
    c = 299792458.       # m / s
    factor = h*1e9/kB

    freq = (nu/nu_d)**(beta+1.)
    expo1 = np.exp(factor*nu_d/T) - 1.
    expo2 = np.exp(factor*nu / T) - 1.

    B = A*freq*expo1/expo2
    return(B)

def run_sampler(Nside, nu, nu_ind=5):
    """
    Function to run the sampler algorithm. Read in the extinction data
    Parameters:
    -----------

    Return:
    -----------
    """

    #Ag = Read_H5(Ag_file, 'Ag')
    #R = Read_H5(R_file, 'R')
    data_err = [5.,5.,0.3]
    Npix = hp.pixelfunc.nside2npix(Nside)
    data = Data_Intensity(Npix, nu, data_err)
    mean_data = np.mean(data, axis=0)
    data_mean = [mean_data[nu_ind],25., 1.5]

    print(np.shape(data), data_mean)
    #print(data[0,:])
    hp.mollview(data[:,nu_ind])
    #plt.show()
    #sys.exit()
    mean = [10., 10., 1.] # initial guess on mean values
    cov = np.eye(len(mean))
    for i in range(len(mean)):
        for j in range(len(mean)):
            if j != i:
                cov[i,j] = 0.81
            #
        #
    #
    print(mean)
    print(cov)

    #params = sampler(mean, cov, nu, Npix)
    for i in range(Npix):
        print('Sampling for pixel {}'.format(i))
        params = sampler(Npix, i, data[i,nu_ind], data_mean, data_err, mean,\
                         cov, nu)
        #if i > 10:
        sys.exit()

def sampler(Npix, pix, data, data_mean, data_err, mean, cov, nu, nu_ind=5,\
            Niter=10000, sigma=5.):
    """
    Sampler function to fit the parameters, {A, T, beta}

    Parameters:
    -----------
    - data, array.      An array contain the data points for a sight line
    - x, array.         The distances to bins/clouds
    - mean, array.      The initial mean values of the parameters.
    - cov, ndarray.     The covariace matric of the parameters.
    - Niter, integer.   The number of sampling iterations to preform.

    Return:
    -----------
    accept, array.      The most likely parameter values that describe the data.
    """
    print('Sampling')

    Nparams = len(mean)
    posterior = np.zeros((Niter))
    params = np.zeros((Niter, Nparams))
    accept = np.zeros((Niter-1, Nparams))
    #sigma = 1 #???

    # initialize:
    curr_params = np.random.multivariate_normal(mean, cov) # A, T, beta
    model = Model_Intensity(Npix, nu, curr_params, sigma)[pix, nu_ind]
    pd_old = logLikelihood(data, model, sigma)
    pm_old = logPrior(curr_params, data_mean, data_err)
    posterior_curr = pd_old + pm_old
    #print(np.shape(curr_params))
    #print(np.shape(model), np.shape(data), np.shape(curr_params))
    #print(np.shape(pd_old), np.shape(pm_old))
    #print(pd_old, pm_old)
    #print('---')
    #sys.exit()
    # sampling:
    max_like = -50
    params_max_like = mean
    steplength = 1.
    counter = 0
    t0 = time.time()
    for i in range(Niter-1):

        # draw parameters and contruct proposed posterior
        prop_params = np.random.multivariate_normal(mean, cov)
        model_prop = Model_Intensity(Npix, nu, prop_params, sigma)[pix, nu_ind]
        posterior_prop = logLikelihood(data, model_prop, sigma)\
                         + logPrior(prop_params, data_mean, data_err)

        # check for acceptance:
        posterior_prev = pd_old + pm_old
        a = np.exp(posterior_prop - posterior_prev)
        #print(np.shape(a), np.shape(model_prop), np.shape(posterior_prev))
        draw = np.random.uniform(0,1)

        if (a > draw) and (a < np.inf):
            like = logLikelihood(data, model, sigma)
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
        pd_old = logLikelihood(data, Model_Intensity(Npix, nu, curr_params,\
                                sigma)[pix,nu_ind], sigma)
        pm_old = logPrior(curr_params, data_mean, data_err)

        mean = accept[i,:]
        if (i+1)%1000==0:
            print(i, posterior_prop - posterior_prev)
        #"""
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
        #"""
        if (i+1)%3000==0:
            #print(accept[i,:])
            cov = np.cov(accept[:i,:].T)
            #print(cov)
        #"""
    #
    t1 = time.time()
    print('-----', counter, counter/float(Niter), '------')
    print('Sampling time: {}s'.format(t1-t0))
    #print(accept)
    #print(len(np.where(accept <= 0)[0]))
    print(params_max_like, max_like)
    burnin = 3000
    args = np.mean(accept[burnin:,:], axis=0)
    sigma_args = np.std(accept[burnin:,:], axis=0)
    print(args)
    print(sigma_args)

    #for i in range(len(args)):
    plt.figure()
    plt.plot(accept[burnin:,:])

    plt.figure()
    plt.hist(accept[burnin:,:], bins=30)
    plt.show()
    return(args, sigma_args)

def logLikelihood(data, model, sigma):
    """
    Compute the log likelihood of the data, P(d|m) for each sight line

    Parameters:
    -----------
    - data, array.            Contains the data points
    - model, array.           Contains the f(x) points
    - sigma, array/scalar.    The uncertainty of the data/model

    Return:
    -----------
    - L, scalar.        The likelihood of the data fitting the model
    """
    L = 0
    #print(np.shape(data), np.shape(model))
    #for i in range(len(data)):
    L = -0.5*((data - model)/sigma)**2
    L -= (np.log(sigma) + 0.5*np.log(2*np.pi))
    return(L)

def logPrior(params, mu, sigma):
    """
    Compute the prior, p(m). The parameters must be positive

    Parameters:
    -----------
    - params, array.      Array with the parameters

    Return:
    -----------
    - ln(1)
    """
    #sigma = np.ones(len(mu))

    pm = 0.
    #print((params[0]))
    if params.all() > 0:
        for i in range(len(mu)):
            pm += -0.5*((params[i] - mu[i])**2/sigma[i]**2)
            pm -= (np.log(sigma[i]) + 0.5*np.log(2*np.pi))

        return(pm)
    else:
        return(-30.)

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

Nside = 16
#Npix = hp.pixelfunc.nside2npix(Nside)
nu = np.array([30., 60., 90., 100.,200.,300.,400.,500.,600.,700.,800.,900.])
nu_d = np.array([30., 44., 70., 100., 143., 217., 353., 545., 857.])

####################################
#           Run code:              #
#         Function calls:          #
####################################

#Im = Model_Intensity(Npix, nu, 10)
run_sampler(Nside, nu)
#hp.mollview(Im + n)
plt.show()
