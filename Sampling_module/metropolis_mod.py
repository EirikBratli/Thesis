"""
Module for the sampling using Metropolis Hasting algorithm, includes also the
Initialization function.
"""

import numpy as np
#import healpy as hp
#import matplotlib.pyplot as plt
import sys, time
#import h5py


def Initialize(nu, log_like, log_prior, Model_func, mean, cov, const):
    """
    Initialize the parameters, model, likelihood and prior for the sampling.
    Check also for negative parameters, not acceptable, use new mean for those
    parameters given as mu = mean_i - params_i

    Parameters:
    -----------
    - nu, array.                The frequencies
    - log_like, function.       function to calculate the log likelihood of the
                                model given the data. Takes in an assumed model.
    - log_prior, function.      Function to calculate the logarithm of the prior
                                by the drawn parameters. Takes in the assumed
                                parameters.
    - Model_func, function.     Function to calculate an assumed model using the
                                drawn and constant parameters.
    - mean, array.          The drawn parameters
    - cov, array.           The uncertainty of the parameters
    - const, array.         The parameters not drawn

    Return:
    - curr_params, array.
    - curr_model, array.
    - curr_like, scalar.
    - curr_prior, scalar.
    """

    curr_params = proposal_rule(cov, mean)
    #print('-',mean)
    #print('--', curr_params)
    # check for negative parameters.
    if len(mean) > 1:
        c = len(curr_params)-1
        for i in range(len(curr_params)-1):
            while curr_params[i] < 0:
                c -= 1
                mu = mean[i] - curr_params[i]
                print(i, c, curr_params[i])
                curr_params[i] = np.random.normal(mu, np.sqrt(cov[i,i]))


        print(c, curr_params)
    else:
        while curr_params < 0:
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
    #print(curr_model)
    print('--')
    return(curr_params, curr_model, curr_like, curr_prior)


def MetropolisHastings(nu, log_like, log_prior, Model_func, sigma, curr_params,\
                        curr_model, curr_like, curr_prior, mean, cov, Nparams,\
                        const, Niter=100):
    """
    Do the samlping loop of the samling.
    Parameters:
    -----------
    - nu, array.                The frequencies
    - log_like, function.       function to calculate the log likelihood of the
                                model given the data. Takes in an assumed model.
    - log_prior, function.      Function to calculate the logarithm of the prior
                                by the drawn parameters. Takes in the assumed
                                parameters.
    - Model_func, function.     Function to calculate an assumed model using the
                                drawn and constant parameters.
    - sigma, scalar.            The uncertainty to use in the log likelihood.
    - curr_params, array.       The initial parameters drawn.
    - curr_model, array.        The initial model using the drawn parameters.
    - curr_like, scalar.        The current likelihood of the initial sample.
    - curr_prior, scalar.       The current prior of the initial sample.
    - mean, array.              The previous parametes to draw from.
    - cov, array.               The covariance/uncertainty of the mean argument.
    - Nparams, integer.         The number of parameters sampling.
    - const, array.             The previous parameters not to draw from.

    Return:
    -----------
    """
    accept = np.zeros(Niter)
    params = np.zeros((Niter, Nparams))
    counter = 0
    steplength = 1.
    max_like = -50

    params_max_like = curr_params
    # sampling
    print('-----')
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

        # update current likelihood and prior:
        if Nparams == 1:
            curr_model_new = Model_func(nu, b=curr_params[0], T=const[0],\
                                    beta_d=const[1], A_cmb=const[2],\
                                    A_s=const[3], beta_s=const[4])
        else:
            curr_model_new = Model_func(nu, b=const[0], T=curr_params[0],\
                                    beta_d=curr_params[1],A_cmb=curr_params[2],\
                                    A_s=curr_params[3], beta_s=curr_params[4])


        curr_like = log_like(curr_model_new)
        curr_prior = log_prior(curr_params)
        mean = curr_params
        #print(i, curr_like, curr_prior, params_max_like)
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
        if (i)%20 == 0:
            print(i, max_like, curr_prior, params_max_like)
        #    print('-', curr_params, curr_like, curr_prior)



    #
    print(counter, counter/float(Niter), max_like, params_max_like)

    return(curr_params, params_max_like)


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
                                drawn and constant parameters.
    - prop_params, array.       The proposed parameters
    - nu, array.                The frequencies
    - curr_like, scalar.        The current likelihood of the previous sample.
    - curr_prior, scalar.       The current prior of the previous sample.
    - curr_params, array.       The last accepted parameters.
    - max_like, scalar.         The maximum likelihood so far.
    - params_max_like, array.   The parameters giving the maximum likelihood.
    - const, array.             The parameters not drawn.

    Return:
    -----------
    """
    # proposal
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
