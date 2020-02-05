"""
Module for the statisics functions.
"""

import numpy as np
#import healpy as hp
#import matplotlib.pyplot as plt
import sys, time
#import h5py

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

    for i in range(len(data)):
        L += -0.5*((data[i] - model[i])/sigma)**2
    #l = -0.5*((data - model)/sigma)**2
    #L = np.sum(l)
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
    #pm = -0.5*((params - mu)/sigma)**2
    #pm = np.sum(pm)
    #"""
    for i in range(len(mu)):
        pm += -0.5*((params[i] - mu[i])/sigma[i])**2

        if (params[i] <= 0) and (i < len(params)-1):
            c += 1

    if c > 0:
        pm = -50
    else:
        pm = pm
    return(pm)

def Cov(N, err):
    """
    Function to compute the covariace matrix of the parameters.

    Parameters:
    -----------
    - N, integer.   The length of the (NxN) matrix, represent the number of
                    parameters.
    - err, array.   The standard deviation/uncertainty of the parameters.
    Return:
    -----------
    - cov, ndarray. The computed covariance matrix.
    """

    if N > 1:
        C = np.eye(N)
        for i in range(N):
            for j in range(N):
                if i != j:
                    C[i,j] = 0.81
                else:
                    C[i,j] = err[i]**2
        #
    else:
        C = 0.81
    return(C)
