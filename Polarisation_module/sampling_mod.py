"""
Module for sampling polarization, to solve the problem of large intercept in Uu
relation. Use Metropolis Hastings algorithm. 

Sample R_Pp over pixels, then background pol.
"""

import numpy as np
import healpy as hp
import tools_mod as tools
import sys, time
from functools import partial
import matplotlib.pyplot as plt

np.random.seed(1843)


def main_sampler(QU, qu, x_mean, x_err, data_mean, mask, Nside=256,\
                 Niter=1000):
    """
    Main sampling function.

    Input:
    - amp. The submm intensity/polarisation (T, P, [Q,U]) in MJy/sr
    """


    #x_mean = params_mean
    #x_err = params_err
    print(x_mean, x_err)
    cov0 = Cov(x_mean)
    burnin = int(Niter/2)
    Pol_model = np.zeros((3, hp.nside2npix(Nside)))
    Pol_err = np.zeros((2, hp.nside2npix(Nside)))
    params_maxL = np.zeros(len(mask)+2)
    params = np.zeros((Niter, len(x_mean)))#, len(mask)))

    print(np.shape(params), len(params_maxL))
    print('-----------')
    t0 = time.time()
    # sample R_Pp per pixel:
    for i, pix in enumerate(mask):
        t01 = time.time()
        print('Sampling for pixel:', pix, i)
        dt_mean = np.array([data_mean[i], data_mean[-2], data_mean[-1]])
        sigma = np.array([x_err[i], x_err[-2], x_err[-1]])
        mean = np.array([x_mean[i], x_mean[-2], x_mean[-1]])

        log_like = partial(logLike, data=QU[:,pix])
        log_prior = partial(logPrior, mu=dt_mean, sigma=sigma)
        QU_func = partial(model_QU, pol=qu[:,pix])

        # Initialize:        
        params0, model0, loglike0, logprior0 = Initialize(log_like,\
                                                          log_prior,\
                                                          QU_func, mean,\
                                                          cov0[i:i+3, i:i+3],\
                                                          R=True)
        #sys.exit()
        # Metropolis Hastings algorithm:
        params_maxL[i:i+3], params[:,i:i+3] = MH(log_like, log_prior,\
                                               QU_func, params0, model0,\
                                               loglike0, logprior0,\
                                               mean, cov0[i:i+3,i:i+3],\
                                               Niter=Niter)
        # params returns shape (Niter, 3) into (Niter, 3(npix+2))

        t1 = time.time()
        print('Sampling time for pixel {}: {} s'.format(pix, t1-t01))
        #sys.exit()
    #
    
    x_mean[:-2] = params_maxL[:-2]\
                + np.random.normal(np.zeros(len(mask)),\
                                   np.fabs(params_maxL[:-2])/30.)
    
    print('-> sample background')
    # sample [Q, U]_bkgr:
    log_like = partial(logLike, data=QU[:,mask])
    log_prior = partial(logPrior, mu=data_mean, sigma=x_err)
    QU_func = partial(model_QU, pol=qu[:,mask])
    
    # Initialize:
    params0, model0, loglike0, logprior0 = Initialize(log_like, log_prior,\
                                                      QU_func, x_mean,\
                                                      cov0, R=False)

    # MH sampling:
    bkgr_maxL, bkgr_params = MH(log_like, log_prior, QU_func, params0,\
                                model0, loglike0,logprior0, x_mean,\
                                cov0, Niter=Niter)

    # bkgr_params returns shape (Niter, Nparams=20)
    t2 = time.time()
    print('Total sampling time: {} s'.format(t2-t0))
    print('------------')
    params_maxL[-2:] = bkgr_maxL[-2:]
    params[:,-2:] = bkgr_params[:,-2:]
    print(np.shape(bkgr_params))
    print(params_maxL, np.shape(params))
    
    model = model_QU(params_maxL, qu[:,mask])
    star_model = model_QU(params_maxL, qu[:,mask], star=True)
    bkgr_model = params_maxL[-2:]

    model_err, star_err, bkgr_err = Error_estimation(params[burnin:,:],\
                                                     qu[:,mask])
    print(model)
    print('Model maxL:', logLike(model, QU[:,mask]))
    print('R_pp mean:', np.mean(params_maxL[:-2]))
    print('[QU]_bkgr:', params_maxL[-2:], bkgr_err)
    print(np.std(params_maxL[:-2]), np.std(params[burnin:,:-2])) # which 1??

    plt.figure()
    plt.scatter(qu[0,mask], QU[0,mask], marker='.', c='k')
    plt.scatter(qu[1,mask], QU[1,mask], marker='.', c='b')
    plt.scatter(qu[0,mask], model[0,:], marker='x', c='gray')
    plt.scatter(qu[1,mask], model[1,:], marker='x', c='g')
    
    plt.figure()
    plt.plot(params[burnin:,-2:])
    plt.figure()
    plt.plot(params[burnin:,:-2])
    
    #

def MH(log_like, log_prior, model_func, curr_params, curr_model,\
       curr_like, curr_prior, mean, cov, Niter=1000):
    """
    The metropolis-Hastings algorithm.
    """
    accept = np.zeros(Niter)
    params = np.zeros((Niter, len(mean)))
    counter = 0
    steplength = 1
    max_like = -50
    params_max_like = curr_params
    
    # Sampling loop:
    for i in range(Niter):
        # propose new parameters:
        prop_params = proposal_rule(cov*steplength, mean)
        
        # call MH_step:
        accept[i], curr_params, params_max_like, max_like =\
                                    MH_step(log_like, log_prior,\
                                            model_func, prop_params,\
                                            curr_params, curr_like,\
                                            curr_prior, max_like,\
                                            params_max_like)
        params[i,:] = curr_params
        #print(params[i,:])
        # define current model, logL and logPrior from accepted parameters:
        curr_model_new = model_func(curr_params)
        curr_like = log_like(curr_model_new)
        curr_prior = log_prior(curr_params)

        # update the steplength in the covariance matrix:
        if accept[i] == True:
            counter += 1
        if (i+1)%50 == 0:
            if counter/float(i+1) < 0.2:
                steplength /= 2
            elif counter/float(i+1) > 0.5:
                steplength *= 2
            else:
                pass
        
        # make new covariance matrix from the drawn parameters:
        if (i+1)%200 == 0:
            cov = Cov(params[:i,:].T)
            #print(i+1, cov)
    #
    print(counter/float(Niter))
    print('max.like. {}, max.like. params:'.format(max_like), params_max_like)
    #plt.figure('params')
    #plt.plot(params)    
    #plt.show()
    return(params_max_like, params)

def MH_step(log_like, log_prior, model_func, prop_params, curr_params,\
            curr_like, curr_prior, max_like, max_like_params):
    """
    The step in the MH algorithm.
    """
    # proposed model:
    prop_model = model_func(prop_params)
    prop_like = log_like(prop_model)
    prop_prior = log_prior(prop_params)
    
    # posterior:
    post_old = curr_like + curr_prior
    post_new = prop_like + prop_prior
    
    # acceptance testing:
    a = np.exp(post_new - post_old)
    draw = np.random.uniform(0, 1)
    #print(a, post_new-post_old)
    if (a > draw) and (a < np.inf):
        accept = True
        curr_params = prop_params
        if prop_like > max_like:
            max_like = prop_like
            max_like_params = curr_params
    else:
        accept = False
        curr_params = curr_params
    
    #
    return(accept, curr_params, max_like_params, max_like)


def Initialize(log_like, log_prior, model_func, mean, cov, R=True):
    """
    Initialization of the parameters and functions.
    """
    
    if R is True:
        # init. for pol. ratio
        curr_Rpp = proposal_rule(cov[0,0], mean[0], Rpp=True)
        curr_params = np.append(curr_Rpp, mean[-2:])
    else:
        curr_bkgr = proposal_rule(cov[-2:,-2:], mean[-2:], Rpp=False)
        curr_params = np.append(mean[:-2], curr_bkgr)
    print('params', curr_params)
    curr_model = model_func(curr_params)
    print('model', curr_model)
    curr_like = log_like(curr_model)
    print('loglike', curr_like)
    curr_prior = log_prior(curr_params)
    print('logprior', curr_prior)
    return(curr_params, curr_model, curr_like, curr_prior)

def logLike(model, data, sigma=10.):
    """
    What shape has model and data?
    """
    #L = 0
    
    #for i in range(len(data)):
    #    L += -0.5*((data[i] - model[i])/sigma)**2
    L = -0.5*((data - model)/sigma)**2
    return(np.sum(L))

def logPrior(params, mu=None, sigma=None):
    """
    
    """
    pm = 0
    #print(params, mu, sigma)
    for i in range(len(params)):
        pm += -0.5*((params[i] - mu[i])/sigma[i])**2
    #pm = -0.5*((params - mu)/sigma)**2
    return(pm)
    
def proposal_rule(cov, mean, Rpp=False):
    """
    Draw new parameters for proposal.
    """
    
    if Rpp is False:
        params = np.random.multivariate_normal(mean, cov)
        
        #params[-2] = test_params(params[-2], cov[-2,-2], part='bq')
        #params[-1] = test_params(params[-1], cov[-1,-1], part='bu')
        #params[-2] = test_params(params[-2], cov[0,], part='Rpp')
    
    else:
        params = np.random.normal(mean, cov)

    return(params)

def test_params(param, cov, part='bq'):
    i = 0
    if part == 'bq':
        while abs(param) > 0.05:
            param = np.random.normal(param,np.sqrt(cov))
            i += 1
            if i > 10:
                break
        return param
    elif part == 'bu':
        while (param > 0.05) or (param < 0):
            param = np.random.normal(param,np.sqrt(cov))
            i += 1
            if i > 10:
                break
        return param
    elif part == 'Rpp':
        while (param > 5) or (param < 2):
            param = np.random.normal(param,np.sqrt(cov))
            i += 1
            if i > 10:
                break
        return param


def Cov(x):
    if np.ndim(x) == 1:
        N = len(x)
        return(np.eye(N))
    else:
        return(np.cov(x))
        
def model_QU(params, pol, star=False):
    """
    Polarisation model for the submm Stokes parameters from stellar 
    polarisation and background contribution.
    Q_s = a*q_v + Q_bkgr or
    U_s = b*u_v + U_bkgr
    """
    if star is True:
        return np.array([-params[:-2]*pol[0], -params[:-2]*pol[1]])
    else:
        Q = -params[:-2]*pol[0] + params[-2]
        U = -params[:-2]*pol[1] + params[-1]
        return(np.array([Q, U]))

def Error_estimation(params, qu):
    """                      
    Compute the uncertainty of sampling model, star model and background.

    Input:                                       
    - params, ndarray (Niter, Nparams, Npix)   
    - qu ndarray (2, Npix)               
    - model_func, function that returns the model
    """
    print('Estimate uncertainties')
    star = np.zeros((2, len(params[0,:-2]), len(params[:,0])))
    model = np.zeros(np.shape(star))
    for i in range(len(params[:,0])):

        star[:,:,i] = model_QU(params[i,:], qu, star=True)
        model[:,:,i] = model_QU(params[i,:], qu)
        
    bkgr_err = np.std(params[:,-2:], axis=0)
    star_err = np.std(star, axis=2)
    model_err =np.std(model, axis=2)
    return(model_err, star_err, bkgr_err)

