"""
Module for sampling background polarisation and stellar polarisation
in submm.
"""

import numpy as np
import healpy as hp
import tools_mod as tools
import matplotlib.pyplot as plt
import sys, time
from functools import partial



def QU_sampler(QU, qu, x_mean, x_err, mask, data_mean,\
               Nside=256, Niter=1000, R_Pp=None):
    """
    Input:
    - QU, 2d array of submm polarization (2, Npix)
    - qu, 2d array of visual polarization (2, Npix)
    - x_mean, seq. of mean parameters values to be fitted. 
                   (R_Pp, Qbkgr, Ubkgr)
    ...
    Return:
    QU stat and background. list of arrays
    """
    
    print(x_mean)
    print(x_err)
    burnin = int(Niter/2)    
    print(burnin)
    cov0 = Cov(x_mean)
    pixels = np.arange(hp.nside2npix(Nside))
    QU_model = np.zeros((2, hp.nside2npix(Nside)))
    QU_err = np.zeros((2, hp.nside2npix(Nside)))
    params_maxL = np.zeros((len(x_mean), hp.nside2npix(Nside)))
    params = np.zeros((Niter, len(x_mean), hp.nside2npix(Nside)))

    #print(R_Pp)
    dt_mean = np.array([-R_Pp, np.zeros(len(mask)), np.zeros(len(mask))])
    print('-----------')
    mod = QU_func(data_mean, qu[:,mask])
    mod2 = QU_func(dt_mean, qu[:,mask])
    #print(QU[:,mask], np.mean(QU[:,mask], axis=1), np.std(QU[:,mask], axis=1))
    #print(mod, np.mean(mod, axis=1), np.std(mod, axis=1))
    #print(mod2, np.mean(mod2, axis=1), np.std(mod2, axis=1))
    #print(logLike(mod, QU[:,mask]))
    #print(QU_func(np.array([-R_Pp[0], -0.005, 0.025]), qu[:,mask[0]]))
    res = (QU[:,mask] - mod)/QU[:,mask]
    res2 = (QU[:,mask] - mod2)/QU[:,mask]
    #print(res/res2)
    print(np.mean(QU[:,mask]/qu[:,mask], axis=1))
    print(np.mean(R_Pp*qu[0,mask]), np.mean(R_Pp*qu[1,mask]))

    """
    plt.scatter(res[0,:], res[1,:], marker='.', c=R_Pp, cmap='jet', vmin=3.8,\
                vmax=5.2)
    plt.colorbar()
    plt.grid(True)
    plt.figure()
    plt.scatter(qu[0,mask], QU[0,mask], marker='x', c='k')
    plt.scatter(qu[1,mask], QU[1,mask], marker='x', c='b')
    #plt.scatter(qu[0,mask], mod[0], marker='.', c=R_Pp, cmap='jet')
    #plt.scatter(qu[1,mask], mod[1], marker='.', c=R_Pp, cmap='jet')
    plt.scatter(qu[0,mask], mod2[0], marker='.', c=R_Pp, cmap='brg',\
                vmin=3.8, vmax=5.2)
    plt.scatter(qu[1,mask], mod2[1], marker='.', c=R_Pp, cmap='brg',\
                vmin=3.8, vmax=5.2)
    plt.colorbar()
    """
    t0 = time.time()
    for i, pix in enumerate(pixels[mask]):
        t01 = time.time()
        
        # Initiate functions:
        log_like = partial(logLike, data=QU[:,pix])
        log_prior = partial(logPrior, mu=data_mean, sigma=x_err)
        func = partial(QU_func, qu=qu[:,pix])
        
        # Initialize:
        params0, model0, loglike0, logprior0 = Initialize(log_like,\
                                                          log_prior,\
                                                          func, x_mean,\
                                                          cov0)

        # Metropolis Hastrings:
        params_maxL[:,pix], params[:,:,pix] = MH(log_like, log_prior,\
                                                 func, params0, model0,\
                                                 loglike0, logprior0,\
                                                 x_mean, cov0, burnin, Niter)

        #
        print(QU[:,pix], QU_func(params_maxL[:,pix], qu[:,pix]))
        #QU_model[:,pix] = QU_func(params_maxL[:,pix], qu[:,pix])
        #QU_err[:,pix] = None
        #print(np.mean(params[:,:,pix],axis=0), np.std(params[:,:,pix],axis=0))
        t11 = time.time()
        print('Sampling time for pixel {}: {} s'.format(pix, t11-t01))
        print('-->')
        #break
    #
    t2 = time.time()
    print('Total sampling time: {} s'.format(t2-t0))
    print('===================')
    #plot_params(params_maxL, xlab=[], ylab=[])
    #plot_params(params, hist=True, xlab=[], ylab=[])
    #plot_params(params, xlab=[], ylab=[])
    #print(params_maxL[:, mask])
    plt.subplot(311)
    plt.plot(params_maxL[0,mask], '.r')
    plt.subplot(312)
    plt.plot(params_maxL[1,mask], '.k')
    plt.subplot(313)
    plt.plot(params_maxL[2,mask], '.b')
    print(np.mean(params_maxL[:,mask], axis=1), np.std(params_maxL[:,mask], axis=1))
    print(np.shape(QU_err[:,mask]))
    params_err = error_est(params[burnin:,:,mask], model=False)
    QU_model[:, mask] = QU_func(params_maxL[:,mask], qu[:,mask])
    QU_err[:,mask] = error_est(params[burnin:,:,mask], qu[:,mask], model=True)

    res_mod = (QU[:,mask]-QU_model[:,mask])/QU[:,mask]
    #print(QU_model[:,mask])
    print(np.mean(QU[:,mask], axis=1), np.std(QU[:,mask], axis=1))
    print(np.mean(QU_model[:,mask], axis=1), np.std(QU_model[:,mask], axis=1))
    #print(res_mod)

    R_mod = np.sqrt((QU_model[0,mask]**2 + QU_model[1,mask]**2) \
                    / (qu[0,mask]**2 + qu[1,mask]**2))
    
    plt.figure()
    plt.plot(R_Pp, R_mod, '.g')
    plt.plot(R_Pp, -params_maxL[0,mask], '.r')

    print('Stellar and background polarisation')
    QU_star = QUstar(params_maxL[0,mask], qu[:,mask])
    QU_bkgr = QUbkgr(params_maxL[1:,mask])
    print(np.mean(QU[:,mask], axis=1))
    print(np.mean(QU_star, axis=1), np.std(QU_star, axis=1))
    print(np.mean(QU_bkgr, axis=1), np.std(QU_bkgr, axis=1))
    print(np.mean(R_mod), np.std(R_mod))
    plot_model1(qu, QU, QU_model, R_Pp)
    #"""
    plt.figure()
    plt.scatter(qu[0,mask], QU_star[0,:], c='k', marker='^')
    plt.scatter(qu[1,mask], QU_star[1,:], c='b', marker='^')
    plt.scatter(qu[0,mask], QU_model[0,mask], c='grey', marker='^')
    plt.scatter(qu[1,mask], QU_model[1,mask], c='skyblue', marker='^')
    
    plt.figure()
    plt.scatter(qu[0,mask], QU_bkgr[0,:], c='k', marker='^')
    plt.scatter(qu[1,mask], QU_bkgr[1,:], c='b', marker='^')
    #plt.ylim(-0.01, 0.015)
    #"""
    return None

def Initialize(log_like, log_prior, model_func, mean, cov):
    """
    Initialization of the parameters and functions.
    """
    
    curr_params = proposal_rule(cov, mean)
    #print('curr params:', curr_params)
    curr_model = model_func(curr_params)
    #print('model', curr_model)
    curr_like = log_like(curr_model)
    #print('curr like:', curr_like)
    curr_prior = log_prior(curr_params)
    #print('curr prior:', curr_prior)
    return(curr_params, curr_model, curr_like, curr_prior)

def MH(log_like, log_prior, model_func, curr_params, curr_model,\
       curr_like, curr_prior, mean, cov, burnin, Niter=1000):

    """
    The Metropolis Hastings algorthm.
    """
    accept = np.zeros(Niter)
    params = np.zeros((Niter, len(mean)))
    counter = 0
    steplength = 1
    max_like = -50
    maxL_params = curr_params

    # Sampling loop:
    for i in range(Niter):
        # propose new parameters:
        prop_params = proposal_rule(cov*steplength, mean)
        
        # call MH_step:  
        accept[i], curr_params, maxL_params, max_like =\
                                            MH_step(log_like, log_prior,\
                                            model_func, prop_params,\
                                            curr_params, curr_like,\
                                            curr_prior, max_like,\
                                            maxL_params)
        params[i,:] = curr_params
        #print(accept[i], prop_params)
        # define current model, logL and logPrior from accepted parameters:
        curr_model_new = model_func(curr_params)
        curr_like = log_like(curr_model_new)
        curr_prior = log_prior(curr_params)
        mean = curr_params
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
            #print(i, curr_params, curr_like)
            #print(cov)
    #
    print(counter/float(Niter))
    print('max.like. {}, max.like. params:'.format(max_like), maxL_params)
    mod = model_func(curr_params)
    modL = model_func(maxL_params)
    print(mod)
    print(modL)
    print(np.mean(params[burnin:,:],axis=0),np.std(params[burnin:,:],axis=0))
    
    #plt.subplot(311)
    #plt.plot(params[:,0])
    #plt.subplot(312)
    #plt.plot(params[:,1])
    #plt.subplot(313)
    #plt.plot(params[:,2])
    #plt.show()
    return(maxL_params, params)


def MH_step(log_like, log_prior, model_func, prop_params, curr_params,\
            curr_like, curr_prior, max_like, maxL_params):

    """
    The step in the MH algorithm
    """
    # proposed model: 
    prop_model = model_func(prop_params)
    prop_like = log_like(prop_model)
    prop_prior = log_prior(prop_params)

    # posterior:                     
    post_old = curr_like + curr_prior
    post_new = prop_like + prop_prior
    #print(prop_prior, curr_prior, prop_like, curr_like)
    # acceptance testing:            
    a = np.exp(post_new - post_old)
    draw = np.random.uniform(0, 1)
    #print(a, draw, post_old, post_new)
    if (a > draw) and (a < np.inf):
        accept = True
        curr_params = prop_params
        if prop_like > max_like:
            max_like = prop_like
            maxL_params = curr_params
    else:
        accept = False
        curr_params = curr_params
    #                                                           
    return(accept, curr_params, maxL_params, max_like)
    

def logLike(model, data, sigma=0.01):
    L = -0.5*((data - model)/sigma)**2
    return(np.sum(L))

def logPrior(params, mu=None, sigma=None):
    pm = 0
    for i in range(len(params)):
        pm += -0.5*((params[i] - mu[i])/sigma[i])
    return(pm)

def proposal_rule(cov, mean=None):
    """
    Draw new parameters for proposal.                                         
    """
    params = np.random.multivariate_normal(mean, cov)
    # check if R_Pp has the right sign? should be negative
    params[0] = test_params(params[0], mean[0], cov[0,0], crit='a')
    params[1] = test_params(params[1], mean[1], cov[1,1], crit='bq')
    params[2] = test_params(params[2], mean[2], cov[2,2], crit='bu')
    #print(params)
    return(params)

def test_params(p, mean, cov, crit='q', i=0):
    if crit != 'a':
        while p >= 0:
            p = np.random.normal(mean, np.sqrt(cov))
            i += 1
            if i > 10:
                break
        return(p)
    elif crit == 'bq':
        while abs(p) > 1:
            p = np.random.normal(mean, np.sqrt(cov))
            i += 1
            if i > 10:
                break
        return(p)
    elif crit == 'bu':
        while (abs(p) > 1):
            p = np.random.normal(mean, np.sqrt(cov))
            i += 1
            if i > 10:
                break
        return(p)
    else:
        return(p)

def Cov(x):
    if np.ndim(x) == 1:
        N = len(x)
        return(np.eye(N))
    else:
        return(np.cov(x))

def QU_func(params, qu):
    """
    Input:
    - params, list. (R_Pp, Qbkgr, Ubkgr)
    - qu, 2d array of visula pol.
    return
    QU array
    """

    QU = np.array([params[0]*qu[0] + params[1],\
                   params[0]*qu[1] + params[2]])
    return(QU)

def QUstar(a, qu):
    """
    Estimate the submm polarisation contribution from stellar parts
    """
    return(a*qu)

def QUbkgr(bs):
    """
    Estimate the background polarisation 
    """
    return(np.array([bs[0,:], bs[1,:]]))

def error_est(params, qu=None, model=True):
    """
    Function to estimate the uncertainties of the model, or for each
    parameter
    
    Input:
    - params, array with all sampled parameter values after burnin
    - qu, array with the visual polarization data
    - model, bool. If True estimate uncertainties for the model, else
    estimate uncertainties of the parameters.
    Returns:
    uncertianties for model or parameters
    """
    #print(np.shape(params))
    if model is True:
        print('uncertainties for model')
        QU = np.zeros((len(params[:,0,0]), 2, len(qu[0,:])))
        for i in range(len(params[:,0,0])):
            QU[i,:,:] = QU_func(params[i,:,:], qu)
        print(np.shape(QU)) # (500,2,30)
        QU_err = np.std(QU, axis=0)
        #print(QU_err) 
        return QU_err
    else:
        print('Uncertainties for parameters')
        params_err = np.std(params, axis=0)
        #print(np.shape(params_err))
        return(params_err)

def plot_model1(qu, QU, mod, R_mod):
    plt.figure()
    plt.scatter(qu[0,mask], QU[0,mask], c='grey', marker='^', label='data')
    plt.scatter(qu[1,mask], QU[1,mask], c='skyblue', marker='^')
    plt.scatter(qu[0,mask], QU_model[0,mask], c=R_mod, marker='.', cmap='brg',\
                vmin=3.8, vmax=5.2, label='model')
    plt.scatter(qu[1,mask], QU_model[1,mask], c=R_mod, marker='.', cmap='brg',\
                vmin=3.8, vmax=5.2)
    cbar = plt.colorbar()
    cbar.set_label(r'$R_{{P/p}}$ [MJy/sr]')
    plt.xlabel(r'$q, u$')
    plt.ylabel(r'$Q, U$ [MJy/sr]')
    plt.title('Plot data vs model')
    plt.savefig('Figures/correlations/test/model_vs_data_sampler.png')

def plot_params(p, hist=False, xlab=None, ylab=None):
    if np.dim(p) == 3:
        print('plot histogram of returned samples')
        f1, (ax1, ax2) = plt.subplots(2,1)
        if hist is True:
            for i in range(len(p[0,0,:])):
                ax1.hist(p[:,0,i], bins=50, c='r', histtype='step')
        
                ax2.hist(p[:,1,i], bins=50, c='k',histtype='step',\
                         label=r'$Q$')
                ax2.hist(p[:,2,i], bins=50, c='b',histtype='step',\
                         label=r'$U$')
        else:
            for i in range(len(p[0,0,:])):
                ax1.plot(p[:,0,i], '-r')
                ax2.plot(p[:,1,i], '-k', label=r'$Q$')
                ax2.plot(p[:,2,i], '-b', label=r'$U$')
        #

        ax1.set_xlabel(xlab[0])
        ax1.set_ylabel(ylab)
        
        ax2.set_xlabel(xlab[1])
        ax2.set_ylabel(ylab)
        ax2.legend()
        
   
    else:
        print('Plot maximum likelihood parameters')
        f, ((a1,a2,a3),(a4,a5,a6)) = plt.subplots(3,2)
        
        a1.plot(p[0,:], '.r')
        a2.plot(p[1,:], '.k')
        a3.plot(p[2,:], '.b')

#
