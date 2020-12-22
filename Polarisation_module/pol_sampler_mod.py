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


#np.random.seed(8798345)


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

    cov0 = Cov(x_mean)
    pixels = np.arange(hp.nside2npix(Nside))
    QU_model = np.zeros((2, hp.nside2npix(Nside)))
    QU_err = np.zeros((2, hp.nside2npix(Nside)))
    QU_star = np.zeros((2, hp.nside2npix(Nside)))
    QU_bkgr = np.zeros((2, hp.nside2npix(Nside)))
    params_maxL = np.zeros((len(x_mean), hp.nside2npix(Nside)))
    params = np.zeros((Niter, len(x_mean), hp.nside2npix(Nside)))
    params_err = np.zeros((3, hp.nside2npix(Nside)))

    dt_mean = np.array([R_Pp, np.zeros(len(mask)), np.zeros(len(mask))])
    print('-----------')
    mod = QU_func(data_mean, qu[:,mask])
    mod2 = QU_func(dt_mean, qu[:,mask])

    res = (QU[:,mask] - mod)/QU[:,mask]
    res2 = (QU[:,mask] - mod2)/QU[:,mask]

    print(np.mean(QU[:,mask]/qu[:,mask], axis=1))
    print(np.mean(R_Pp*qu[0,mask]), np.mean(R_Pp*qu[1,mask]))

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
                                                 x_mean, cov0, \
                                                 burnin, Niter)

        #
        # store maxL params and curr params.
        print(QU[:,pix], QU_func(params_maxL[:,pix], qu[:,pix]))

        t11 = time.time()
        print('-->')
        #break

    t2 = time.time()
    print('Total sampling time: {} s'.format(t2-t0))
    print('===================')
    
    #sys.exit()
    print(params_maxL[:,mask])
    plot_params(params_maxL[:,mask], xlab=[r'Iterations'],\
                ylab=[r'$R_{{P/p}}$ [MJy/sr]', r'$Q_{{bkgr}}$ [MJy/sr]',\
                      r'$U_{{bkgr}}$ [MJy/sr]'])
    plot_params(params[:,:,mask], hist=True, xlab=[r'$R_{{P/p}}$ [MJy/sr]',\
                                         r'Background $Q, U$ [MJy/sr]'],\
                ylab=['Counts'])
    plot_params(params[:,:,mask], xlab=['Iterations'],\
                ylab=[r'$R_{{P/p}}$ [MJy/sr]',\
                      r'$Q_{{bkgr}},U_{{bkgr}}$ [MJy/sr]'])

    print(np.mean(params_maxL[:,mask], axis=1), np.std(params_maxL[:,mask], axis=1))
    print('median',np.median(params_maxL[:,mask], axis=1))
    print(np.shape(QU_err[:,mask]))
    params_err[:,mask] = error_est(params[burnin:,:,mask], model=False)
    QU_model[:, mask] = QU_func(params_maxL[:,mask], qu[:,mask])
    QU_err[:,mask] = error_est(params[burnin:,:,mask], qu[:,mask], model=True)
    
    print(np.shape(params_err))
    
    res_mod = (QU[:,mask]-QU_model[:,mask])/QU[:,mask]
    print(np.mean(QU[:,mask], axis=1), np.std(QU[:,mask], axis=1))
    print(np.mean(QU_model[:,mask], axis=1), np.std(QU_model[:,mask], axis=1))

    R_mod = np.sqrt((QU_model[0,mask]**2 + QU_model[1,mask]**2) \
                    / (qu[0,mask]**2 + qu[1,mask]**2))
    
    print('Stellar and background polarisation')
    QU_star[:,mask] = QUstar(params_maxL[0,mask], qu[:,mask])
    QU_bkgr[:,mask] = QUbkgr(params_maxL[1:,mask])
    print(np.mean(QU[:,mask], axis=1))
    print(np.mean(QU_star, axis=1), np.std(QU_star, axis=1))
    print(np.mean(QU_bkgr, axis=1), np.std(QU_bkgr, axis=1))
    print(np.mean(R_mod), np.std(R_mod))

    # Check polarisation angle:
    dpsi, psi_v, psi_s = tools.delta_psi(QU_model[0,mask], qu[0,mask],\
                                         QU_model[1,mask], qu[1,mask])
    dx1, x1_v, x1_s = tools.delta_psi(QU_star[0,mask], qu[0,mask],\
                                      QU_star[1,mask], qu[1,mask])
    dx2, x2_v, x2_s = tools.delta_psi(QU_bkgr[0,mask], qu[0,mask],\
                                      QU_bkgr[1,mask], qu[1,mask])
    
    plot_model1(qu, QU, QU_model, R_Pp, mask)

    """
    Correlation plots data and model. residual maps. slopes of model, star, 
    bkgr in correlation plots with visual. Need chi^2, residual estimates 
    
    Correlations plotted after returned to main:
    """
    print(np.shape(QU_model), np.shape(QU_star), np.shape(QU_bkgr))
    print(np.shape(params_maxL), np.shape(params))
    print(np.shape(QU_err), np.shape(params_err))
    #correlation(qu, QU_model, sq, su, model_err, mask, lab='model', save='model')
    return([QU_model, QU_star, QU_bkgr], [params_maxL, params[burnin:,:,:]],\
           [QU_err, params_err])

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
    #mod = model_func(curr_params)
    #modL = model_func(maxL_params)
    #print(mod)
    #print(modL)
    #print(np.mean(params[burnin:,:],axis=0),np.std(params[burnin:,:],axis=0))
    
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
    # check if parameters are in right domain
    params[0] = test_params(params[0], mean[0], cov[0,0], crit='a')
    params[1] = test_params(params[1], mean[1], cov[1,1], crit='bq')
    params[2] = test_params(params[2], mean[2], cov[2,2], crit='bu')
    #params = test_params(params, mean, cov, crit='all')
    #print(params)
    return(params)

def test_params(p, mean, cov, crit='a', i=0):
    
    if crit == 'a':
        while (p <= 2.5) or (p > 6.):
            p = np.random.normal(mean, np.sqrt(cov))
            i += 1
            if i > 10:
                break
        return(p)
    elif crit == 'bq':
        while (abs(p) > 0.025):
            p = np.random.normal(mean, np.sqrt(cov))
            i += 1
            if i > 10:
                break
        return(p)
    elif crit == 'bu':
        while (p > 0.06) or (p < 0):
            p = np.random.normal(mean, np.sqrt(cov))
            i += 1
            if i > 10:
                break
        return(p)
    elif crit == 'all':
        while (p[0] <= 0) or (p[0] > 10) or (abs(p[1]) > 0.05) or (p[2] > 0.1)\
              or (p[2] < 0):
            p = np.random.multivariate_normal(mean, cov)
            i += 1
            if i > 10:
                break
        print(i)
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

    QU = np.array([-params[0]*qu[0] + params[1],\
                   -params[0]*qu[1] + params[2]])
    return(QU)

def QUstar(a, qu):
    """
    Estimate the submm polarisation contribution from stellar parts
    """
    return(-a*qu)

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

def star_error(a, a_err, pol, pol_err):
    """
    Estimation of the uncertainties of Q_star, U_star:
    -a 1d array (Npixels)
    -a_err 1d array (Npixels)
    -pol 1d array (Npixels)
    -pol_err 1d array (Npixels)
    """
    sigma = np.abs(a*pol_err) + np.abs(a_err*pol)
    return(sigma)
    
def residual(d, m, mask):
    r = np.full(np.shape(d), fill_value=hp.UNSEEN)
    r[:,mask] = (d[:,mask] - m[:,mask])/np.abs(d[:,mask])
    return(r)

def correlation(qu, mod, sq, su, mod_err, mask, lab='', save='', QU=None,\
                QU_err=None, R_Pp=None, R_err=None):
    """
    Plot correlation plot of model vs visual polarisation. If QU is not none 
    include in plot.

    """
    unit = 287.45*1e-6
    unit2 = 287.45*1e-12
    qu_a = np.concatenate((qu[0,:], qu[1,:]))
    mod_a = np.concatenate((mod[0,:], mod[1,:]))
    
    Ps = tools.get_P(mod[0,:], mod[1,:])
    pv = tools.get_P(qu[0,:], qu[1,:])
    P_err = tools.get_P_err(mod[0,:], mod[1,:], mod_err[0,:], mod_err[1,:])
    p_err = tools.get_P_err(qu[0,:], qu[1,:], sq, su)
    if R_Pp is None:
        R_mod = np.mean(tools.MAS(Ps, P_err)/tools.MAS(pv, p_err))
        print(R_mod)
    else:
        R_Pp.append(np.mean(tools.MAS(Ps, P_err)/tools.MAS(pv, p_err)))
        print(R_Pp)

    # chi^2 estimate:
    print('Joint')
    param, sigma, chi2 = tools.Chi2(mod[0,:]/unit, mod[1,:]/unit, qu[0,:],\
                                    qu[1,:], (mod_err/unit)**2,\
                                    sq, su, sampler=True)
    print(np.corrcoef(qu_a, mod_a))
    print('Qq')
    param_q, sigma_q, chi2_q = tools.Chi2(mod[0,:]/unit, None, qu[0,:], None,\
                                          (mod_err/unit)**2, sq, \
                                          None, sampler=True)
    print(np.corrcoef(qu[0,:], mod[0,:]))
    print('Uu')
    param_u, sigma_u, chi2_u = tools.Chi2(None, mod[1,:]/unit, None, qu[1,:],\
                                          (mod_err/unit)**2, None,\
                                          su, sampler=True)
    print(np.corrcoef(qu[1,:], mod[1,:]))

    tools.delta_psi(mod[0,:], mod[1,:], qu[0,:], qu[1,:])

    x = np.linspace(np.min(qu_a), np.max(qu_a), 10)
    plt.figure('scatter {}'.format(lab))
    # points
    plt.scatter(qu[0,:], mod[0,:], marker='^', c='k')
    plt.scatter(qu[1,:], mod[1,:], marker='v', c='b')

    if QU is not None:
        plt.scatter(qu[0,:], QU[0,:], marker='.', c='grey')
        plt.scatter(qu[1,:], QU[1,:], marker='.', c='skyblue')
    # Slopes
    plt.plot(x, x*param_q[0]*unit + param_q[1]*unit, '-k',\
             label=r'$a_{{Qq}}$={}'.format(round(param_q[0]*unit,3)))
    plt.plot(x, x*param_u[0]*unit + param_u[1]*unit, '-b',\
             label=r'$a_{{Uu}}$={}'.format(round(param_u[0]*unit,3)))
    plt.plot(x, x*param[0]*unit + param[1]*unit, '-r',\
             label=r'$a_{{QU-qu}}$={}'.format(round(param[0]*unit,3)))

    if R_Pp is not None:
        print('Data R_Pp:', R_Pp[0])
        print('Model R_Pp:', R_Pp[1])
        plt.plot(x, -x*R_Pp[0], '--r', label=r'$R_{{Pp}}^{{data}}$={}'.\
                 format(round(R_Pp[0], 3)))
        plt.plot(x, -x*R_Pp[1], ':r', label=r'$R_{{Pp}}^{{max L}}$={}'.\
                 format(round(R_Pp[1], 3)))
        
    plt.xlabel(r'$q_v, u_v$')
    plt.ylabel(r'{} $Q_s, U_s$ [MJy/sr]'.format(lab))
    plt.grid(True)
    plt.legend()
    plt.savefig('Figures/Sampling/QUqu_{}_{}_corr.png'.format(save, lab))

    plt.figure('errorbar {}'.format(lab))
    x = np.linspace(np.min(qu), np.max(qu), 10)
    plt.errorbar(qu[0,:], mod[0,:], xerr=sq, yerr=mod_err[0,:], fmt='none',\
                 ecolor='k')
    plt.errorbar(qu[1,:], mod[1,:], xerr=su, yerr=mod_err[1,:], fmt='none',\
                 ecolor='b')

    # Slopes
    plt.plot(x, x*param_q[0]*unit + param_q[1]*unit, '-k',\
             label=r'$a_{{Qq}}={}\pm{}$'.format(round(param_q[0]*unit, 3),\
                                                round(sigma_q[0]*unit, 3)))
    plt.plot(x, x*param_u[0]*unit + param_u[1]*unit, '-b',\
             label=r'$a_{{Uu}}={}\pm{}$'.format(round(param_u[0]*unit, 3),\
                                           round(sigma_u[0]*unit, 3)))
    plt.plot(x, x*param[0]*unit + param[1]*unit, '-r',\
             label=r'$a_{{QUqu}}={}\pm{}$'.format(round(param[0]*unit, 3),\
                                              round(sigma[0]*unit, 3)))
    if R_Pp is not None:
        plt.plot(x, -x*R_Pp[0], '--r', label=r'$R_{{Pp}}^{{data}}={}\pm{}$'.\
                 format(round(R_Pp[0], 3), round(R_err[0], 3)))
        plt.plot(x, -x*R_Pp[0], ':r', label=r'$R_{{Pp}}^{{max L}}={}\pm{}$'.\
                 format(round(R_Pp[1], 3), round(R_err[1], 3)))
    
    plt.legend()
    plt.xlabel(r'$q_v, u_v$')
    plt.ylabel(r'{} $Q_s, U_s$ [MJy/sr]'.format(lab))
    plt.grid(True)
    plt.savefig('Figures/Sampling/QUqu_{}_{}_ebar.png'.format(save, lab))
    
    #

def plot_model_vs_data(q, u, Q, U, c='r', m='.', lab=False):
    """
    Make one plot with the data vs the model including full, star and bkgr
    
    Include slopes?
    """
    leg = []
    leg.append(lab)
    
    plt.figure('Data vs model')
    plt.scatter(q, Q, c=c[0], marker=m)
    plt.scatter(u, U, c=c[1], marker=m)
    plt.xlabel(r'$q_v, u_v$')
    plt.ylabel(r'$Q_s, U_s$ [MJy/sr]')
    plt.grid(True)
    plt.legend(leg)
    plt.savefig('Figures/Sampling/model_vs_data.png')

def plot_model1(qu, QU, mod, R_mod, mask, lab=''):
    plt.figure()
    plt.title('Data vs model {}'.format(lab))
    plt.scatter(qu[0,mask], QU[0,mask], c='grey', marker='^', label='data')
    plt.scatter(qu[1,mask], QU[1,mask], c='skyblue', marker='^')
    plt.scatter(qu[0,mask], mod[0,mask], c='k', marker='.', label='model')
    plt.scatter(qu[1,mask], mod[1,mask], c='b', marker='.')
    
    plt.legend()
    plt.xlabel(r'$q, u$')
    plt.ylabel(r'$Q, U$ [MJy/sr]')
    plt.title('Plot data vs model')
    plt.savefig('Figures/Sampling/model_vs_data_sampler.png')
    
    # Residuals:
    res = residual(QU, mod, mask)
    print(np.shape(res))
    hp.gnomview(res[0,:], title=r'Residuals, $\frac{Q_s - Q_m}{|Q_s|}$',\
                cmap='bwr', min=-1, max=1, rot=[104,22.2], xsize=150)
    hp.graticule()
    plt.savefig('Figures/Sampling/Q_residuals.png')
    hp.gnomview(res[1,:], title=r'Residuals, $\frac{U_s - U_m}{|U_s|}$',\
                cmap='bwr', min=-1, max=1, rot=[104, 22.2], xsize=150)
    hp.graticule()
    plt.savefig('Figures/Sampling/U_residuals.png')

def plot_params(p, hist=False, xlab=None, ylab=None):
    if np.ndim(p) == 3:
        print('plot histogram of returned samples')
        f1, (ax1, ax2) = plt.subplots(2,1)
        if hist is True:
            name = 'hist'
            for i in range(len(p[0,0,:])):
                ax1.hist(p[:,0,i], bins=50, color='r', histtype='step')
        
                ax2.hist(p[:,1,i], bins=50, color='k',histtype='step')
                ax2.hist(p[:,2,i], bins=50, color='b',histtype='step')
            #ax1.set_xlabel(xlab[0])
            ax1.set_ylabel(ylab[0])
        
            ax2.set_xlabel(xlab[1])
            ax2.set_ylabel(ylab[0])
            f1.suptitle('Sample distribution for parameters')
        else:
            name = 'chain'
            for i in range(len(p[0,0,:])):
                ax1.plot(p[:,0,i], '-r')
                ax2.plot(p[:,1,i], '-k')
                ax2.plot(p[:,2,i], '-b')
            #ax1.set_xlabel(xlab[0])
            ax1.set_ylabel(ylab[0])
        
            ax2.set_xlabel(xlab[0])
            ax2.set_ylabel(ylab[1])
            f1.suptitle('Sampling chains for parameters')
        #
        ax2.legend([r'$Q$',r'$U$'])
        f1.savefig('Figures/Sampling/MH_samples_{}.png'.format(name))

    else:
        print('Plot maximum likelihood parameters')
        f, ((a1,a2,a3),(a4,a5,a6)) = plt.subplots(2,3, figsize=(9, 5))
        
        sub_plot(a1, p[0,:], c='r', lab=r'$R_{{P/p}}$ [MJy/sr]')
        sub_plot(a2, p[1,:], c='k', lab=r'$Q_{{bkgr}}$ [MJy/sr]')
        sub_plot(a3, p[2,:], c='b', lab=r'$U_{{bkgr}}$ [MJy/sr]') 

        sub_plot(a4, p[0,:], c='r', hist=True, lab=r'$Q_{{bkgr}}$ [MJy/sr]')
        sub_plot(a5, p[1,:], c='k', hist=True, lab=r'$Q_{{bkgr}}$ [MJy/sr]')
        sub_plot(a6, p[2,:], c='b', hist=True, lab=r'$Q_{{bkgr}}$ [MJy/sr]')
        a1.set_title(r'$R_{{P/p}}$ [MJy/sr]')
        a2.set_title(r'$Q_{{bkgr}}$ [MJy/sr]')
        a3.set_title(r'$U_{{bkgr}}$ [MJy/sr]')
        f.suptitle('Maximum likelihood parameters')
        f.savefig('Figures/Sampling/maxL_params.png')
    #

def sub_plot(ax, p, c='k', hist=False, lab=None):
    if hist is True:
        ax.hist(p, bins=10, histtype='step', color=c)
        ax.legend(['mean {}'.format(round(np.mean(p),3))])
    else:
        ax.plot(p, '.{}'.format(c))
        #ax.legend([lab])
#
