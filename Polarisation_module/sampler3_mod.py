import numpy as np
import healpy as hp
import tools_mod as tools
import matplotlib.pyplot as plt
import sys, time
from functools import partial

np.random.seed(1843)

def sampler(QU, qu, x_mean, x_err, mask, data_mean,\
               Nside=256, Niter=10000, R_Pp=None):
    """
    Sample R_Pp per pixel and Q,U background as one value for the sky patch 
    
    Vectorized sampling, shape=(npix+2)
    """
    
    print(data_mean)
    print(x_mean, np.shape(x_mean))
    print(x_err)
    QU_model = np.full((2, hp.nside2npix(Nside)), np.nan)
    QU_star = np.full((2, hp.nside2npix(Nside)), np.nan)
    model_err = np.full((3, hp.nside2npix(Nside)), np.nan)
    star_err = np.full((3, hp.nside2npix(Nside)), np.nan)
    burnin = int(Niter/2)
    
    t0 = time.time()
    # Initialize function:
    log_like = partial(logLike, data=QU[:,mask])
    log_prior = partial(logPrior, mu=data_mean, sigma=x_err)
    func = partial(QU_func, qu=qu[:,mask])
        
    print(log_like(func(data_mean)))
    cov0 = Cov(x_mean)
    #print(cov0)

    # Initialize:
    params0, model0, loglike0, logprior0 = Initialize(log_like,\
                                                      log_prior,\
                                                      func, x_mean,\
                                                      cov0)
    #sys.exit()
    # Metropolis Hastrings:
    params_maxL, params = MH(log_like, log_prior, func, params0,\
                             model0, loglike0, logprior0, x_mean,\
                             cov0, burnin, Niter)

    t1 = time.time()
    print('Sampling time: {} s'.format(t1-t0))
    print(np.mean(params_maxL[:len(mask)]), np.mean(params_maxL[len(mask):-1]), params_maxL[-1])
    sys.exit()
    #print(np.std(params[burnin:,:], axis=0))
    #print(np.shape(params))
    model = QU_func(params_maxL, qu[:,mask])
    star_model = QU_func(params_maxL, qu[:,mask], star=True)
    uncertainties = Error_estimation(params[burnin:, :], qu[:,mask]) #!
    R_err = np.std(params[burnin:, :len(mask)]) #!

    QU_model[:,mask] = model
    QU_star[:,mask] = star_model
    model_err[:,mask] = uncertainties[0]
    star_err[:,mask] = uncertainties[1]

    plt.figure()
    plt.plot(qu[0,mask], QU[0,mask], '.k')
    plt.plot(qu[1,mask], QU[1,mask], '.b')
    plt.scatter(qu[0,mask], model[0,:], marker='x', c='gray')
    plt.scatter(qu[1,mask], model[1,:], marker='x', c='g')
    x = np.linspace(np.min(qu), np.max(qu), 10)
    plt.plot(x, -np.mean(params_maxL[:-2])*x, '-r')
    plt.plot(x, -np.mean(params_maxL[:-2])*x+params_maxL[-2], '-k')
    plt.plot(x, -np.mean(params_maxL[:-2])*x+params_maxL[-1], '-b')
    plt.grid(True)

    plt.figure()
    plt.plot(params)
    
    #plot_params(params_maxL, xlab='hei', ylab='hopp')
    #plot_params(params[burnin:,:], xlab='hei', ylab='hopp')
    #plot_params(params[burnin:,:], hist=True, xlab='hei', ylab='hopp')
    return [QU_model, QU_star,\
            params_maxL[-2:]], [model_err, star_err,\
                                uncertainties[2]]
    # end


def Initialize(log_like, log_prior, model_func, mean, cov):
    """                                                                       
    Initialization of the parameters and functions.                          
    """

    curr_params = proposal_rule(cov, mean, (len(mean)-1)/2)
    print('Init params:', curr_params)                                         
    print_params(curr_params, int((len(mean)-1)/2))
    curr_model = model_func(curr_params)
    print('Init model', curr_model)
    curr_like = log_like(curr_model)
    print('Init like:', curr_like)                                             
    curr_prior = log_prior(curr_params)
    print('Init prior', curr_prior)
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
    max_like = -500
    maxL_params = curr_params

    # Sampling loop:                                                          
    for i in range(Niter):
        # propose new parameters:                                             
        prop_params = proposal_rule(cov*steplength, mean, (len(mean)-1)/2)

        # call MH_step:                                                      
        accept[i], curr_params, maxL_params, max_like =\
                                            MH_step(log_like, log_prior,\
                                            model_func, prop_params,\
                                            curr_params, curr_like,\
                                            curr_prior, max_like,\
                                            maxL_params)
        
        params[i,:] = curr_params
        # define current model, logL and logPrior from accepted parameters:
        curr_model_new = model_func(curr_params)
        curr_like = log_like(curr_model_new)
        curr_prior = log_prior(curr_params)
        mean = curr_params
        # update the steplength in the covariance matrix:
        if accept[i] == True:
            counter += 1

        if (i+1)%300 == 0:
            print(i, counter/float(i+1), curr_like, max_like)
            #print('  ', np.mean(curr_params[:-2]), curr_params[-2:])
            #print_params(maxL_params, int((len(mean)-1)/2))
            #print('-')
            #print_params(curr_params, int((len(mean)-1)/2))
            if counter/float(i+1) < 0.2:
                steplength /= 2
            elif counter/float(i+1) > 0.5:
                steplength *= 2
            else:
                pass
        # make new covariance matrix from the drawn parameters:
        if (i+1)%500 == 0:
            cov = Cov(params[:i,:].T)
            #print_params(np.diag(cov), int((len(mean)-1)/2))
        #print_params(curr_params, int((len(mean)-1)/2))
    #
    print(curr_like, curr_prior)
    print(counter/float(Niter))
    print('max.like. {}, max.like. params:'.format(max_like), maxL_params)
    return(maxL_params, params)

def print_params(p, N):
    print('-',np.mean(p[:N]), np.mean(p[N:-1]), p[-1])

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
    
    # acceptance testing:
    a = np.exp(post_new - post_old)
    draw = np.random.uniform(0, 1)
    
    if (a > draw) and (a < np.inf):
        accept = True
        curr_params = prop_params
        #print(curr_like, max_like)
        if prop_like > max_like:
            max_like = prop_like
            maxL_params = curr_params
    else:
        accept = False
        curr_params = curr_params
    
    return(accept, curr_params, maxL_params, max_like)

def logLike(model, data, sigma=0.01):
    # change when bkgr?
    #print(np.shape(data))
    L = -0.5*((data - model)/sigma)**2
    #print(np.shape(L))
    return(np.sum(L))

def logPrior(params, mu=None, sigma=None):
    # change when bkgr?
    #pm = 0
    #print(np.shape(params), mu, sigma)
    #for i in range(len(params)):
    pm = -0.5*((params - mu)/sigma)
    return(np.sum(pm))

def proposal_rule(cov, mean, npix):
    """
    Draw new parameters for proposal.                          
    """
    npix = int(npix)
     
    params = np.random.multivariate_normal(mean, cov)
    # check if parameters are in right domain
    
    params[:npix] = test_params(params[:npix], mean[:npix],\
                              cov[:npix,:npix], crit='Rpp')
    params[npix:-1] = test_params(params[npix:-1], mean[npix:-1],\
                                  cov[npix:-1,npix:-1], crit='Pb')
    params[-1] = test_params(params[-1], mean[-1], cov[-1,-1], crit='psib')
    
    #print(params)                          
    return(params)

def test_params(p, mean, cov, crit='a', i=0):
    #print(p, len(p))
    if crit == 'Rpp':
        #p_ = np.zeros(len(p))
        for k, param in enumerate(p):
            #p_[k] = param
            i = 0
            while (param < 1.5) or (param > 6.2):
                p_ = np.random.multivariate_normal(mean, cov)
                #p = np.random.normal(mean, np.sqrt(np.diag(cov)))
                #p_[k] = np.random.normal(mean[k], np.sqrt(cov[k,k]))
                i += 1
                param = p_[k]
                if i > 20:
                    break
            p[k] = param
        return(p)
    elif crit == 'Pb':
        for k, param in enumerate(p):
            while (param > 0.05) and (param < 0):
                p_ = np.random.multivariate_normal(mean, cov)
                i += 1
                param = p_[k]
                #print(k, i, param)
                if i > 20:
                    break
            p[k] = param    
        return(p)
    elif crit == 'psib':
        while (p > np.pi/2) or (p < 0.):
            p = np.random.normal(mean, np.sqrt(cov))
            i += 1
            if i > 20:
                break
        #    #print(p, ',')
        return(p)

def Cov(x):
    if np.ndim(x) == 1:
        N = len(x)
        return(np.eye(N))
    else:
        return(np.cov(x))

def QU_func(params, qu, star=False):
    """                                 
    Input:                              
    - params, list. (R_Pp, Qbkgr, Ubkgr)
    - qu, 2d array of visual pol.
    return    
    QU array  
    """
    npix = len(qu[0])
    #print(params, qu, len(params[:-2]))
    if star is True:
        return np.array([-params[:npix]*qu[0], -params[:-2]*qu[1]])
    else:
        bkgr = background(params[npix:])
        QU = np.array([-params[:npix]*qu[0] + bkgr[0,:],\
                       -params[:npix]*qu[1] + bkgr[1,:]])
        return(QU)

def background(params):
    """
    Model describing the background polarisation [Q,U]_bkgr, where
    [Q,U]_bkgr = P_bkgr^pix * [sin, cos](2*psi_bkgr)
    """
    P_b = params[:-1]
    psi_b = params[-1]
    return(np.array([P_b*np.cos(2*psi_b), P_b*np.sin(2*psi_b)]))


def Error_estimation(params, qu):
    """
    Estimate the uncertainties of the background polarisation, model 
    and stellar model.
    
    Input:
    -params ndarray (3, Niter/2, Npixs)
    -qu, ndarray (2, Npixs)
    """

    star = np.zeros((2, len(params[0,:-2]), len(params[:,0])))
    model = np.zeros((2, len(params[0,:-2]), len(params[:,0])))
    for i in range(len(params[:,0])):
        star[:,:,i] = QU_func(params[i,:], qu, star=True)
        model[:,:,i] = QU_func(params[i,:], qu)

    bkgr_err = np.std(params[:,-2:], axis=0)
    star_err0 = np.std(star, axis=2)
    mod_err0 = np.std(model, axis=2)
    star_err = np.array([star_err0[0,:], star_err0[1,:],\
                         np.sqrt(star_err0[0,:]*star_err0[1,:])])
    model_err = np.array([mod_err0[0,:], mod_err0[1,:],\
                          np.sqrt(mod_err0[0,:]*mod_err0[1,:])])                
    return(model_err, star_err, bkgr_err)

def plot_params(p, hist=False, xlab=None, ylab=None):
    """
    Plotting function of the parameters. 
    Input:
    - p, array. The parameters
    - hist, bool. If true make histogram dirtributions
    - xlab, string. The x-label.
    - ylab, string. The y-label.
    """
    print(np.ndim(p))
    if np.ndim(p) == 2:
        print('plot histogram of returned samples')
        f1, (ax1, ax2) = plt.subplots(2,1)
        if hist is True:
            name = 'hist'
            
            ax1.hist(p[:,:-2].flatten(), bins=50, color='r', histtype='step')
            ax2.hist(p[:,-2], bins=50, color='k',histtype='step')
            ax2.hist(p[:,-1], bins=50, color='b',histtype='step')
            #ax1.set_xlabel(xlab[0])                                            
            ax1.set_ylabel(ylab[0])

            ax2.set_xlabel(xlab[1])
            ax2.set_ylabel(ylab[0])
            f1.suptitle('Sample distribution for parameters')
        else:
            name = 'chain'
            
            ax1.plot(p[:,:-2], '-r')
            ax2.plot(p[:,-2], '-k')
            ax2.plot(p[:,-1], '-b')
            #ax1.set_xlabel(xlab[0])                                            
            ax1.set_ylabel(ylab[0])

            ax2.set_xlabel(xlab[0])
            ax2.set_ylabel(ylab[1])
            f1.suptitle('Sampling chains for parameters')
        #                                                                       
        ax2.legend([r'$Q$',r'$U$'])
        #f1.savefig('Figures/Sampling/MH_samples_{}.png'.format(name))

    else:
        print('Plot maximum likelihood parameters')
        f, ((a1,a2,a3),(a4,a5,a6)) = plt.subplots(2,3, figsize=(9, 5))

        sub_plot(a1, p[:-2], c='r', lab=r'$R_{{P/p}}$ [MJy/sr]')
        sub_plot(a2, p[-2], c='k', lab=r'$Q_{{bkgr}}$ [MJy/sr]')
        sub_plot(a3, p[-1], c='b', lab=r'$U_{{bkgr}}$ [MJy/sr]')

        sub_plot(a4, p[:-2], c='r', hist=True, lab=r'$Q_{{bkgr}}$ [MJy/sr]')
        sub_plot(a5, p[-2], c='k', hist=True, lab=r'$Q_{{bkgr}}$ [MJy/sr]')
        sub_plot(a6, p[-1], c='b', hist=True, lab=r'$Q_{{bkgr}}$ [MJy/sr]')
        a1.set_title(r'$R_{{P/p}}$ [MJy/sr]')
        a2.set_title(r'$Q_{{bkgr}}$ [MJy/sr]')
        a3.set_title(r'$U_{{bkgr}}$ [MJy/sr]')
        f.suptitle('Maximum likelihood parameters')
        #f.savefig('Figures/Sampling/maxL_params.png')
    #                                                                           

def sub_plot(ax, p, c='k', hist=False, lab=None):
    if hist is True:
        ax.hist(p, bins=10, histtype='step', color=c)
        ax.legend(['mean {}'.format(round(np.mean(p),3))])
    else:
        ax.plot(p, '.{}'.format(c))
        #ax.legend([lab])


#
