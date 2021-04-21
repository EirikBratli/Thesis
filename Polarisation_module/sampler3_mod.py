import numpy as np
import healpy as hp
import tools_mod as tools
import matplotlib.pyplot as plt
import sys, time
from functools import partial

"""
Sampling module in order to estimate background polarisation contribution.
The model assumes a uniform background magnetic field but with varying 
polarisation intensity P. The model is on the form y = ax + b
where a = R_Pp estimated for each pixel, x is the visual polarisation 
[q_v, u_v] and b = P_b * [cos(2*psi_b), sin(2*psi_b)], where P_b varies 
over pixel and psi_b is constant.

The parameter space is for R_Pp (> 2, < 4) MJy/sr, P_b (>0, < 0.05) MJy/sr, 
psi_b (> 0.8, < 1.2) radians.
"""



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
    N = len(mask)
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
    print(np.mean(data_mean[N:-1]))    
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
    print(params_maxL[:len(mask)], params_maxL[len(mask):-1], params_maxL[-1])
    print_params(params_maxL, int((len(x_mean)-1)/2))
    
    #print(np.std(params[burnin:,:], axis=0))
    #print(np.shape(params))
    QU_model[:,mask] = QU_func(params_maxL, qu[:,mask])
    QU_star[:,mask] = QU_func(params_maxL, qu[:,mask], star=True)
    bkgr_model = background(params_maxL[len(mask):])
    sample_err = Error_estimation(params[burnin:, :], par=True) #!
    model_err[:,mask], star_err[:,mask], bkgr_err\
        = Error_estimation(params[burnin:,:], qu[:,mask])

    mod0 = QU_func(data_mean, qu[:,mask])
    plt.figure()
    plt.plot(qu[0,mask], QU[0,mask], '.k')
    plt.plot(qu[1,mask], QU[1,mask], '.b')
    plt.scatter(qu[0,mask], QU_model[0,mask], marker='x', c='gray')
    plt.scatter(qu[1,mask], QU_model[1,mask], marker='x', c='skyblue')
    #plt.scatter(qu[0,mask], mod0[0,:], marker='^', c='orange')
    #plt.scatter(qu[1,mask], mod0[1,:], marker='^', c='g')
    #"""
    x = np.linspace(np.min(qu), np.max(qu), 10)
    b = np.mean(bkgr_model, axis=1)
    print(b, np.shape(model_err), '<-----------')
    print(np.corrcoef(params_maxL, data_mean))

    plt.plot(x, -np.mean(params_maxL[:N])*x, '-r')
    plt.plot(x, -np.mean(params_maxL[:N])*x + b[0], '-k')
    plt.plot(x, -np.mean(params_maxL[:N])*x + b[1], '-b')
    plt.grid(True)
    #"""
    
    compare_params(params_maxL, data_mean, N)
    
    plot_params(params_maxL, xlab='hei', ylab='hopp')
    #plot_params(data_mean, xlab='hei', ylab='hopp')
    plot_params(params[:,:], xlab='hei', ylab='hopp')
    #plot_params(params[burnin:,:], hist=True, xlab='hei', ylab='hopp')
    print(np.std(params[burnin:,:], axis=0))
    #plt.show()
    #sys.exit()
    return([QU_model, QU_star, bkgr_model, params_maxL, params[burnin:,:]],\
           [model_err, star_err, bkgr_err, sample_err])
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
    max_like = -100
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
        
        if (i+1)%100 == 0: # 150
            
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
        if (i+1)%250 == 0: # 800
            cov = Cov(params[:i,:].T)
            #print_params(np.diag(cov), int((len(mean)-1)/2))
        #print_params(curr_params, int((len(mean)-1)/2))
    #
    print(curr_like, curr_prior)
    print(counter/float(Niter))
    print('max.like. {}, max.like. params:'.format(max_like), maxL_params)
    print('Prior: {}'.format(log_prior(maxL_params)))
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
            while (param < 2.0) or (param > 4.0): # 2-4
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
            i = 0
            while (param > 0.05) or (param < 0):
                p_ = np.random.multivariate_normal(mean, cov)
                i += 1
                param = p_[k]
                #print(k, i, param)
                if i > 50:
                    break
            p[k] = param    
        return(p)
    elif crit == 'psib':
        while (p > 1.2) or (p < 0.8):
            p = np.random.normal(mean, np.sqrt(cov))
            i += 1
            if i > 20:
                break
        #    #print(p, ',')
        return(p)

def Cov(x, y=None):
    if np.ndim(x) == 1:
        N = len(x)
        if y is None:
            return(np.eye(N))
        else:
            mu_x = np.mean(x)    
            mu_y = np.mean(y)
            C_xy = np.sum((x - mu_x)*(y - mu_y))/N
            return(C_xy)
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
    if np.ndim(params) == 2:
        mod = np.zeros((2, npix, len(params[:,0])))
        if star is True:
            for i in range(len(params[:,0])):
                mod[:,:,i] = np.array([-params[i,:npix]*qu[0],\
                                       -params[i,:npix]*qu[1]])
        else:
            for i in range(len(params[:,0])):
                bkgr = background(params[i,npix:])
                mod[:,:,i] = np.array([-params[i,:npix]*qu[0] + bkgr[0,:],\
                                       -params[i,:npix]*qu[1] + bkgr[1,:]])
        return mod
    else:
        if star is True:
            return np.array([-params[:npix]*qu[0], -params[:npix]*qu[1]])
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

def par_err(params):
    # Compute the standard deviation of the samples.
    return(np.std(params, axis=0))
    
def Error_estimation(params, qu=None, qu_err=None, p_maxL=None, par=False):
    """
    Estimate the uncertainties of the background polarisation, model 
    and stellar model.
    
    Input:
    -params ndarray (2*Npix+1) or (Niter/2, 2*Npix+1)
    -qu, ndarray (2, Npixs)
    """
    
    if par is True:
        print('Estimate uncertainties for the parameters')
        sigma = par_err(params)
        return(sigma)
    elif qu is not None:
        N = len(qu[0,:])
        model_err = np.zeros((3, N))
        star_err = np.zeros((3,N))
        msg = 'Estimate model uncertainties'
        print(msg)
        if qu_err is None:
            print(' -> by varying the model')
            star_err[:-1,:] = np.std(QU_func(params, qu, star=True), axis=2)
            model_err[:-1,:] = np.std(QU_func(params, qu), axis=2)
            bkgr_err = np.std(background(params[N:,:]), axis=2) #?
        
        else:
            print('-> using data and parameters')
            err = par_err(params)
            s_ax = (p_maxL[:N]*qu_err)**2 + (qu*err[:N])**2
            s_b = np.array([(err[N:-1]*np.cos(2*p_maxL[-1]))**2 +\
                            (2*p_maxL[N:-1]*err[-1]*np.sin(2*p_maxL[-1]))**2,\
                            (err[N:-1]*np.sin(2*p_maxL[-1]))**2 +\
                            (2*p_maxL[N:-1]*err[-1]*np.sin(2*p_maxL[-1]))**2])
            model_err[:-1,:] = np.sqrt(s_ax + s_b)
            star_err[:-1,:] = np.sqrt(s_ax)
            bkgr_err = np.sqrt(s_b)
        star_err[-1,:] = np.sqrt(Cov(star_err[0,:], star_err[1,:]))
        model_err[-1,:] = np.sqrt(Cov(model_err[0,:], model_err[1,:]))
        return(model_err, star_err, bkgr_err)

def compare_params(samples, data, N):
    """
    """
    print(samples[-1]/data[-1])
    f, (ax1, ax2) = plt.subplots(2,1)
    ax1.scatter(data[:N], samples[:N], marker='x', c='r')
    ax1.set_xlabel(r'$R_{{P/p}}$ data')
    ax1.set_ylabel(r'$R_{{P/p}}$ samples')
    ax2.scatter(data[N:-1], samples[N:-1], marker='d', c='g')
    ax1.set_xlabel(r'$P_{{bkgr}}$ data')
    ax1.set_ylabel(r'$P_{{bkgr}}$ samples')

def plot_params(p, hist=False, xlab=None, ylab=None):
    """
    Plotting function of the parameters. 
    Input:
    - p, array. The parameters
    - hist, bool. If true make histogram dirtributions
    - xlab, string. The x-label.
    - ylab, string. The y-label.
    """
    #print(np.ndim(p))

    if np.ndim(p) == 2:
        N = int((len(p[0,:])-1)/2)
        print('plot histogram of returned samples')
        f1, (ax1, ax2) = plt.subplots(2,1)
        if hist is True:
            name = 'hist'
            
            ax1.hist(p[:,:N].flatten(), bins=50, color='r', histtype='step')
            ax2.hist(p[:,N:-1], bins=50, color='k',histtype='step')
            ax2.hist(p[:,-1], bins=50, color='b',histtype='step')
            #ax1.set_xlabel(xlab[0])                                            
            ax1.set_ylabel(ylab[0])

            ax2.set_xlabel(xlab[1])
            ax2.set_ylabel(ylab[0])
            f1.suptitle('Sample distribution for parameters')
        else:
            name = 'chain'
            
            ax1.plot(p[:,:N], '-r')
            ax2.plot(p[:,N:-1], '-k')
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
        N = int((len(p)-1)/2)
        sub_plot(a1, p[:N], c='r', lab=r'$R_{{P/p}}$ [MJy/sr]')
        sub_plot(a2, p[N:-1], c='k', lab=r'$Q_{{bkgr}}$ [MJy/sr]')
        sub_plot(a3, p[-1], c='b', lab=r'$U_{{bkgr}}$ [MJy/sr]')

        sub_plot(a4, p[:N], c='r', hist=True, lab=r'$Q_{{bkgr}}$ [MJy/sr]')
        sub_plot(a5, p[N:-1], c='k', hist=True, lab=r'$Q_{{bkgr}}$ [MJy/sr]')
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
