"""
Sampling module for model of y = a'*x + b', giving:
    x = a*y - b,     
where x = [q, u], y = [Q, U], a = 1/R_Pp, b = [q, u]_bkgr,
e.g. ([Q, U]_bkgr/R_Pp)
"""

import numpy as np
import healpy as hp
import tools_mod as tools
import matplotlib.pyplot as plt
import sys, time

from scipy import stats
from functools import partial

np.random.seed(1843)

def sampler(QU, qu, C_ij, x_mean, x_err, mask, data_mean, sq=None, su=None,\
            p=None, sp=None, Nside=256, Niter=1000, N_Gibbs=20, R_Pp=1):
    """
    Method: Gibbs-chain over Metropolis Hastings

    Input:
    - QU, nd array.   The submm polarisation in units uK_cmb, 
                      shape=(2, Npix)
    - qu, nd array.   The visual polarisation (fractional), 
                      shape=(2, Npix)
    - C_ij, nd array. The covariance matrix of QU, in units of uK_cmb
                      (may be convertet from K_cmb), shape=(2,2,Npix)
    -

    Returns:
    -
    """
    
    print(data_mean)
    print(x_mean, np.shape(x_mean))
    print(x_err)
    
    # Definitions: (integers/scalars)
    N = len(mask)
    burnin = int(Niter/2)
    uKcmb2MJysr = 287.45 * 1e-6

    # Arrays
    models = np.zeros((N_Gibbs, 2, N))
    maxL_params = np.zeros((N_Gibbs, len(x_mean)))
    maxL = np.zeros(N_Gibbs)
    pv = np.zeros((N_Gibbs, N))
    y = np.zeros(np.shape(models))
    # Get covariance matrices in the correct shape
    C_qu = Cov(C_ij, dt=True) * uKcmb2MJysr**2
    cov0 = Cov(x_mean)
    print(logLike(model_func(data_mean, x=QU[:,mask]), data=qu[:,mask]))
    print(np.mean(p[mask]), np.std(p[mask]))
    t0 = time.time()
    # Gibbs chain:
    for i in range(N_Gibbs):
        t01 = time.time()

        # Draw QU' from N(QU, C_ij) [uKcmb] and convert to MJy/sr
        # Assume drawn y is perfect representation. shape=(2,len(mask))
        y[i,:,:] = draw_y(QU[:,mask], C_qu[:,:,mask], N)#np.array([yQ, yU])
        if p is not None:
            # Draw p_v values
            pv[i,:] = draw_y(p[mask], sp[mask], N)
        
        # Initialize
        log_like = partial(logLike, data=qu[:,mask])
        log_prior = partial(logPrior, mu=data_mean, sigma=x_err)
        func = partial(model_func, x=y[i,:,:]) # y or QU??

        params0, model0, loglike0, logprior0 = Initialize(log_like,\
                                                          log_prior,\
                                                          func, x_mean,\
                                                          cov0)
        #sys.exit()
        # MH algo.
        maxL_params[i, :], maxL[i] = MH(log_like, log_prior, func,\
                                            params0, model0, loglike0,\
                                            logprior0, x_mean, cov0,\
                                            burnin, Niter)
        
        
        models[i,:,:] = func(maxL_params[i,:])
        #print(mod, QU)
        t02 = time.time()
        print('Sampling time for Gibbs step {}: {} s'.format(i+1, t02-t01))
        print(' ')
        #sys.exit()
    # Gibbs chain end

    t1 = time.time()
    print('Total sampling time: {} s'.format(t1-t0))
    print(maxL)

    # Remove Gibbs-steps with bad maximum Likelihood
    cut_badL = np.logical_and(maxL, maxL > -10)
    #print(cut_badL)
    maxL = maxL[cut_badL]
    maxL_params = maxL_params[cut_badL,:]
    models = models[cut_badL,:,:]
    pv = pv[cut_badL,:]

    # Get best fit model + uncertainties:
    ind_best = np.where(maxL == max(maxL))[0][0]
    print(ind_best, maxL_params[ind_best,:])
    qu_model = models[ind_best,:,:]
    best_params = maxL_params[ind_best,:]
    print(qu_model)
    model_err, params_err = Error_estimation(models, maxL_params)
    #print(model_err)
    print(np.mean(qu_model/qu[:,mask], axis=1), np.std(qu_model/qu[:,mask], axis=1))
    print(np.corrcoef(qu_model[0,:], qu[0,mask]))
    print(np.corrcoef(qu_model[1,:], qu[1,mask]))
    #

    model_1A = model_func(best_params, x=QU[:,mask])
    ML_1 = logLike(qu_model, data=qu[:,mask])
    ML_1A = logLike(model_1A, data=qu[:,mask])
    
    # Check orthogonality:    
    tools.delta_psi(QU[0,mask], qu[0,mask], QU[1,mask], qu[1,mask])
    tools.delta_psi(QU[0,mask], qu_model[0,:], QU[1,mask], qu_model[1,:])
    tools.delta_psi(QU[0,mask], model_1A[0,:], QU[1,mask], model_1A[1,:])

    # Plotting:
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4.5))
    xx = np.linspace(-0.025,0.005,10)
    ax1.plot(qu[0,mask], qu_model[0,:], '.k')
    ax1.plot(qu[1,mask], qu_model[1,:], '.b')
    ax1.plot(xx, xx, 'gray')
    ax1.grid(True)
    #plt.subplot(212)
    a = best_params[:-2]
    ax2.scatter(qu[0,mask], QU[0,mask], marker='^', color='gray')
    ax2.scatter(qu[1,mask], QU[1,mask], marker='^', color='steelblue')
    ax2.plot(qu_model[0,:], QU[0,mask], '.k')
    ax2.plot(qu_model[1,:], QU[1,mask], '.b')
    ax2.plot(xx, xx/np.mean(a), 'gray', linestyle=':')
    ax2.grid(True)

    plot_params(maxL_params)

    

    if sq is not None:
        compare_with_data(QU[0,mask], QU[1,mask], qu[0,mask], qu[1,mask],\
                          qu_model[0,:], qu_model[1,:], C_ij[:,mask], \
                          sq[mask], su[mask], model_err,\
                          best_params, params_err, save=N_Gibbs)
    if p is not None:
        get_bkgr_percentage(maxL_params[:,-2:], params_err[-2:], pv,\
                            p_data=p[mask], best_fit=ind_best)
    
        
    # compare with pix independence:
    best_guess_pix = np.array([0.24027794, 0.25232319, 0.28067511,\
                               0.22065324, 0.27751509, 0.21551718,\
                               0.28457226, 0.2577072, 0.26976363,\
                               0.31075199, 0.33175059, 0.22250278,\
                               0.2449938, 0.24938268, 0.25435339,\
                               0.21490348, 0.2120685, 0.25049663,\
                               -0.00206514, 0.00212248])

    qu_mod_pix = model_func(best_guess_pix, x=QU[:,mask])
    model0 = model_func(data_mean, x=QU[:,mask])
    ML_0A = logLike(qu_mod_pix, data=qu[:,mask])
    ML_0 = logLike(model0, data=qu[:,mask])
    print('ML model  0', ML_0)
    print('ML model 0A', ML_0A)
    print('ML model  1', ML_1)
    print('ML model 1A', ML_1A)

    """
    f1, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 4))
    ax1.scatter(qu[0,mask], QU[0,mask], marker='o', color='g',\
                label='qQ data')
    ax2.scatter(qu[1,mask], QU[1,mask], marker='o', color='r',\
                label='uU data')
    #ax1.scatter(qu[0,mask], Q_mod0, marker='^', color='m',\
    #            label='Qq model0')
    #ax2.scatter(qu[1,mask], U_mod0, marker='^', color='c',\
    #            label='Uu model0')
    ax1.scatter(qu[0,mask], Q_mod_pix, marker='*',\
                color='k', label='qQ pix dependent')
    ax2.scatter(qu[1,mask], U_mod_pix, marker='*',\
                color='b', label='uU pix dependent')
    ax1.scatter(qu[0,mask], Q_mod, marker='x',\
                color='dimgrey', label='qQ pix indep.')
    ax2.scatter(qu[1,mask], U_mod, marker='x',\
                color='royalblue', label='uU pix indep.')
    ax2.legend()
    ax1.legend()
    ax1.set_xlabel('q data')
    ax2.set_xlabel('u data')
    ax1.set_ylabel('sampled (q + bq)/a = Q')
    ax2.set_ylabel('sampled (u + bu)/a = U')
    """
    plt.show()
    return qu_model, maxL_params[ind_best,:], model_err, params_err
    # end

def Initialize(log_like, log_prior, model_func, mean, cov):
    """                                                                       
    Initialization of the parameters and functions.                          
    """

    curr_params = proposal_rule(cov, mean, init=True)
    #print('Init params:')#, curr_params)
    #print_params(curr_params, int((len(mean)-2)/2))
    curr_model = model_func(curr_params)
    #print('Init model', curr_model)
    curr_like = log_like(curr_model)
    #print('Init like:', curr_like)  
    curr_prior = log_prior(curr_params)
    #print('Init prior', curr_prior)
    return(curr_params, curr_model, curr_like, curr_prior)

def MH(log_like, log_prior, model_func, curr_params, curr_model,\
       curr_like, curr_prior, mean, cov, burnin, Niter=1000):

    """                                                                       
    The Metropolis Hastings algorthm.                                         
    """
    accept = np.zeros(Niter)
    params = np.zeros((len(mean), Niter))
    counter = 0
    steplength = 1
    max_like = -100
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
        
        params[:,i] = curr_params
        # define current model, logL and logPrior from accepted parameters:
        curr_model_new = model_func(curr_params)
        curr_like = log_like(curr_model_new)
        curr_prior = log_prior(curr_params)
        mean = curr_params
        # update the steplength in the covariance matrix:
        if accept[i] == True:
            counter += 1

        if (i+1)%50 == 0: # 70 for R(pix)
            #print(i, counter/float(i+1), curr_like, max_like)
            #print_params(maxL_params, int((len(mean)-1)/2))
            #print_params(curr_params, int((len(mean)-1)/2))
            if counter/float(i+1) < 0.2:
                steplength /= 2
            elif counter/float(i+1) > 0.5:
                steplength *= 2
            else:
                pass
        # make new covariance matrix from the drawn parameters:
        if (i+1)%220 == 0: # 220
            cov = Cov(params[:,:i])
        
    #
    #print(curr_like, curr_prior, log_prior(maxL_params))
    accept_ratio = (counter/float(Niter))
    print(accept_ratio)
    print('max.like. {}, max.like. params:'.format(max_like), maxL_params)
    plt.figure('accept ratio vs maxlike')
    plt.plot(accept_ratio, max_like, '.r')
    plt.axvline(0.2, color='gray', linestyle=':')
    plt.axvline(0.5, color='gray', linestyle=':')
    #plt.plot(params.T)
    return(maxL_params, max_like)

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
    L = -0.5*((data - model)/sigma)**2
    return(np.sum(L))

def logPrior(params, mu=None, sigma=None):
    # change when bkgr?
    pm = -0.5*((params - mu)/sigma)**2
    return(np.sum(pm))

def draw_y(y, C, N):
    """Draw y values"""
    if np.ndim(y) > 1:
        y_new = np.zeros((2, N))
        for i in range(N):
            y_new[:,i] = np.random.multivariate_normal(y[:,i], C[:,:,i])
    else:
        y_new = np.random.normal(y, C)
    return(y_new)

def proposal_rule(cov, mean, init=False):
    """
    Draw new parameters for proposal.                          
    """

    params = np.random.multivariate_normal(mean, cov)
    # check if parameters are in right domain
    params[:-2] = test_params(params[:-2], mean[:-2], cov[:-2,:-2], crit='a')
    params[-2] = test_params(params[-2], mean[-2], cov[-2,-2],\
                             crit='bq', init=init)
    params[-1] = test_params(params[-1], mean[-1], cov[-1,-1],\
                             crit='bu', init=init)

    return(params)

def test_params(p, mean, cov, crit='a', i=0, init=False):
    """
    Test the drawn parameters, making sure they are not too off.
    """
    if crit == 'a':
        for k, param in enumerate(p):
            i = 0 # 1/5.5, 1/3.5 # 0.15, 0.35/40
            while (param < 0.15) or (param > 0.35):
                p_ = np.random.multivariate_normal(mean, cov)
                i += 1
                param = p_[k]
                if i > 20:
                    break
            p[k] = param
        return(p)

    elif crit == 'bq':
        while abs(p) > 0.01:
            p = np.random.normal(mean, np.sqrt(cov))
            i += 1
            if init is True:
                pass
            else:
                if i > 50:
                    break
        return(p)
    elif crit == 'bu':
        #while abs(p) > 0.015:
        while p > 0.01 or p < 0:
            p = np.random.normal(mean, np.sqrt(cov))
            i += 1
            if init is True:
                pass
            else:
                if i > 50:
                    break
        return(p)

def test_p(p, cov, crit):
    if abs(p) > crit:
        return(cov / 2)
    else:
        return(cov)


def Cov(x, y=None, dt=False):
    if np.ndim(x) == 1:
        N = len(x)
        return(np.eye(N))
    elif dt is True:
        return(np.array([[x[0,:], x[1,:]],[x[1,:], x[2,:]]]))
    elif y is not None:
        pass
    else:
        return(np.cov(x))

def model_func(params, x, star=False):
    """
    Model x = a*y - b
    
    Input:                              
    - params, list. (1/R_Pp, q_bkgr, u_bkgr)
    - x, 2d array of  pol.
    return    
    qu array  
    """
    a = params[:-2]
    q_bkgr, u_bkgr = params[-2:]
    q = -a*x[0,:] + q_bkgr
    u = -a*x[1,:] + u_bkgr
    return(np.array([q, u]))
                
def Error_estimation(x, y):
    """
    Estimate the uncertainties of the background polarisation, model 
    and stellar model.
    
    Input:
    -x ndarray. The array containting the models
    -y, ndarray. The array with parameters

    Returns:
    - C_model, std_params. The covariance matrix elements of the model and 
                           the standard deviation of the parameters.
    """
    C_ = np.zeros((2, 2, len(x[0,0,:])))
    for i in range(len(x[0,0,:])):
        C_[:,:,i] = np.cov(x[:,:,i], rowvar=False)
    
    C = np.array([C_[0,0,:], C_[0,1,:], C_[1,1,:]])
    return(C, np.std(y, axis=0))

def print_params(p, N):
    print('-',np.mean(p[:N]), p[-2], p[-1])
                                                    
def plot_params(p, hist=False, xlab=None, ylab=None, chain=False):
    """
    Plotting function of the parameters. 
    Input:
    - p, array. The parameters
    - hist, bool. If true make histogram dirtributions
    - xlab, string. The x-label.
    - ylab, string. The y-label.
    """
    print(np.ndim(p))

    
    N = int((len(p)-1)/2)
    print('Plot maximum likelihood parameters')
    f, ((a1,a2,a3),(a4,a5,a6)) = plt.subplots(2,3, figsize=(9, 5))
    
    sub_plot(a1, p[:,:-2], c='r', lab=r'$R_{{P/p}}^{{-1}}$ [sr/MJy]')
    sub_plot(a2, p[:,-2], c='k', lab=r'$q_{{bkgr}}$')
    sub_plot(a3, p[:,-1], c='b', lab=r'$u_{{bkgr}}$')

    sub_plot(a4, p[:,:-2].flatten(), c='r', hist=True, lab=r'$R_{{P/p}}^{{1}}$ [sr/MJy]')
    sub_plot(a5, p[:,-2], c='k', hist=True, lab=r'$u_{{bkgr}}$')
    sub_plot(a6, p[:,-1], c='b', hist=True, lab=r'$Q_{{bkgr}}$')
    a1.set_title(r'$R_{{P/p}}^{{-1}}$ [sr/MJy]')
    a2.set_title(r'$q_{{bkgr}}$ [%]')
    a3.set_title(r'$u_{{bkgr}}$ [%]')
    f.suptitle('Maximum likelihood parameters')
    #f.savefig('Figures/Sampling/maxL_params.png')
    #plt.show()

def sub_plot(ax, p, c='k', hist=False, lab=None):
    if hist is True:
        ax.hist(p, bins=10, histtype='step', color=c)
        ax.legend([r'${}\pm{}$'.format(round(np.mean(p),3),\
                                       round(np.std(p),3))])
    else:
        ax.plot(p, '.{}'.format(c))
        #ax.legend([lab])

def get_bkgr_percentage(bkgr, bkgr_err, p_sample, p_data=None, best_fit=0):
    """
    Get the polarisation percentage of the background, and a histogram 
    of the percent:
    - p_samples/p_v. Need pv_err??

    Input:
    - bkgr (N_Gibbs, 2)
    - pv (N_Gibbs, N)
    - bkgr_err (2)
    """
    #print(np.shape(bkgr), np.shape(pv), np.shape(bkgr_err))
    p_bkgr = tools.MAS(tools.get_P(bkgr[:,0], bkgr[:,1]),\
                       tools.get_P_err(bkgr[:,0], bkgr[:,1], bkgr_err[0],\
                                       bkgr_err[1]))
    print(bkgr[best_fit], bkgr_err)
    bkgr_percent1 = 100*p_bkgr/np.mean(p_sample, axis=1)
    bkgr_best1 = bkgr_percent1[best_fit]
    bkgr_percent1a = 100*p_bkgr/np.mean(p_sample)
    bkgr_best1a = bkgr_percent1a[best_fit]
    print(p_bkgr[best_fit])
    print(bkgr_best1, np.std(bkgr_percent1))
    print(bkgr_best1a, np.std(bkgr_percent1a))
    if p_data is not None:
        bkgr_percent2 = 100*p_bkgr/np.mean(p_data)
        bkgr_best2 = bkgr_percent2[best_fit]
        print(bkgr_best2, np.std(bkgr_percent2))
    bkgr_err_percent = np.std(bkgr_percent1)
    # Histogram:
    plt.figure()
    plt.hist(bkgr_percent1, bins=20, color='g', alpha=0.7,\
             density=True, stacked=True)
    #plt.hist(bkgr[:,0], bins=20, histtype='step', color='k')
    #plt.hist(bkgr[:,1], bins=20, histtype='step', color='b')
    plt.axvline(bkgr_best1, linestyle=':', color='k',\
                label=r'Best fit: ${}\pm{}$ $\%$'.\
                format(round(bkgr_percent1[best_fit], 3),\
                       round(bkgr_err_percent, 3)))
    plt.xlabel(r'Background polarization $p_{{bkgr}}/p_v$ $\%$')
    plt.ylabel('Density')
    plt.legend()
    
def compare_with_data(Q, U, q_v, u_v, q_m, u_m, C_pl, sq, su, C_m,\
                      params, params_err, save=None):
    """
    Compare the model against data in a scatter plot with uncertainties. 
    Try to estimate slopes with chi^2

    Input:
    q_v, u_v, sq, su. Polarisation data
    q_m, u_m. Sampled model
    C_m, C_pl. Covariance matrix elements estimates from the sampled 
    model and planck.
    params. maxL parameters
    params_err. uncertainties of maxL parameters.
    - all data arrays in the same length (N=Npix_used)
    """

    unit = 287.45*1e-6
    print('estimate slopes')
    # Estimate slopes:
    a = params[:-2] # 1/R
    #print(1/a, np.mean(1/a))
    sa = params_err[:-2]
    
    #print(np.sqrt(C_m))
    
    print('QU vs qu')
    par_qu, std_qu, chi2_qu = tools.Chi2(Q/unit, U/unit, q_m, u_m,\
                                      C_pl*1e-12, np.sqrt(C_m[0,:]),\
                                      np.sqrt(C_m[2,:]))
    print('Q vs q')
    print(np.corrcoef(q_m, Q/unit))
    par_q, std_q, chi2_q = tools.Chi2(Q/unit, None, q_m, None,\
                                      C_pl*1e-12, np.sqrt(C_m[0,:]), None)
    print('U vs u')
    print(np.corrcoef(u_m, U/unit))
    par_u, std_u, chi2_u = tools.Chi2(None, U/unit, None, u_m,\
                                      C_pl*1e-12, None, np.sqrt(C_m[2,:]))
    par_qu *= unit
    par_q *= unit
    par_u *= unit
    std_qu *= unit
    std_q *= unit
    std_u *= unit

    x = np.linspace(np.min(u_v), np.max(q_v), 10)

    plt.figure()
    # data points (Planck vs Tomo):
    e1 = plt.errorbar(q_v, Q, xerr=sq, yerr=np.sqrt(C_pl[0,:])*unit,\
                      fmt='none', ecolor='grey', label=r'$q_m, Q_s$ data')
    e2 = plt.errorbar(u_v, U, xerr=su, yerr=np.sqrt(C_pl[2,:])*unit,\
                      fmt='none', ecolor='steelblue', label=r'$u_m, U_s$ data')
    # data points (Planck vs sampled)
    e3 = plt.errorbar(q_m, Q, xerr=np.sqrt(C_m[0,:]),\
                      yerr=np.sqrt(C_pl[0,:])*unit, fmt='none',\
                      color='k', label=r'$q_m, Q_s$ sampled')
    e4 = plt.errorbar(u_m, U, xerr=np.sqrt(C_m[2,:]),\
                      yerr=np.sqrt(C_pl[2,:])*unit, fmt='none',\
                      color='b', label=r'$u_m, U_s$ sampled')
    
    legend1 = plt.legend(handles=[e1, e2, e3, e4], loc=1)
    ax = plt.gca().add_artist(legend1)

    # 1/a line:
    l0, = plt.plot(x, -x/np.mean(a), linestyle=':', c='silver')

    # slopes of the scatters:
    l1, = plt.plot(x, par_q[0]*x + par_q[1], '-k',\
                   label=r'$a_{{Q,q}}={}\pm{}$ MJy/sr'.\
                   format(round(par_q[0],2),round(std_q[0],2)))
    l2, = plt.plot(x, par_u[0]*x + par_u[1], '-b',\
                   label=r'$a_{{U,U}}={}\pm{}$ MJy/sr'.\
                   format(round(par_u[0],2),round(std_u[0],2)))
    l3, = plt.plot(x, par_qu[0]*x + par_qu[1], '-r',\
                   label=r'$a_{{QU,qu}}={}\pm{}$ MJy/sr'.\
                   format(round(par_qu[0],2),round(std_qu[0],2)))
    
    # layout:
    plt.legend(handles=[l1,l2,l3], loc=3)
    plt.grid(True)
    plt.xlabel(r'Sampled visual polarization, ($q_v, u_v$)')
    plt.ylabel(r'Planck 353 GHz, $Q_s, U_s$, [MJy/sr]')
    plt.savefig('Figures/Sampling/planck_vs_vis_sample_{}.png'.format(save))

    ############################################
    #  Produce a errorbar plot with q' = q-bq  #
    ############################################

    plt.figure() # plot tomo data vs sampled q, and u
    #print(stats.linregress(q_v, q_m))
    #print(stats.linregress(u_v, q_m))
    print(' ')
    print('-> Model; subtract background: x-b = -y/a')
    print(params[-2:], params_err[-2:])
    q_prime = q_m - params[-2]
    u_prime = u_m - params[-1]

    sq_prime = np.sqrt(C_m[0,:]) + params_err[-2]
    su_prime = np.sqrt(C_m[2,:]) + params_err[-1]

    print('QU vs qu')
    par_qu1, std_qu1, chi2_qu1 = tools.Chi2(Q/unit, U/unit, q_prime, u_prime,\
                                      C_pl*1e-12, sq_prime, su_prime)
    print('Q vs q')
    print(np.corrcoef(q_m, Q/unit))
    par_q1, std_q1, chi2_q1 = tools.Chi2(Q/unit, None, q_prime, None,\
                                      C_pl*1e-12, sq_prime, None)
    print('U vs u')
    print(np.corrcoef(u_m, U/unit))
    par_u1, std_u1, chi2_u1 = tools.Chi2(None, U/unit, None, u_prime,\
                                      C_pl*1e-12, None, su_prime)
    par_qu1 *= unit
    par_q1 *= unit
    par_u1 *= unit
    std_qu1 *= unit
    std_q1 *= unit
    std_u1 *= unit

    # data points:
    e5 = plt.errorbar(q_prime, Q, xerr=sq_prime,\
                      yerr=np.sqrt(C_pl[0,:])*unit, fmt='none',\
                      ecolor='k', label=r'$q-b,Q$')
    e6 = plt.errorbar(u_prime, U, xerr=su_prime,\
                      yerr=np.sqrt(C_pl[2,:])*unit, fmt='none',\
                      ecolor='b', label=r'$u-b,U$')

    legend1 = plt.legend(handles=[e5, e6], loc=1)
    ax = plt.gca().add_artist(legend1)

    # 1 to 1 line:
    plt.plot(x, -x/np.mean(a), linestyle=':', c='silver')

    # slopes of the scatters:
    l4, = plt.plot(x, par_q1[0]*x + par_q1[1], '-k',\
                   label=r'$a_{{Q,q}}={}\pm{}$ MJy/sr'.\
                   format(round(par_q1[0],2),round(std_q1[0],2)))
    l5, = plt.plot(x, par_u1[0]*x + par_u1[1], '-b',\
                   label=r'$a_{{U,U}}={}\pm{}$ MJy/sr'.\
                   format(round(par_u1[0],2),round(std_u1[0],2)))
    l6, = plt.plot(x, par_qu1[0]*x + par_qu1[1], '-r',\
                   label=r'$a_{{QU,qu}}={}\pm{}$ MJy/sr'.\
                   format(round(par_qu1[0],2),round(std_qu1[0],2)))
    
    # layout:
    plt.title(r'Modified visual polarization, $x-b = -y/a$')
    plt.legend(handles=[l4,l5,l6], loc=3)
    plt.grid(True)
    plt.xlabel(r'Sampled visual pol. $[q,u] - b$')
    plt.ylabel(r'Submm polarization$ 353 GHz [MJy/sr]')




    
#
