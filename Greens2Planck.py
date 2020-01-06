"""
Program to compare Planck 857Ghz map to Greens et.al. 2019 extinction map.
And modeling the scale factor between the the two maps, in units of RJ?
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import h5py
import sys, time

import convert_units
#from Greens19maps import distance_array, Greens_maps
from mcmc_sampler import MetropolisHastings, mh_step, proposal_rule
from functools import partial

##### Functions #####

def load_planck_map(file):
    m857, hdr = hp.fitsfunc.read_map(file, h=True)
    return(m857)

def fix_resolution(map, Nside_in=0, ordering='RING'):
    sides = [8,16,32,64,128]
    maps = []
    smaps = []
    t0 = time.time()
    if Nside_in > 0:
        print('Fix resolution to Nside={}'.format(Nside_in))
        m = hp.pixelfunc.ud_grade(map, Nside_in, order_in=ordering,\
                                order_out=ordering)
        return(m)

    else:
        for Ns in Nsides:
            t1 = time.time()
            print('Fix resolution to Nside={}'.format(Ns))
            m = hp.pixelfunc.ud_grade(map, Ns, order_in=ordering,\
                                    order_out=ordering)
            maps.append(m)

            t2 = time.time()

            print('Interation time: {}s, total time: {}s'.format((t2-t1), (t2-t0)))
        #
        return(maps)

def smoothing(map, Nside):
    FWHM = 2.5*64/(Nside)*(np.pi/180) # need radians!
    smap = hp.sphtfunc.smoothing(map, fwhm=FWHM, iter=3)
    return(smap)

def Read_GreensMap(file):
    f = h5py.File(file, 'r')
    best_fit = np.asarray(f['/best_fit'])
    pixel_info = np.asarray(f['pixel_info'])
    samples = np.asarray(f['samples'])
    f.close()
    return(best_fit, pixel_info, samples)

def distance_array():
    """
    Make array of the Greens et.al. 2019 maps of 120 distance bins.
    Return distance in pc and distance modulus.
    """
    N = 120
    xmin = np.log10(63)
    xmax = np.log10(6.3e4)
    x = np.logspace(xmin, xmax, N)
    mu = 5*np.log10(x) - 5
    return(x, mu)

def construct_pixel_values(Nside, Npix, pix_ind, Nsides, samples, dist_ind=-1):
    """
    Construct an pixel array of reddening at a given distance.

    Parameters:
    -----------
    - Nside, integer.       The Healpix resolution number
    - Npix, integer.        The number of Healpix pixels in the map.
    - pixel_info, nd array. Arrays containing information of the pixels. need
                            nsides and pixel index
    - samples, nd array.    Array containg the sampling data for each sight line
                            and distance bins.
    - dist_ind, integer.    The index of the distance bin to evaluate.
                            Default is -1.

    Returns:
    -----------
    - pixel_val, array.     Array containing the pixel values of the map at the
                            evaluation distance.
    """

    pixel_val = np.empty(Npix, dtype='f8')
    pixel_val[:] = np.nan

    EBV_far_median = np.median(samples[:,:,dist_ind], axis=1)
    # fill in the sampled map
    #for Ns in np.unique(pixel_info['nside']):
    for Ns in np.unique(Nsides):
        # Get the indices of all pixels at current Nside level
        print('Get pixel values for Nside {} at distance index {}'.\
                format(Ns,dist_ind))

        ind = Nsides == Ns
        # get the reddening of each selected pixel
        pix_val_n = EBV_far_median[ind]

        # Determine nested index of each selected pixel in sampled map
        mult_factor = (Nside/Ns)**2

        pix_ind_n = pix_ind[ind]*mult_factor
        # write the selected pixels into the sampled map
        for offset in range((mult_factor)):
            pixel_val[pix_ind_n+offset] = pix_val_n[:]

    return(pixel_val)

def Greens_maps(file, ind, ind2=None, dist_ind=None):
    """
    Create sky map of reddening.

    Parameters:
    -----------
    - file, string.         Path to the file containing the Greens et.al. 2019 maps.
    - ind, list.            List of bin indices for which bins to store
    - dist_ind, int/list.   If int make one map. If list, make several maps with
                            the given distance indices. Index max=120
    """
    print('load Greens data')
    best_fit, pixel_info, samples = Read_GreensMap(file)
    Nsides = np.int16(pixel_info['nside'])
    pix_ind = np.int64(pixel_info['healpix_index'])
    x, mu = distance_array()

    bins = [np.min(ind), np.max(ind)]
    if ind2 is not None:
        bins.append(np.min(ind2))
        bins.append(np.max(ind2))

    print(x[bins])
    print(bins)
    Nbins = len(best_fit[0,:])

    Nside_max = np.max(pixel_info['nside'])
    Npix = hp.pixelfunc.nside2npix(Nside_max)
    prev_map = np.zeros(Npix)
    print(Nside_max, Npix)

    maps = []
    for dist_ind in bins:
        print('Bin number {}'.format(dist_ind))
        print('Make sky map at distance {}'.format(x[dist_ind]))
        if dist_ind == Nbins:
            dist_ind = Nbins-1
        pixel_val = construct_pixel_values(Nside_max, Npix, pix_ind, Nsides,\
                    samples, dist_ind)

        curr_map = pixel_val - prev_map
        maps.append(curr_map)

        prev_map = curr_map

        #hp.mollview(pixel_val, nest=True)
    #
    return(maps)

def Analyse_maps(fileplanck, fileGreens, Nside_in=0):
    t0 = time.time()
    # Load planck map
    m857 = load_planck_map(fileplanck)

    # Load Greens et.al.2019 map
    x, mu = distance_array()
    ind = np.arange(len(x))
    map_greens = Greens_maps(fileGreens, ind)

    t1 = time.time()
    print('Loading time: {}s'.format(t1-t0))
    print(np.shape(map_greens[1]))
    print(hp.pixelfunc.get_min_valid_nside(len(map_greens[1])))

    # get maps to same resolution, Nside 128
    map857 = fix_resolution(m857, Nside_in, 'RING')
    #smap857 = smoothing(map857, Nside_in)
    hp.mollview(np.log10(map857), title='Planck 857 GHz', unit='MJy/sr')
    plt.savefig('Figures/green_vs_planck/planck857_{}.png'.format(Nside_in))

    mapGreens = np.nan_to_num(map_greens[1])
    mapGreens = fix_resolution(mapGreens, Nside_in, 'NESTED')
    #mapGreens = np.nan_to_num(mapGreens)
    mapGreens_r = hp.pixelfunc.reorder(mapGreens, n2r=True)
    #smapGreens = smoothing(mapGreens_r, Nside_in)
    hp.mollview(mapGreens_r, title='Greens reddening full', unit='mag')
    plt.savefig('Figures/green_vs_planck/greens_{}.png'.format(Nside_in))
    # mask planck map to match greens
    Npix_new = hp.pixelfunc.nside2npix(Nside_in)
    newmap = np.zeros(Npix_new)
    cond = np.logical_and(map857, mapGreens_r > 0)

    newmap[cond] = map857[cond]
    hp.mollview((newmap), title='planck with greens mask')
    plt.savefig('Figures/green_vs_planck/masked_planck857_{}.png'.format(Nside_in))

    # Sampling:
    ind = np.where(newmap > 0)[0]
    Planck_model = newmap[ind]
    Greens_data = mapGreens_r[ind]
    #plt.figure()
    #plt.plot(Greens_data)
    print(np.min(Greens_data), len(Greens_data))
    #print(Greens_data)
    #plt.show()

    maxlike_a_g, a_g_array = sampler(Planck_model, Greens_data, 10.)
    print(maxlike_a_g, np.mean(a_g_array))
    #ind2 = np.where(newmap==0)[0]
    new_model = model_function(maxlike_a_g, mapGreens_r)
    diff_map = newmap - new_model

    hp.mollview(new_model, title='The estimated map')
    plt.savefig('Figures/green_vs_planck/Model_greens2planck_{}.png'.format(Nside_in))

    hp.mollview(diff_map, title='planck - new model')
    plt.savefig('Figures/green_vs_planck/diff_planck_model_{}.png'.format(Nside_in))

    plt.show()

def sampler(model, data, a_g0, Niter=1000):
    """
    Sampling routine to sample the convertion factor between Greens et.al 2019
    reddening map to Planck maps. Assuming the relationsip
    planck_map = a_g*greens_map + "some more".

    Parameters:
    -----------
    - model, array_like.    Planck map
    - data, array_like.     Greens et.al. 2019 map
    a_g0, scalar.           the convertion factor between Greens and Planck.

    Return:
    -----------
    """

    mean0 = np.mean(model)  # ??? how to define this one?
    sigma0 = np.std(model)
    print(mean0, sigma0)

    accept = np.zeros(Niter)
    params = np.zeros(Niter)

    model_func = partial(model_function, data=data)
    log_like = partial(logLikelihood, data=model)
    log_prior = partial(logPrior, mu=mean0, sigma=sigma0)
    print('----')
    # initialize
    curr_param = np.random.normal(mean0, sigma0) # > 0
    while curr_param < 0:
        curr_param = np.random.normal(mean0, sigma0)
        print('-', curr_param)

    curr_model = model_func(curr_param)
    curr_like = log_like(curr_model)
    curr_prior = log_prior(curr_param)
    print(curr_param, curr_like, curr_prior)
    cm1 = model_func(curr_param*2)
    cm2 = model_func(curr_param*5)
    cm3 = model_func(160.0)
    print('-', curr_param*2, log_like(cm1))
    print('-', curr_param*5, log_like(cm2))
    print('-', curr_param*10, log_like(cm3))
    #sys.exit()
    # sampling:
    counter = 0
    steplength = 1.
    maxlike = -10
    mean = mean0
    sigma = sigma0
    maxlike_param = curr_param
    print('-----')
    for i in range(Niter):
        prop_param = np.random.normal(mean, sigma*steplength)

        # MetropolisHastings step
        accept[i], params[i], maxlike, maxlike_param = mh_step(log_like,\
                                            log_prior, model_func, prop_param,\
                                            curr_param, curr_like, curr_prior,\
                                            maxlike, maxlike_param)

        #
        curr_model_new = model_func(params[i])
        curr_like = log_like(curr_model_new)
        curr_prior = log_prior(params[i])
        mean = params[i]
        if i%100==0:
            print(i, prop_param, maxlike, maxlike_param, curr_like, curr_prior)

        if accept[i] == True:
            counter += 1

        if (i+1)/50==0:
            if counter/float(i+1) < 0.2:
                steplength /= 2.
            elif counter/float(i+1) > 0.5:
                steplength *= 2.
            else:
                pass
        #
        if (i+1)%1000==0:
            sigma = np.std(params[:i])
    #
    print(counter, counter/float(Niter), maxlike)
    plt.figure()
    plt.hist(params, bins=30)
    plt.figure()
    plt.plot(params)
    return(maxlike_param, params)

def mh_step(log_like, log_prior, model_func, prop_param, curr_param, curr_like,\
            curr_prior, maxlike, maxlike_param):

    #
    prop_model = model_func(prop_param)
    prop_like = log_like(prop_model)
    prop_prior = log_prior(prop_param)

    # posterior analysis
    post_old = curr_like + curr_prior
    post_new = prop_like + prop_prior

    # acceptable testing:
    a = np.exp(post_new - post_old)
    draw = np.random.uniform(0,1)
    #print('-', prop_param, curr_param, prop_like, a)
    if (a > draw) and (a < np.inf):
        accept = True
        curr_param = prop_param
        if prop_like > maxlike:
            maxlike = prop_like
            maxlike_param = curr_param

    else:
        accept = False
        curr_param = curr_param
    #
    return(accept, curr_param, maxlike, maxlike_param)

def model_function(a_g, data, c=0.0):
    """
    The model to fit, planck_map = a_g*greens_map + c.
    """

    return(a_g*data + c)

def logLikelihood(model, data, sigma=10.):
    """
    Compute the log likelihood of the data goven the model, ln(P(d|m))

    Parameters:
    -----------
    Return:
    -----------
    """
    #print(np.sum(data-model), np.mean(data), np.mean(model))
    L_array = -0.5 * ((data - model)/sigma)**2
    L = np.sum(L_array)
    return(L)

def logPrior(param, mu, sigma):
    """
    Compute the log prior ln(P(m)).

    Parameters:
    -----------
    Return:
    -----------
    """
    pm = 0
    if param > 0:
        pm += -0.5*((param - mu)/sigma)**2
    else:
        pm = -30
    return(pm)

###################################
#         Function calls          #
###################################
planckfile = 'Data/HFI_SkyMap_857-field-Int_2048_R3.00_full.fits'
greensfile = 'Data/bayestar2019.h5'

Analyse_maps(planckfile, greensfile, 16)
