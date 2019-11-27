"""
This program loads the maps from Greens et.al 2019, and then make sky maps at
different distances. Further make line of sight models for the reddening, convert
to extinction to compare with Gaia A_G estimation. Pan-STARRS1 has two suitable
passbands, g_P1 (500 nm) and r_P1 (650 nm).
"""


import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import h5py
import sys, glob, time
import pandas as pd


def main_los(Nside_gaia, greens_file):
    """
    Function to compare Greens etal 2019 with Gaia extinction estimate for sight
    lines.

    Parameters:
    - Nside_gaia, int. The Nside for the given Gaia map to load.
    - greens_file, string. File name of the Greens et.al 2019 map.
    """

    lat_los = [70, 90, 110]
    lat_los = [67, 70, 73]
    # make distance array of the 120 bins form 63 pc to 63 kpc,
    # index 68 give 3000 pc
    x_green, mu_green = distance_array()
    ix_max = np.max(np.where(x_green <= 3500)[0])

    # load gaia data
    print('Load and plot los extinction for Gaia')
    t0 = time.time()
    Ag, Ag_err, R, pixels, Npix, theta, phi = load_gaia(Nside_gaia)
    lon_gaia = phi * 180/np.pi
    lat_gaia = theta * 180/np.pi
    #hp.mollview(Ag[:,3])
    #plt.show()
    #sys.exit()
    # load greens data:
    print('Load and plot los extinction from Greens et.al. 2019')
    #Ebv, E_err, lon_green, lat_green = Greens_los_reddening(greens_file,\
    #                                                        lat_los, ix_max)
    t1 = time.time()
    print('Loading time: {} min'.format((t1-t0)/60.))

    # convert to extinction
    #Al = Ebv2Ag(Ebv)
    #Al_err = Ebv2Ag(E_err)

    # use delta = 2.5 for longitide = 90
    #lon_in = 90.
    delta = 2.5 # degrees

    # plotting
    print('Make line of sight extinction profile')
    """
    ind = np.where((lon_gaia >= lon_in - delta) & (lon_gaia < lon_in + delta))[0]
    plot_Ag_los(R[ind,:], Ag[ind,:], Ag_err[ind,:,:], lat_gaia[ind],\
                lon_gaia[ind], lat_los)

    plot_green_extinction(x_green[:ix_max], Al[:,:], Al_err[:,:],\
                        lon_green, lat_green, lat_los, lon_in)
    """

    #lon_list = [45., 90., 135.,180., 225.]
    lon_list = [220.,225.,230]
    for l in lon_list:
        print('--> Plot for longitide = {}'.format(l))
        ind = np.where((lon_gaia >= l - delta) & (lon_gaia < l + delta))[0]
        plot_Ag_los(R[ind,:], Ag[ind,:], Ag_err[ind,:,:], lat_gaia[ind],\
                    lon_gaia[ind], lat_los, l)

        #plot_green_extinction(x_green[:ix_max], Al[:,:], Al_err[:,:],\
        #                    lon_green, lat_green, lat_los, l)

    #########################
    #        end main       #
    #########################


def load_gaia(Nside):
    """
    Ag points are orientend after pixel number form 0 to Npix
    """
    path = 'Data/data/'
    Ag = Read_GaiaH5(path+'Ag_points_full_Nside{}.h5'.format(Nside), 'Ag')
    Ag_err = Read_GaiaH5(path+'Ag_err_points_full_Nside{}.h5'.format(Nside),\
                         'Ag_err low upp')
    R = Read_GaiaH5(path+'R_points_full_Nside{}.h5'.format(Nside), 'R')

    Npix = hp.nside2npix(Nside)
    pixels = np.arange(Npix)
    theta, phi = hp.pixelfunc.pix2ang(Nside, pixels)
    return(Ag, Ag_err, R, pixels, Npix, theta, phi)

def plot_Ag_los(R, Ag, Ag_err, theta, phi, lat_los, phi_in=90):
    """
    Plotting function for the data points of Gaia. Sight lines are of 5 sq.deg.
    """

    print('Plot Gaia los extinction')
    lat_los = [65,70,75]
    delta = 2.5
    color = ['b', 'r', 'g']#, 'm', 'purple']
    markers = ['x', '.', '^']#, '+', 'd']
    plt.figure('Greens vs Gaia extinction: l={}'.format(int(phi_in)))

    for i, b in enumerate(lat_los):
        ind = np.where((theta >= b - delta) & (theta < b + delta))[0]

        print('lat:', b)
        R_new = np.mean(R[ind,:], axis=0)
        Ag_new = np.mean(Ag[ind,:], axis=0)
        #err = np.std([Ag_new, Ag_err[ind,:,:]], axis=0)
        #err = np.percentile(Ag_err[ind,:,:], [16, 84], axis=0)
        err = np.mean(Ag_err[ind,:,:], axis=0)
        #print(np.shape(err))
        #print(err)
        plt.scatter(R_new, Ag_new, color=color[i], marker=markers[i])#,\
        #            label='b={}'.format(b))
        plt.errorbar(R_new, Ag_new, yerr=err, ecolor=color[i], linestyle='None')

    plt.xlabel(r'$r$, [pc]')
    plt.ylabel(r'extinction [mag]')
    plt.legend(loc=2)

def Greens_los_reddening(file, lat_los, ind_x):
    """
    Calculate the line of sight reddening/extinction of Greens et.al 2019 for
    specific sight lines. Then plot.

    Parameters:
    ----------
    - file, string. path and filename of the Greens map

    Return:
    ----------
    - reddening, array. Array with the median values of the samples
    - E_err, array.     The 16, 84 percentile of the median reddening
    - lon, array.       The longitude coordinates of each sight line
    - lat, array.       The Latitude coordinates of each sight line
    """

    print('Load Greens map')
    best_fit, pixel_info, samples = Read_GreensMap(file)
    best_fit = best_fit[:,:ind_x]
    samples = samples[:,:,:ind_x]

    # Get maximum Nside of the map, find Npix
    Nside_max = np.max(pixel_info['nside'])
    Npix = hp.nside2npix(Nside_max)

    # Convert to an usable format...
    #Nsides = pixel_info['nside']
    #pix_ind_n = pixel_info['healpix_index']
    Nsides = np.int16(pixel_info['nside'])
    pix_ind_n = np.int64(pixel_info['healpix_index'])

    # convert to ringed and get theta, phi:
    pix_ind_r, theta, phi = get_position(Nsides, pix_ind_n)
    lon = phi * 180/np.pi
    lat = theta * 180/np.pi

    # cumulative reddening along each sight line
    print('Calculate reddening')
    t2 = time.time()
    reddening = median(samples)#np.median(samples[:,:,:], axis=1)
    t3 = time.time()
    print('Time used in calculating reddening: {}s'.format(t3-t2))

    print('Calculate error in reddening')
    t4 = time.time()
    E_err = sigma_E(samples[:,:,:])
    t5 = time.time()
    print('Calculating reddening error in: {}s'.format(t5-t4))

    return(reddening, E_err, lon, lat)

def plot_green_extinction(x, A_array, A_error, lon, lat, lat_los=None, l_in=0):
    """
    Plot los reddening for sight lines of size 5 square degrees
    """
    if lat_los == None:
        return(None)
    elif len(lat_los) == 0:
        return(None)
    else:
        delta = 1.5 # deg
        print('Plot Grees los extinction')
        plt.figure('Greens vs Gaia extinction: l={}'.format(int(l_in)))
        colors = ['b', 'r', 'g']#, 'y', 'purple']
        path = 'Figures/Greens/'

        for j, b in enumerate(lat_los):
            print('Latitude:', b)
            ind = np.where((lat >= b - delta) & (lat < b + delta)\
                            & (lon >= l_in - delta) & (lon < l_in + delta))[0]

            # get the mean reddening for each sight line
            A_los = np.mean(A_array[ind, :], axis=0)
            #A_err_low = np.mean(A_err[0,ind,:], axis=0)
            #A_err_upp = np.mean(A_err[1,ind,:], axis=0)
            A_err = np.mean(A_error[ind,:], axis=0)
            A_err_low = A_los - A_err
            A_err_upp = A_los + A_err

            plt.plot(x, A_los, c=colors[j], label='l={}, b={}'.format(l_in, b))
            plt.plot(x, A_err_low, c=colors[j], linestyle=':')
            plt.plot(x, A_err_upp, c=colors[j], linestyle='--')

        #
        plt.legend(loc=2)
        plt.savefig(path + 'Compare_Extinction_lon{}_1sqdeg.png'.format(int(l_in)))


### Reading functions and helping function ###

def Read_GreensMap(file):
    f = h5py.File(file, 'r')
    best_fit = np.asarray(f['/best_fit'])
    pixel_info = np.asarray(f['pixel_info'])
    samples = np.asarray(f['samples'])
    f.close()
    return(best_fit, pixel_info, samples)

def Read_GaiaH5(file, name):
    f = h5py.File(file, 'r')
    data = np.asarray(f[name])
    f.close()
    return(data)

def get_position(Nsides, pixel_nest):
    pix_r = hp.nest2ring(Nsides, pixel_nest)
    theta, phi = hp.pixelfunc.pix2ang(Nsides, pix_r)
    return(pix_r, theta, phi)

def median(samples):
    # Compute the median values for each bin in each sight line.
    E = np.empty((len(samples[:,0,0]), len(samples[0,0,:])))
    i_prev = 0
    for i in range(100000, len(samples[:,0,0])+1, 100000):
        E[i_prev:i, :] = np.median(samples[i_prev:i,:,:], axis=1)
        i_prev = i
    return(E)

def sigma_E(samples):
    # compute the standard error of the samples,
    #return(np.percentile(samples, [16, 84], axis=1))
    sigma = np.empty((len(samples[:,0,0]), len(samples[0,0,:])))
    i_prev = 0
    for i in range(100000, len(samples[:,0,0])+1, 100000):
        sigma[i_prev:i,:] = np.std(samples[i_prev:i,:,:], axis=1)
        i_prev = i
    return(sigma)

def Ebv2Ag(reddening, Rvec=2.617):
    """
    Convert the reddening from Greens et.al to extinction similar to A_G, using
    the extinction vector given in Greens et.al. 2019. Default Rvec input is for
    pass band PS_r. Other passbands give:
    Rvec = [3.518, 2.617, 1.971, 1.549, 1.263, 0.7927, 0.469, 0.3026]
    """

    return(reddening*Rvec)

def distance_array():
    """
    Make array of the Greens et.al. 2019 maps of 120 distance bins.
    Return distance in pc and distance modulus.
    """
    N = 120
    xmin = np.log10(63)
    xmax = np.log10(6.3e4)
    x = np.logspace(xmin, xmax, N)
    #print(x[:70])
    mu = 5*np.log10(x) - 5
    return(x, mu)

############# Greens map functions ###############
def convert_mu(mu):
    """
    convert the distance modulus to parsec
    """
    # test if mu is +/- inf
    if len(mu) is not None:
        if_inf = np.isinf(mu)
        ind_ok = np.where(if_inf == False)
        mu = mu[ind_ok]
        d = 10**((mu + 5.0)/5.0)
    else:
        if_inf = np.isinf(mu)
        if if_inf == False:
            mu = mu
            d = 10**((mu + 5.0)/5.0)
        else:
            d = -1
    return(d)

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
        print('Get pixel values for Nside {} at distance index {}'.format(Ns,dist_ind))
        #ind = pixel_info['nside'] == Ns
        ind = Nsides == Ns
        # get the reddening of each selected pixel
        pix_val_n = EBV_far_median[ind]

        # Determine nested index of each selected pixel in sampled map
        mult_factor = (Nside/Ns)**2
        #pix_ind_n = pixel_info['healpix_index'][ind]*mult_factor
        pix_ind_n = pix_ind[ind]*mult_factor
        # write the selected pixels into the sampled map
        for offset in range((mult_factor)):
            pixel_val[pix_ind_n+offset] = pix_val_n[:]

    return(pixel_val)

def Greens_maps(file, ind, ind2=None, dist_ind=None):
    """
    Plot the sky map of reddening.

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
    #ind = np.where((x >= 100) & (x <= 300))[0]
    #if np.max(ind) < 60:
    #    ind_max = 2*np.max(ind)
    #    bins = [np.min(ind), np.max(ind), ind_max]
    #else:
    #bins = np.array([ind_min, ind_max])

    bins = [np.min(ind), np.max(ind)]
    if ind2 != None:
        bins.append(np.min(ind2))
        bins.append(np.max(ind2))

    print(x[bins])
    print(bins)
    Nbins = len(best_fit[0,:])

    #print(bins, Nbins)
    Nside_max = np.max(pixel_info['nside'])
    Npix = hp.pixelfunc.nside2npix(Nside_max)
    prev_map = np.zeros(Npix)
    print(Nside_max, Npix)

    #for dist_ind in range(0, Nbins+1, 10):
    maps = []
    for dist_ind in bins:
        print('Bin number {}'.format(dist_ind))
        #sys.exit()
        print('Make sky map at distance {}'.format(x[dist_ind]))
        if dist_ind == Nbins:
            dist_ind = Nbins-1
        pixel_val = construct_pixel_values(Nside_max, Npix, pix_ind, Nsides,\
                    samples, dist_ind)

        #pixel_val = hp.pixelfunc.nest2ring(Nside_max, pixel_val)

        curr_map = pixel_val - prev_map
        maps.append(curr_map)
        #curr_map = Ebv2Ag(curr_map)

        #greens_map = hp.pixelfunc.ud_grade(curr_map, 64)
        #hp.mollview(curr_map, nest=True, title='Extinction')

        #hp.mollview(np.log(curr_map), nest=True, title='log of Reddening',\
        #            rot=(130., 0.))

        prev_map = curr_map

    #
    return(maps)

def compare_Greens_Gaia(Nside, greens_file, xmin, xmax, xmin2=None, xmax2=None):
    """
    Compare the extinction maps of Gaia and Greens et.al 2019 at given distance
    intervals, up to two comaprison maps. We use the condition for Gaia Data
    to be 'Ag > 0.5, else Ag = 0', because of the large uncertainty.

    Parameters:
    -----------
    - Nside, integer.
    - greens_file, string.  Filename of the greens map
    - xmin, scalar.         Minimum distance to cut from
    - xmax, scalar.         Maximum distance to cut to.
    - xmin2, scalar.        If given, Minimum distance to the second cut
    - xmax2, scalar.        If given, maximum distance to the second cut

    Return:
    -----------
    """

    x, mu = distance_array()
    print('Compare extincion between Gaia and Greens.')
    print('Compare for distances between: {} pc and {} pc'.format(xmin, xmax))

    # Load gaia maps:
    Npix = hp.pixelfunc.nside2npix(Nside)
    Ag, Ag_err, R, pixels, Npix, theta, phi = load_gaia(Nside)
    Rmean = np.mean(R, axis=0)

    ind2 = np.where((x >= xmin) & (x <= xmax))[0]
    ind22 = np.where((x >= xmin2) & (x <= xmax2))[0]

    ind1 = np.where((Rmean >= xmin) & (Rmean <= xmax))[0]
    ind12 = np.where((Rmean >= xmin2) & (Rmean <= xmax2))[0]

    #cut = Ag/Ag_err > 3 recalc the error to error of mean, not mean of error.
    #Ag_diff = []
    #ind_min_gaia = []
    #ind_max_gaia = []
    #ind_min_green = []
    #ind_max_green = []

    """
    if isinstance(xmin, list) and isinstance(xmax, list):
        print('ok')
        xmin = np.sort(xmin)
        xmax = np.sort(xmax)

        for i in range(len(xmin)):
            ind_min_gaia.append(np.min(np.where(Rmean >= xmin[i])))
            ind_max_gaia.append(np.max(np.where(Rmean <= xmax[i])))
            ind_min_green.append(np.min(np.where(x >= xmin[i])))
            ind_max_green.append(np.max(np.where(x <= xmax[i])))

        for i in range(len(xmin)):
            diff = Ag[:,ind_max_gaia[i]] - Ag[:,ind_min_gaia[i]]
            Ag_diff.append(diff)
        #sys.exit()
    else:
        ind2 = np.where((x >= xmin) & (x <= xmax))[0]
        ind1 = np.where((Rmean >= xmin) & (Rmean <= xmax))[0]
        ind_min_gaia = np.min(ind1)
        ind_max_gaia = np.max(ind1)
        ind_min_green = np.min(ind2)
        ind_max_green = np.max(ind2)

        Ag_diff = Ag[:,ind_max_gaia] - Ag[:,ind_min_gaia]

    """
    # load Greens maps:
    Av_maps = Greens_maps(greens_file, ind2)
    print(len(Av_maps))

    Ag_diff = Ag[:,np.max(ind1)] - Ag[:,np.min(ind1)]
    if (xmin2 != None) and (xmax2 != None):
        Ag_diff2 = Ag[:, np.max(ind12)] - Ag[:, np.min(ind12)]
        R_bins = [np.min(ind1), np.max(ind1), np.min(ind12), np.max(ind12)]
        print(R_bins)
        print(Rmean[R_bins])
    else:
        R_bins = [np.min(ind1), np.max(ind1)]
        print(R_bins)
        print(Rmean[R_bins])
    #"""

    # down grade the Greens maps.
    dgrade_maps = []
    #Av_new = hp.pixelfunc.ud_grade(Av_maps, Nside, order_out='RING')
    c = 0
    for m in Av_maps:
        c += 1
        m_new = hp.pixelfunc.ud_grade(m, Nside, order_in='NESTED', order_out='NESTED')
        dgrade_maps.append(m_new)
        print(c-1)

    #
    plot_comparison(Ag_diff, dgrade_maps[1], xmin, xmax, Nside)
    #plot_comparison(Ag_diff2, dgrade_maps[3], xmin2, xmax2, Nside)


def plot_comparison(Ag, Av, xmin, xmax, Nside):
    Npix = hp.nside2npix(Nside)
    Ag_new = np.zeros(Npix)
    print(np.shape(Ag_new), np.shape(Ag))
    Ag_new[np.where(Ag > 0.5)] = Ag[np.where(Ag > 0.5)]
    Ag_new = hp.pixelfunc.reorder(Ag_new, r2n=True)

    hp.mollview(Av, nest=True, title='Greens, {}pc to {}pc'.\
                format(int(xmin), int(xmax)))
    plt.savefig('Figures/Greens/Compare/dgrade_greensmap_r{}to{}_Nside{}.png'.\
                format(int(xmin), int(xmax), Nside))

    hp.mollview(Ag_new, nest=True, title='Gaia, {}pc to {}pc'.\
                format(int(xmin),int(xmax)))
    plt.savefig('Figures/Greens/Compare/Ag_map_r{}to{}_over05_Nside{}.png'.\
                format(int(xmin), int(xmax), Nside))

    #Ag_new = hp.pixelfunc.reorder(Ag_new, r2n=True)
    newmap = np.zeros(Npix)
    cond = np.logical_and(Av, Ag_new > 0.5)
    newmap[cond] = Av[cond]

    hp.mollview(newmap, nest=True, title='Comparison, {}pc to {}pc'.\
                format(int(xmin), int(xmax)))
    plt.savefig('Figures/Greens/Compare/Gaia_vs_Greens_r{}to{}pc_Nside{}.png'.\
                format(int(xmin), int(xmax), Nside))



######################
#   Function calls   #
######################

file = 'Data/bayestar2019.h5'

#plt.scatter(R_array[0,0,:], Ag_array[0,0,:])
#Greens_maps(file)
#distance_array()
#Greens_los_reddening(file)
#x, mu = distance_array()
#print(x)


main_los(16, file)

#compare_Greens_Gaia(16, file, 100, 400)
#compare_Greens_Gaia(16, file, 100, 400, xmin2=1000, xmax2=2000)
#compare_Greens_Gaia(16, file, [100,500,1000], [600,1500,2000])




"""
Ag_file = 'Data/Ag_los_points.npy'
Ag_array = np.load(Ag_file)
R_array = np.load('Data/R_los.npy')
print(np.shape(Ag_array))

f1 = h5py.File('Data/data/Ag_points_full_Nside16.h5', 'r')
f2 = h5py.File('Data/data/Ag_err_points_full_Nside16.h5', 'r')
f3 = h5py.File('Data/data/R_points_full_Nside16.h5', 'r')
Ag = np.asarray(f1['Ag'])
Ag_err = np.asarray(f2['Ag_err low upp'])
R = np.asarray(f3['R'])
f1.close()
f2.close()
f3.close()
print(np.shape(Ag), np.shape(Ag_err))

plt.scatter(R[9,:], Ag[9,:])
plt.errorbar(R[9,:], Ag[9,:], yerr=Ag_err[9,:,:], ecolor='grey')
"""

plt.show()




plt.show()
