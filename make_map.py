

import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import sys, os, time

from scipy.optimize import curve_fit

##########################

def Read_H5(file, name):

    f = h5py.File(file, 'r')
    data = np.asarray(f[name])
    f.close()
    return(data)

def PixelCoord(Nside, theta, phi):
    if np.min(theta) < 0:
        theta = np.pi/2 - theta
    else:
        pass
    #
    pixpos = hp.pixelfunc.ang2pix(Nside, theta, phi)
    return(pixpos)

def parallax2dist(p, p_err):
    # Add the zp to the parallax angle.
    p = p + 0.029
    dist = 1000./p

    dist_err = p_err/(p**2)
    return(dist, dist_err)

def dust_map():
    return(None)

##########################

class Make_Map():
    
    def __init__(self, Nside, Rmax):
        self.Nside = Nside
        self.Rmax = Rmax
        self.Npix = hp.nside2npix(Nside)
        
        print('Load data')
        self.parallax = Read_H5('Data/Parallax_v2.h5', 'parallax')
        self.parallax_error = Read_H5('Data/Parallax_error_v2.h5', 'parallax_error')
        self.Ag = Read_H5('Data/Extinction_v2.h5', 'a_g_val')
        self.Ag_low = Read_H5('Data/Extinction_lower_v2.h5', 'a_g_percentile_lower')
        self.Ag_upp = Read_H5('Data/Extinction_upper_v2.h5', 'a_g_percentile_upper')
        self.longitude = Read_H5('Data/gal_longitude_v2.h5', 'l')
        self.latitude = Read_H5('Data/gal_latitude_v2.h5', 'b')

        self.phi = self.longitude * np.pi/180
        self.theta = self.latitude * np.pi/180

        # get pixels:
        self.pixpos = PixelCoord(Nside, self.theta, self.phi)

        # get distance:                           
        self.dist, self.dist_err = parallax2dist(self.parallax, self.parallax_error)
        self.bin = np.arange(0, Rmax+10, 100)
        
        print(self.bin, len(self.bin))
        self.x = np.arange(0, 10001, 1)
        
        self.Bin_ind = np.searchsorted(self.bin, self.dist)
        self.ind_sort = np.argsort(self.Bin_ind)
        self.Nbins = len(self.bin)+1
        self.order = 4
        
        # Arrays:
        self.fx_array = np.zeros((self.Npix, self.order+1))
        print(self.x)

        #####

    def call_make_map(self, one=False, R_slice=None):
        # mcmc to find interpolations along a line of sight.
        # d_i = Ag(r_i) + n_i, index i is los
        # sample the coeff of Ag(r_i)

        print('Make simulated extinction maps up to {} pc with Nside {}'.\
              format(self.Rmax, self.Nside))

        if (R_slice != None) and (R_slice > self.Rmax):
            print('Evaluate extinction map at too high R_slice. R_slice <= Rmax')
            sys.exit()
        else:
            pass


        Ag_map = self.make_map()
        print(np.shape(Ag_map))
        
        if one == False:
            for r in self.bin:
                if r > 0:
                    
                    print(r)
                    ind = np.where(self.x == r)[0]
                    map = Ag_map[:, ind[0]]
                    self.draw_map(map, r)
                    smap = self.smoothin(Ag_map[:, ind[0]], self.Nside)
                    self.draw_map(smap, r, s=True)
                #
            #
        
        elif (one == True) and (R_slice != None):

            ind = np.where(self.x == R_slice)[0]
            map = Ag_map[:, ind[0]]
            self.draw_map(map, R_slice)
            smap = self.smoothing(map,self.Nside)
            self.draw_map(smap, R_slice, s=True)
            self.write_map(map, 'Ag_map_r{}_Nside{}.h5'.\
                           format(R_slice, self.Nside))

        else:
            sys.exit()
        #
        plt.show()


    def make_map(self):
        #if pixels or angles?
        
        # sort after pixel:
        pixpos = self.pixpos[self.ind_sort]
        Bin_ind = self.Bin_ind[self.ind_sort]
        dist = self.dist[self.ind_sort]
        Ag = self.Ag[self.ind_sort]

        Ag_array = np.zeros((self.Npix, len(self.x)))
        counter = 0
        t0 = time.time()
        for pix in range(self.Npix):
            ind1 = np.where(self.pixpos == pix)[0]
            Ag = self.Ag[ind1]
            dist = self.dist[ind1]
            Bin_ind = self.Bin_ind[ind1]
            # Find interpolation polynomial
            fit = self.Ag_func(Ag, dist, Bin_ind)
            #self.fx_array[j, :] = fit
            if fit[0] > 0.95:
                counter += 1

            Ag_array[pix] = self.extrapolate_extinction(self.x, fit)
            
            if pix%1000 == 0:
                t2 = time.time()
                print('time so far:', (t2-t0))
                print('pixel:', pix, hp.pixelfunc.pix2ang(self.Nside, pix))
            #
            sys.exit()
        #    
        t1 = time.time()
        print('Time used:', (t1-t0)/60)
        print(counter)
        return(Ag_array)

    def Ag_func(self, Ag, dist, Bin_ind):
        """
        Find the mean extinction in bins along los and fit a polynomial to it
        """
        Ag_list = np.zeros(self.Nbins)
        R_list = np.zeros(self.Nbins)
        counter = 0
        for i in range(self.Nbins-1):
            ind2 = np.where((Bin_ind <= i+1) & (Bin_ind > i))[0]
            if len(ind2) == 0: # have more stars??
                if i == 0:
                    Ag_list[i+1] = 0.0
                else:
                    random = np.random.normal(Ag_list[i], 0.46)
                    Ag_list[i+1] = Ag_list[i] #+ abs(random)
                #
                if i < max(Bin_ind):
                    R_list[i+1] = (self.bin[i+1] + self.bin[i])/2
                else:
                    R_list[i+1] = self.Rmax + 1000
            else:
                Ag_list[i+1] = np.mean(Ag[ind2])
                R_list[i+1] = np.mean(dist[ind2])
            
            # test for increasing Ag_mean: Accept one extremal, but ajust for
            # more than one outlier.
            if (i >= 1) and(Ag_list[i+1] < Ag_list[i]):
                counter += 1
                if counter > 1:
                    Ag_list[i+1] = Ag_list[i]
                else:
                    pass
            else:
                pass
            #
            #print(Ag_list[i+1])
        #

        print(Ag_list, len(Ag_list))        
        print(R_list, len(R_list))
        params, params_err = self.mcmc(R_list[1:], Ag_list[1:])

        plt.plot(R_list, Ag_list, 'xk')
        plt.plot(self.x, self.powlaw(self.x, params), '-b')
        plt.show()
        # fit a curve to Ag_list
        #fit = np.polyfit(R_list, Ag_list, self.order)
        #fx = np.poly1d(fit)

        #hx = self.func(R_list, 0.01, 1)
        #gx = self.g(R_list, 0.01, 0.1)
        
        popt, pcov2 = curve_fit(self.g, R_list, Ag_list,\
                                bounds=([0, -np.inf, 0], [1, np.inf, 1]))
        return(popt)

    def extrapolate_extinction(self, x, params):
        """
        Calculate the extinction of the full sky at a given distance
        Input:
        - fx, list/array.  List of the fitting parameters of pixel
        - x, scalar/array. The distance(s) to evaluate extinciton at
        """

        Ag = self.g(x, *params)
        return(Ag)
        
    def mcmc(self, x, data, Niter=10000, Nparams=2):
        sigma = 0.46
        mean = self.mean_array(Nparams) # np.array([0.1, 0.2])
        cov = self.cov_matrix(Nparams)  #np.array([[1, 0.8],[0.8,1]])
        params = np.zeros((Niter, Nparams))
        post = np.zeros(Niter)
        accept = np.zeros((Niter-1, Nparams))

        # initialize
        params[0,:] = np.random.multivariate_normal(mean, cov)
        pd_old = self.log_likelihood(data, self.powlaw(x, params[0,:]), sigma)
        pm_old = self.log_prior(params[0,:])
        print(pd_old, pm_old)
        post[0] = pd_old*pm_old 
        print(post[0])

        # sampling:
        steplength = 1
        counter = 0
        for i in range(Niter-1):
            # check tesplength in cov-matrix
            cov = steplength*cov
            if (i+1)%100==0:
                if (counter/(i+1) < 0.2):
                    cov = cov/2
                elif (counter/(i+1) > 0.5):
                    cov = 2*cov
                else:
                    pass
                #
            # draw parameters:
            params[i+1,:] = np.random.multivariate_normal(mean, cov)
            Ag = self.powlaw(x, params[i+1,:]) 
            post[i+1] = self.log_likelihood(data, Ag, sigma)\
                        + self.log_prior(params[i+1,:])
            #print(post[i+1])

            a = np.exp(post[i+1] - post[i])
            draw = np.random.uniform(0,1) 
            #print(i, a, draw)
            # checking:
            if (a > draw) and (a < np.inf):
                # accept
                #print(a)
                accept[i,:] = params[i+1,:]
                counter += 1
            else:
                accept[i,:] = params[i,:]
            
            #
            #
            pd_old = self.log_likelihood(data, Ag, sigma)
            pm_old = self.log_prior(params[i+1,:])
        #
        print(counter, counter/Niter)
        #print(accept)
        #plt.figure()
        #plt.hist(accept, bins=10)
        #plt.show()
        ab = np.array([np.mean(params[:,0]), np.mean(params[:,1])])
        ab_err = np.array([np.std(params[:,0]), np.std(params[:,1])])
        print(ab)
        return(ab, ab_err)

    def log_likelihood(self, data, Ag, sigma):
        p = 0
        for i in range(len(data)):
            p += (-((data[i] - Ag[i])/sigma)**2)
        return(p)

    def log_prior(self, args):
        if (args[0] >= 0) and (args[1] >= 0):
            return(0.0)
        else:
            return(-30)

    def mean_array(self, Nparams=2):
        return([0.15, 0.1])#([0.5/((i+1)**2) for i in range(Nparams)])
    
    def cov_matrix(self, Nparams=2):
        cov = np.eye(Nparams)
        for i in range(Nparams):
            for j in range(Nparams):
                if i != j:
                    cov[i,j] = 0.81
        return(cov)


    def draw_map(self, Ag_map, dist, s=False):
        print('draw map of simulated extinction at distance {} pc'.format(dist))
        
        hp.mollview(Ag_map, title=r'$A_{{G,sim}}$ at distance {} pc'.format(dist),\
                    unit='mag')
        if s==False:
            plt.savefig('Figures/map_Ag_sim_r{}_Nside{}.png'.\
                        format(dist, self.Nside))
        else:
            plt.savefig('Figures/smoothFigs/smooth_map_Agsim_r{}_Nside{}.png'.\
                        format(dist, self.Nside))
        #plt.show()

        # compute power spectrum for map
        cl, ell = self.power_spectrum(Ag_map)
        self.plot_cl(cl, ell, dist)
        #

    def smoothing(self, map, Nside):
        """
        Smooths the extinction map
        """
        FWHM = 2.5*(64/Nside) * (np.pi/180) # radians
        smap = hp.sphtfunc.smoothing(map, fwhm=FWHM)
        return(smap)

    def power_spectrum(self, map):
        print('Calculate power spectrum')
        cl = hp.sphtfunc.anafast(map)
        ell = np.arange(len(cl))
        return(cl, ell)
        
    def plot_cl(self, cl, ell, dist, s=False):
        plt.figure()
        plt.loglog(ell, ell*(ell+1)*cl)
        plt.xlabel(r'$l$')
        plt.ylabel(r'$l(l+1)C_l$')

        if s==False:
            plt.savefig('Figures/Ag_powspec_r{}_Nside{}.png'.\
                        format(int(dist), self.Nside))
        else:
            plt.savefig('Figures/smoothFigs/smoothed_Ag_powspec_r{}_Nside{}.png'.\
                        format(int(dist), self.Nside))


    def func(self, x, a, b):
        return(a * np.log(b*x + 1))

    def g(self, x, a, b, c):
        return(b*x**a + c)

    def powlaw(self, x, args):
        return(args[0]*x**args[1])

    def write_map(self, Ag_map, filename):
        #ind = np.where(self.x == R_slice)[0]
        #map = Ag_map[:, ind[0]]
        f = h5py.File('Data/' + filename, 'w')
        f.create_dataset('Ag', data=Ag_map.\
                         astype(dtype=np.float32))
        f.close()

        #hp.fitsfunc.write_map(filename, Ag_map,\
        #    coord='G',column_names='Ag',column_units='mag')
        ####




#########################################
#                calls                  #
#########################################

MM = Make_Map(4, 3000) # store Ag_maps for higher Nside? run once?
MM.call_make_map(one=True, R_slice=3000)
