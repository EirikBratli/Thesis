

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
        self.bin = np.array([0,500,1000,1500,2000,Rmax])
        print(self.bin, len(self.bin))
        self.x = np.arange(0, Rmax+1001, 1)
        
        self.Bin_ind = np.searchsorted(self.bin, self.dist)
        self.ind_sort = np.argsort(self.Bin_ind)
        self.Nbins = len(self.bin)+1
        self.order = 4
        
        # Arrays:
        self.fx_array = np.zeros((self.Npix, self.order+1))
        print(self.x)

        #####

    def call_make_map(self, one=False, R_slice=None):
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
        """
        data = pd.DataFrame({'pix': pixpos, 'bin': Bin_ind, 'Ag': Ag})
        print(data)
        data2 = data.groupby(['pix', 'bin'])['Ag'].mean()
        #Ag_data = data2.groupby('bin')['Ag'].mean()
        print(data2)
        index = np.array(data2.index)#.astype(int)
        index = np.reshape(index, (self.Npix, self.Nbins-1))
        #print(index)
        print(data2[index[11,:]])
        Ag_array = np.zeros((self.Npix, self.Nbins-1))
        
        print(Ag_array)
        """
        Ag_array = np.zeros((self.Npix, len(self.x)))
        counter = 0
        t0 = time.time()
        #"""
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
                print('time so far:', (t2-t0)/60)
                print('pixel:', pix, hp.pixelfunc.pix2ang(self.Nside, pix))
            #
        #    
        t1 = time.time()
        print('Time used:', (t1-t0)/60)
        print(counter)
        #"""
        return(Ag_array)
        #self.draw_map(Ag_array, eval_dist)


    def Ag_func(self, Ag, dist, Bin_ind):
        """
        Find the mean extinction in bins along los and fit a polynomial to it
        """
        Ag_list = np.zeros(self.Nbins)
        R_list = np.zeros(self.Nbins)
        counter = 0
        for i in range(self.Nbins-1):
            #print(i)
            ind2 = np.where((Bin_ind <= i+1) & (Bin_ind > i))[0]
            #print(i, len(ind2))
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
        #plt.figure(1)
        #plt.plot(R_list, Ag_list, marker='x')
        
        # fit a curve to Ag_list
        #fit = np.polyfit(R_list, Ag_list, self.order)
        #fx = np.poly1d(fit)

        #hx = self.func(R_list, 0.01, 1)
        #gx = self.g(R_list, 0.01, 0.1)
        
        popt, pcov2 = curve_fit(self.g, R_list, Ag_list,\
                                bounds=([0, -np.inf, 0], [1, np.inf, 1]))
        #popt, pcov = curve_fit(self.func, R_list, Ag_list)
        
        #print(popt)
        #print(self.func(10000, *popt), self.g(10000, *popt2))
        #plt.plot(self.x, fx(self.x), '-r', label='poly')
        #plt.plot(self.x, self.g(self.x, *popt), '-g', label='power law')
        #print(fit)
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
        
    def draw_map(self, Ag_map, dist, s=False):
        print('draw map of simulated extinction at distance {} pc'.format(dist))
        
        hp.mollview(Ag_map, title=r'$A_{{G,sim}}$ at distance {} pc'.format(dist),\
                    unit='mag')
        if s==True:
            plt.savefig('Figures/map_Ag_sim_r{}_Nside{}.png'.\
                        format(dist,self.Nside))
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
        if s==True:
            plt.savefig('Figures/Ag_powspec_r{}_Nside{}.png'.\
                        format(int(dist),self.Nside))
        else:
            plt.savefig('Figures/smoothFigs/smoothed_Ag_powspec_r{}_Nside{}.png'.\
                        format(int(dist), self.Nside))


    def func(self, x, a, b):
        return(a * np.log(b*x + 1))

    def g(self, x, a, b, c):
        return(b*x**a + c)

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

MM = Make_Map(16, 10000) # store Ag_maps for higher Nside? run once?
MM.call_make_map(one=True, R_slice=10000)
