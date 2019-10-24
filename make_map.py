

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import h5py
import sys, os, time

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
        self.x = np.linspace(0, Rmax, 1000)
        
        self.Bin_ind = np.searchsorted(self.bin, self.dist)
        self.ind_sort = np.argsort(self.Bin_ind)
        self.Nbins = len(self.bin)
        self.order = 10
        
        

    def make_map(self):
        #if pixels or angles?
        
        fx_array = np.zeros((self.Npix, self.order+1))
        print(fx_array)
        for j in range(self.Npix):
            ind1 = np.where(self.pixpos == j)[0]
            print(hp.pixelfunc.pix2ang(self.Nside, j))
            Ag = self.Ag[ind1]
            dist = self.dist[ind1]
            
            # Find interpolation polynomial
            fit, fx = self.Ag_func(Ag, dist)
            
            fx_array[j, :] = fx
            print(fx_array[j, :])
            

            #plt.figure(1)
            #plt.plot(self.x, fx(self.x), '-b')
            #plt.show()
            sys.exit()

    def Ag_func(self, Ag, dist):
        """
        Find the mean extinction in bins along los and fit a polynomial to it
        """
        Ag_list = np.zeros(self.Nbins)
        R_list = np.zeros(self.Nbins)
        for i in range(self.Nbins-1):
            #print(i)
            ind2 = np.where((dist <= self.bin[i+1]) &(dist > self.bin[i]))[0]
    
            if len(ind2) == 0: # have more stars??
                if i == 0:
                    Ag_list[i+1] = 0.0
                else:
                    random = np.random.normal(Ag_list[i], 0.46)
                    Ag_list[i+1] = Ag_list[i] + abs(random)
                #    
                R_list[i+1] = (self.bin[i+1] + self.bin[i])/2
            
            else:
                Ag_list[i+1] = np.mean(Ag[ind2])
                R_list[i+1] = np.mean(dist[ind2])
            
            # test for increasing Agmean:
            if Ag_list[i+1] < Ag_list[i]:
                Ag_list[i+1] = Ag_list[i]
            else:
                pass
            #
        #plt.figure(1)
        #plt.plot(R_list, Ag_list, 'xk')
        
        # fit a curve to Ag_list
        fit = np.polyfit(R_list, Ag_list, self.order)
        fx = np.poly1d(fit)
        #print(fit)
        return(fit, fx)

    def extrapolate_extinction(self, fx, x):
        """
        Calculate the extinction of the full sky at a given distance
        """
        
#########################################

# calls:

#########################################

MM = Make_Map(128, 3000)
MM.make_map()
