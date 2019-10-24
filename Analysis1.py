#........1.........2.........3.........4.........5.........6.........7.........8

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd
import h5py
import sys, os, time

#import numba as nb
from scipy import interpolate
from astropy import units as u
from astropy.coordinates import SkyCoord, Longitude, Latitude

from star import Star


###############################################
"""
Program to do simple analysis of extinction from Gaia DR2. Make distributions 
due to distance for different line of sights.

"""


def Read_H5(file, name):
    """
    Read a .h5 file and return the values in an array
    
    Input:
    - file, string. Name of the file to read
    - name, string. Parameter name in the file
    
    Return:
    - data, array. The data of the read file
    """

    f = h5py.File(file, 'r')
    data = np.asarray(f[name])
    f.close()
    return(data)

def PixelCoord(Nside, ra, dec):
    """
    Make file with Healpix coordinates for the given Nside. Also return the 
    array with Healpix coordinates.
    
    Input:
    - Nside, int. The Nside to make Healpix coordinates from.
    
    Return:
    - pixpos, array. Array with pixel coordinates in Healpix
    """
    
    #ra = Read_H5('Data/RightAscension_v2.h5', 'ra')
    #dec = Read_H5('Data/Declination_v2.h5', 'dec')
    #print(ra, dec)
    #c = SkyCoord(l=ra*u.radian, b=dec*u.radian, frame='galactic')
    
    #c.galactic
    #print(c)
    #print(Longitude(ra, unit=u.degree))
    dec = np.pi/2.0 - dec
    Nstars = len(ra)
    pixpos = hp.pixelfunc.ang2pix(Nside, dec, ra)
    
    #f = h5py.File('Data/SPC_Nside_{}.h5'.format(Nside), 'w')
    #f.create_data_set('Healpix coordinates', data=pixpos.astype(int))
    #f.close()
    
    return(pixpos)

def xyz_position(dist, ra, dec):
    """
    Get the euclidian coordiantes of the stars. Can be nice if stars are not to 
    far off?
    Input:
    - dist, array. The radial distance to the stars, in parsec
    - ra, array. The right ascension angle to the stars, in radians
    - dec, array. The declination angle to the stars, in radians
    Return:
    - xyz, array. A (N, 3) array with the euclidian coordinates to the stars.
    """
    #print(np.min(dec), np.max(dec))
    if np.min(dec) > 0:
        pass
    else:
        dec = np.pi/2 - dec
    #print(np.min(dec), np.max(dec))

    xyz_coord = hp.pixelfunc.ang2vec(dec, ra)
    xyz = np.zeros((len(ra), 3))
    for i in range(3):
        xyz[:, i] = dist * xyz_coord[:, i]
    return(xyz)



def parallax2dist(p, p_err):
    # Add the zp to the parallax angle, use also in the error estimate?
    p = p + 0.029
    dist = 1000./p

    dist_err = p_err/(p**2)
    return(dist, dist_err)


def Cel2Gal(ra, dec):
    """
    Transform Ra and Dec to galactic coordinates from celestial.
    """

    deg = np.pi/180
    """
    el0 = 32.93191857 * deg
    r0 = 282.859481208 * deg
    d0 = 62.8717488056 * deg

    l1 = np.cos(dec)*np.cos(ra - r0)
    l2 = np.sin(dec)*np.sin(d0) + np.cos(dec)*np.cos(ra-r0)*np.cos(d0)
    l3 = np.sin(dec)*np.cos(d0) - np.cos(dec)*np.sin(ra-r0)*np.sin(d0)
    """
    Ag = np.array([[-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],\
                   [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],\
                   [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669]])
    #print(Ag[1,0], Ag[0,1])
    #print(Ag)
    r1 = np.cos(ra)*np.cos(dec)
    r2 = np.cos(ra)*np.sin(dec)
    r3 = np.sin(dec)

    l1 = r1*Ag[0,0] + r2*Ag[0,1] + r3*Ag[0,2]
    l2 = r1*Ag[1,0] + r2*Ag[1,1] + r3*Ag[1,2]
    l3 = r1*Ag[2,0] + r2*Ag[2,1] + r3*Ag[2,2]
    #print(l1, l2, l3)
    b = np.arctan2(l3, (np.sqrt(l1**2 + l2**2)))
    l = (np.arctan2(l1, l2))
    #print(b, l)
    return(l, b)
#l, b = Cel2Gal(0,0)
#print(np.arctan2(0,1))
#sys.exit()    

####

class Extinction():
    """
    Contain functions:
    - Extinction_map()
    - simple_map(), map of all stars
    - pixAnalysis()
    - plotSightline()
    """
    def __init__(self, Nside, Nmaps, Rmin=0.1, Rmax=3000):
        print('Nside={}'.format(Nside))
        
        self.Nside = Nside
        self.Nmaps = Nmaps
        self.Rmin = Rmin
        self.Rmax = Rmax

        print('Get data')
        # Read data from file:
        self.ra = Read_H5('Data/RightAscension_v2.h5', 'ra')
        self.dec = Read_H5('Data/Declination_v2.h5', 'dec')        
        self.parallax = Read_H5('Data/Parallax_v2.h5', 'parallax')
        self.parallax_error = Read_H5('Data/Parallax_error_v2.h5', 'parallax_error')
        self.Ag = Read_H5('Data/Extinction_v2.h5', 'a_g_val')
        self.Ag_low = Read_H5('Data/Extinction_lower_v2.h5', 'a_g_percentile_lower')
        self.Ag_upp = Read_H5('Data/Extinction_upper_v2.h5', 'a_g_percentile_upper')
        self.CE = Read_H5('Data/Reddening_v2.h5','e_bp_min_rp_val')
        self.CE_low = Read_H5('Data/Reddening_lower_v2.h5','e_bp_min_rp_percentile_lower')
        self.CE_upp = Read_H5('Data/Reddening_upper_v2.h5','e_bp_min_rp_percentile_upper')
        self.Gmag = Read_H5('Data/Mean_mag_G_v2.h5', 'phot_g_mean_mag')
        self.ind0 = Read_H5('Data/Index_v2.h5', 'indices')
        self.longitude = Read_H5('Data/gal_longitude_v2.h5', 'l')
        self.latitude = Read_H5('Data/gal_latitude_v2.h5', 'b')

        self.Nstars = len(self.ra)
        print('Total length of data: {}'.format(self.Nstars))
        # get pixels:
        #print(self.ra[:10])
        #print(np.min(self.dec), np.max(self.dec))
        print('Get Healpix coordinates for the stars')
        self.pixpos = PixelCoord(Nside, self.ra, self.dec)
        l = self.longitude*np.pi/180.
        b = self.latitude*np.pi/180.
        self.gal_pixpos = PixelCoord(Nside, l, b)
        self.dec = np.pi/2 - self.dec 
        #print(self.pixpos)        
        #print(self.ra*180/np.pi)
        #print(self.dec*180/np.pi)
        # get distance:
        self.dist, self.dist_err = parallax2dist(self.parallax, self.parallax_error)
        
        self.bin = np.linspace(Rmin, Rmax, Nmaps+1)
        self.Npix = hp.nside2npix(Nside)
        self.Nbins = Nmaps + 1
        self.x = np.linspace(0, Rmax, 1001)
        self.map, b = np.histogram(self.pixpos, bins=self.Npix)
        #print(map)
        
        #hp.mollview(pp)
        #hp.mollview(self.map, coord=['C','G'])
        self.m = np.argwhere(self.map > 0)
        self.unique_pixel = np.unique(self.pixpos)

        #self.xyz = xyz_position(self.dist, self.ra, self.dec)
        #t0 = time.time()
        #r1 = self.get_GalCoord(self.ra, self.dec)
        #r2 = self.get_GalCoord(self.ra[:2], self.dec[:2])
        #t1 = time.time()
        #print(t1-t0)
        print('-----------')

    
    def get_GalCoord(self, alpha, delta):
        """
        Coordinate transformation from ICRS to galactic xyz coord, following 
        Gaia DR2 documentation (3.1.7). Then get healpix angles from this. 
        Unfortunately only one star at the time :/
        Input:
        - alpha. scalar. Right ascension angle in deg
        - delta, scalar. Declination angle in deg
        Return:
        - alpha. scalar. longitude in deg
        - delta. scalar. latitude in deg 
        """
        #print(np.shape(alpha))
        Ag = np.empty((3,3,len(alpha)))
        Ag1 = np.array([[-0.0548755604162154, -0.8734370902348850, \
                         -0.4838350155487132],\
            [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],\
            [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669]])
        #print(np.shape(Ag))
        for i in range(3):
            for j in range(3):
                Ag[i,j,:] = Ag1[i,j]
        #print(Ag[:,:,0])
    
        r = np.array([[np.cos(alpha)*np.cos(delta)],\
                      [np.sin(alpha)*np.cos(delta)],\
                      [np.sin(delta)]])
        #print(r[:,:,0])
        #print(np.einsum('ijk, jlk->ilk', Ag, r))  
        rgal = np.einsum('ijk, jlk->ilk', Ag, r)
        alpha1, delta1 = hp.pixelfunc.vec2ang(rgal, lonlat=True)
    
        return(alpha1, delta1)
        #return(rgal)

    def xyzExtinction(self, Rmax=3000, box_size=100):
        """
        Make a 3d map of the extinction of the stars in Euclidian coordinates.
        """

        #ind = np.where(self.dist < 100)[0]
        xyz = xyz_position(self.dist, self.ra, self.dec)
        
        x = np.arange(-Rmax, Rmax+10, box_size)
        y = np.arange(-Rmax, Rmax+10, box_size)
        X, Y = np.meshgrid(x, y)
        
        #print(xyz_new)
        print(x, len(x))
        
        for k in range(3):
            ind2 = np.where((self.dist < Rmax) & (xyz[:,k] >= 0) & (xyz[:,k] <= 10))[0]
            print(len(ind2))
            xyz_new = xyz[ind2, :]
            Ag = self.Ag[ind2]
            Ag_box = np.zeros((len(x)-1, len(y)-1))
            
            if k == 0:
                k0 = 1
                k1 = 2
            elif k == 1:
                k0 = 0
                k1 = 2
            else:
                k0 = 0
                k1 = 1
        
            for i in range(len(x)-1):
                for j in range(len(y)-1):
                    ind3 = np.where((xyz_new[:,k0]>x[i]) & (xyz_new[:,k0] < x[i+1]) \
                                & (xyz_new[:,k1] > y[j]) & (xyz_new[:,k1] < y[j+1]))[0]
                    #print(xyz_new[ind3])
                    #print(Ag[ind3])
                    if len(ind3) > 0:
                        Ag_mean = np.mean(Ag[ind3])
                        r_mean = np.mean(self.dist[ind3])
                    else:
                        Ag_mean = 0.0
                        r_mean = 1.0
                    #
                    
                    #print(i, j, x[i], y[j], Ag_mean)
                    Ag_box[i,j] = Ag_mean
            #
            
            #rint('i=', i)
            #
            plt.figure(k)
            #plt.plot(xyz_new[:,0], xyz_new[:,1], '.b')
            #plt.plot(xyz_new[ind3,0], xyz_new[ind3,1], '.r')
            #print(Ag_box, np.shape(Ag_box))
            plt.pcolormesh(X,Y, Ag_box)
            cb = plt.colorbar()
        
        
    
    def Extinction_mapslice(self, slices1=4, A=False, CE=False):
        """
        Calculate the extinction in cones/cake pieces as function for right 
        ascension and/or declination. Then be able to make 2D maps of extinction 
        as 'plain' slices trough the sphere/cube with stars.
        Input:
        - a, scalar. The half angle of the sector step, in degrees.
        - slices1, integer. The number of slices in RA
        - Rmin, scalar.
        - Rmax, scalar.
        """
        if A == True:
            print('Map extinction')
            Fx = self.Ag
            Fx_low = self.Ag_low
            Fx_upp = self.Ag_upp
        elif CE == True:
            print('Map reddening')
            Fx = self.CE
            Fx_low = self.CE_low
            Fx_upp = self.CE_upp
        elif (CE==True) and (A==True):
            print('Use either extinction (A=True) or reddening (CE=True)')
            sys.exit()
        else:
            print('Use either extinction (A=True) or reddening (CE=True)')
            sys.exit()
        
        delta = 360/(slices1-1)
        phi = np.arange(0.0, 360.0, delta)
        #phi = np.linspace(0.0, 360.0, slices1)
        #theta = phi[:int(slices1/2)+1]
        #dphi = (phi[1]-phi[0])
        
        theta = np.arange(0.0, 180, delta)
        Nintervals = (self.Rmax - self.Rmin)/self.Nmaps
        #print(np.max(radeg), np.min(radeg))
        #print(np.max(decdeg), np.min(decdeg))
        print(phi, len(phi))
        print(theta)
        print(delta)
        
        #Ag_list = np.zeros((slices1, len(theta), self.Nmaps))
        
        #sys.exit()
        Ag_list = self.piece_of_cake(phi, theta, delta)
        #print(Ag_list[0])
        #print((Ag_list[0][0]), (Ag_list[0][1]))
        print('-------------')
        #for i in range(len(phi)):
        #    plt.plot(self.bin, Ag_list[i,0,:])
        #self.Plot_2D_polar(phi, Ag_list) # not ok

        # end Extinction_mapslice
    
    def piece_of_cake(self, phi, theta, delta):
        """
        Calculate the mean extinction in a slice/cone sector. 
        Input:
        - phi, array. 
        - theta, array.
        - delta, scalar.
        Return:
        - Ag_list, nd array
        """
        radeg = self.ra*180.0/np.pi
        decdeg = self.dec*180.0/np.pi + 90.0

        Ag_list = np.zeros((len(phi), len(theta), self.Nbins))
        print(np.shape(Ag_list))
        t0 = time.time()
        for i in range(len(phi)): # right ascension loop
            
            for j in range(len(theta)): # declination loop
                ind2 = np.where((radeg > phi[i]) & (radeg <= phi[i]+delta) &\
                                (decdeg > theta[j]) & (decdeg <= theta[j]+delta))[0]
                if len(ind2) > 0:
                    rad = self.dist[ind2]
                    Ag = self.Ag[ind2]
                    #print(i, phi[i], theta[j], len(ind2), np.mean(Ag))

                    for k in range(self.Nmaps): # distance loop
                        ind3 = np.where((rad > self.bin[k]) & (rad <= self.bin[k+1]))[0]
                        #Ag_mean2 = np.mean(Ag2[ind2])
                        #print(self.bin[k+1], Ag_mean2)
                        Ag_list[i, j, k+1] = np.mean(Ag[ind3])
                        print(Ag_list[i,j,k+1])
                    #end distance loop 
                else:
                    Ag = 0.0
                    Ag_list[i, j, :] = 0.0
                # end if
                #print('Line of sight: ra=({},{}), dec=({},{})'.format(phi[i], phi[i]+delta,\
                #                                                      theta[j],theta[j]+delta))
                #print(np.mean(Ag_list[i, j, :]))
            # end declination loop
            t1 = time.time()
            print('time so fra:', t1-t0)
            print('RA = ({},{})'.format(phi[i], phi[i]+delta))
            #print(Ag_list[i, :, :])
        
        # end right ascension loop
        
        return(Ag_list)
        
        

    def Extinction_map(self, A=False, CE=False):
        """
        Make maps of the extinction as function of distance and position.
        Use distance intervals to make map layers.     
        """
        if A == True:
            print('Map extinction')
            Fx = self.Ag
            Fx_low = self.Ag_low
            Fx_upp = self.Ag_upp
        elif CE == True:
            print('Map reddening')
            Fx = self.CE
            Fx_low = self.CE_low
            Fx_upp = self.CE_upp
        elif (CE==True) and (A==True):
            print('Use either extinction (A=True) or reddening (CE=True)')
            sys.exit()
        else:
            print('Use either extinction (A=True) or reddening (CE=True)')
            sys.exit()

        ##
        # make arrays:
        Fx_list = np.zeros((self.Nbins, len(self.m)))
        Fx_los = np.zeros((self.Nbins, len(self.m)))
        print('Number of maps={}, from distance r={} pc to r={} pc'.\
          format(self.Nmaps, self.Rmin, self.Rmax))

        print(self.bin)
        print(np.max(self.dist), np.min(self.dist))
        print(np.shape(Fx_list), self.Npix)
        maps = []
        # loop over distance:
        t0 = time.time()
        for i in range(self.Nmaps):
            t00 = time.time()
            ind1 = np.where((self.dist > self.bin[i]) & (self.dist <= self.bin[i+1]))[0]
            print('--> Bin {} at distance {} pc'.format(i+1, self.bin[i+1]))
            pixel = self.pixpos[ind1]
            fx = Fx[ind1]
            fx_l = Fx_low[ind1]
            fx_u = Fx_upp[ind1]
            print(len(ind1))
            
            Fx_list, Ag_in_pix = self.pixAnalysis(Fx_list, fx, pixel, i)
            Fx_los[i+1, :] = np.sum(Fx_list, axis=0)
            #print(Fx_list[i+1,:])
            
            m1, b = np.histogram(pixel, self.Npix, weights=Ag_in_pix)
            
            #m1b = m1/np.max(self.m)
            #hp.mollview(m1, coord=['C','G'], nest=False, title='Stars from {}pc to {}pc'.\
            #    format(self.bin[i], self.bin[i+1]), unit='Nstars')
            print('Mean extinction in the map [mag/pix]:', np.sum(Ag_in_pix)/self.Npix)
            
            #self.plotMaps_total(pixel, fx, i)
            maps.append(m1)
            t11 = time.time()
            print('Iteration time: {} s'.format(t11-t00))
        # end map loop
        t1 = time.time()
        print('Computation time: {} s'.format(t1-t0))
        print(np.shape(maps))
        
        # Plots:    
        self.plot_Ag_Maps(maps)
        
        # plot per pixel
        #self.Plot_los(Fx_list, Fx_los, 'Extinction $A_G$')
        
        # End Extinction_map


    def pixAnalysis(self, Fx_list, fx, pixel, i):
        """
        Find the mean extinction in each pixel. Apply the mean to a pixel and to
        a list for further analysis.
        Input:
        - Fx_list, sequence. nd array to update with the mean extinction
        - fx, array. The array with extinction values in a distance interval
        - pixel, array. The pixels with stars in the distance interval
        - i, integer. The current bin index
        Return:
        - Fx_list, array. The updated array
        - Ag_in_pix, array. List with the mean extinction applied to its pixel
        """
        
        Ag_in_pix = np.zeros(len(pixel))
        for j, p in enumerate(self.unique_pixel):
            ind2 = np.where(pixel == p)[0]
            
            if len(ind2) > 0:
                fx_temp = fx[ind2]
                fx_mean = np.mean(fx_temp)
                Ag_in_pix[ind2] = fx_mean/len(ind2)
                #print(fx_mean)
            else:
                fx_mean = 0.0
                Ag_in_pix[ind2] = 0.0
            # end if
            #print(j, p, fx_mean, len(ind2))
            
            Fx_list[i+1, j] = fx_mean
            #if len(ind2) < 2:
            #    print('Few stars (n*={}) in pixel:'.format(len(ind2)), p)
        # end j loop

        #Fx_los[i+1, :] = np.sum(Fx_list, axis=0)
        return(Fx_list, Ag_in_pix)

    def get_relStars(self, Rmax):
        ind = np.where(self.dist < Rmax)[0]

        return(self.dist[ind], self.Ag[ind], self.ra[ind], self.dec[ind],\
               self.pixpos[ind])

        

    def ExtinctionMap2(self):
        """
        """
        #
        ii = np.where(self.dist < self.Rmax)[0]
        #dist = self.dist[ii]
        print(len(ii))
        #sys.exit()
        Nstars = len(ii)
        #plt.hist(self.dist[ii], bins=100)
        #plt.show()
        #sys.exit()

        # convert ra, dec to galactic coord: latitude, longitude (in deg)
        #l, b = self.get_GalCoord(self.ra, self.dec) # dont work properly
        #alpha = l*np.pi/180  # 0, 2pi
        #delta = np.pi/2 - b*np.pi/180  # -pi/2, pi/2
        
        # get pixelcoord for more than 1 Nside?

        # define arrays:
        star_out = np.zeros((self.Nstars, 4))
        t1 = time.time()
        Bin_ind = np.searchsorted(self.bin, self.dist)
        print(Bin_ind, np.max(Bin_ind))
        star_in = np.array([Bin_ind, self.dist, self.ra, self.dec, self.Ag])
        ind = np.argsort(Bin_ind)

        df = pd.DataFrame(star_in[:, ind].T, columns=['bin','r','ra','dec','Ag'])
        print(df)
        """
        # get map layers
        Ag_tot,Ag_mean = Star(self.Nside, self.Rmax, star_in[:,ind]).get_Ag_tot()
        t2 = time.time()
        print('Calculation time:', t2-t1)
        self.plot_Ag_Maps(Ag_mean)
        #"""
        # get map slices
        #xyz = xyz_position(self.dist, self.ra, self.dec)
        Star(self.Nside, self.Rmax, star_in[:,ind]).get_slice_maps(90, 18)
        #Star(self.Nside, self.Rmax, star_in[:, ind]).slicemap(36, 72)

        """
        sys.exit()
        t1 = time.time()
        for i in range(100000):#(self.Nstars):
            #aa = Star(self.Nside, self.Rmax, a[:, i]).star_info(array[i,:])
            # find a Nside
            #Ns = self.Nside
            # call to star class to get Ag in a pixel
            # Ag[p,r] = Ag[p,r] + star  # 
            # n[p,r] = n[p,r]+1   # if (n > some number) in a bin: increase Nside
            if i%1000==0:
                t3 = time.time()
                print('i=', i, t3-t1, aa)
            #pix[i] = self.pixpos[i]
        #
        t2 = time.time()
        print('time: ', t2-t1)
        # where(n>0): Ag = Ag/n
        """
        ######

    def Ag_func(self, Ndraws):
        """
        Compute the 'mean' extinction along a sight line, to be able to parameterise it
        and look for similarities. 
        Sight lines (ra +/- 2.5, dec +/- 2.5): (0,0), (0,45), (0,90), (0,135), (0,180),
        (90,90), (90,45), (90,135), (180,45), (180,90), (180,135),
        (270, 45), (270, 90), (270,135)
        pluss some random drawn directions. Pluss check that Ag(0,0) == Ag(180, 0) etc
        """
        
        # Sight lines
        los_ra = np.arange(0, 361, 45)#[0, 45, 90, 135, 180, 225, 270, 315]
        #los_dec = [10, 45, 60, 75, 90, 105, 120, 135, 170]
        los_dec = [60, 75, 80, 85, 90, 95, 100, 105, 120]
        delta_ang = 2.5                   # degrees
    
        # rotate to galactic coordinates
        rot = hp.rotator.Rotator(coord=['C', 'G'])
        #dec, ra = rot(self.dec, self.ra)  # in radians
        #ra = ra + np.pi
        #l, b = Cel2Gal(self.ra, self.dec)
        l_gal = self.longitude*np.pi/180
        b_gal = self.latitude*np.pi/180
        b_gal = np.pi/2 - b_gal
        #dec, ra = rot(dec, ra)
        print(np.min(self.ra), np.max(self.ra))
        print(np.min(self.dec), np.max(self.dec))
        print(np.min(l_gal), np.max(l_gal))
        print(np.min(b_gal), np.max(b_gal))
        
        
        #plt.figure('hist star distribution of direction')
        #plt.plot(l, b, '.b')
        #plt.hist(b_gal*180/np.pi, bins=180)
        #plt.hist(l_gal*180/np.pi, bins=360)
        #plt.show()        

        # Get pixels
        pixels = self.pixpos
        #ra = (self.ra + np.pi)*180/np.pi       # convert to degrees
        ra = self.ra * 180/np.pi
        dec = self.dec*180/np.pi               # convert to degrees

        # call curve fitting of los Ag
        #x, p, dAg = self.Ag_los_fitcurve(pixel_in=pixels[0])
        
        #"""
        # call funciton for los in random directions
        #self.call_Draw_pixel(Ndraws)

        # call Ag_los       
        self.call_Ag_los(los_ra, los_dec)

        #

    def call_Draw_pixel(self, Ndraws):
        # fit curve in random directions:
        # input: Ndraws, Ag_array=size(Ndraws, Nbins), R_array(Ndraws, Nbins)
        Ag_array = np.zeros((Ndraws, self.Nbins+1))
        R_array = np.zeros((Ndraws, self.Nbins+1))
        Ag_poly, Ag_der, coord = self.Draw_pixel(Ndraws, Ag_array, R_array)
        #print(np.shape(Ag_poly))
        print(Ag_array)
        #print(np.shape(Ag_array))
        
        plt.figure('random')
        for i in range(Ndraws):
            
            c = coord[i]
            plt.plot(self.x, Ag_poly[i,:], label='{}'.format(np.around(c, 2)))
        #
        plt.xlabel('Distance, [pc]')
        plt.ylabel('Ectinction, [mag]')
        plt.legend(loc=2)
        plt.savefig('Figures/Random_Ag_los.png')
        plt.show()


    def call_Ag_los(self, los_ra, los_dec):
        # fit curve in spesific directions:
        # Input: [ra], [dec], Ag_array, R_array (Nra*Ndec, Nbins) 

        Ag_array = np.zeros((len(los_ra), len(los_dec), self.Nbins+1))
        R_array = np.zeros((len(los_ra), len(los_dec), self.Nbins+1))

        Ag_poly, Ag_der = self.Ag_los([los_dec, los_ra], Ag_array, R_array)
        print(np.shape(Ag_poly))
        #print(Ag_array)
        #"""
        for i in range(len(los_ra)):
            plt.figure('longitude: {}'.format(los_ra[i]))
            for j in range(len(los_dec)):
                plt.plot(self.x, Ag_poly[i,j,:], label='dec={}'.format(los_dec[j]))
                
            plt.xlabel('Distance, [pc]')
            plt.ylabel('Extinction, [mag]')
            plt.legend(loc=2)
            plt.savefig('Figures/Ag_los{}_l{}.png'.format(i, int(los_ra[i])))

        #print(np.min(ra), np.max(ra))
        plt.show()
        #"""

    def Ag_los_fitcurve(self, angle_in=None, pixel_in=None):
        """
        Fit a curve to mean Ag along one line of sight. And find the derivative of 
        Ag.
     
        """
        
        if (angle_in == None) and (pixel_in != None):
            # if pixel input
            b, l = hp.pixelfunc.pix2ang(self.Nside, pixel_in)
            print('Angular position:',b*180/np.pi, l*180/np.pi)
            ind1 = np.where(self.gal_pixpos == pixel_in)[0]
            coord = [b*180/np.pi, l*180/np.pi]
        elif (pixel_in == None) and (angle_in != None):
            # if angle input
            dlos = 2.5
            dec_in = angle_in[0]
            ra_in = angle_in[1]
            l = self.longitude
            b = 90 - self.latitude
            print(np.min(l), np.min(b))  
            print(np.max(l), np.max(b))
            ind1 = np.where((b < dec_in + dlos) & (b > dec_in - dlos) &\
                            (l > ra_in - dlos) & (l < ra_in + dlos))[0]
            coord = [dec_in, ra_in]

        else:
            raise('Only one argument as input, "angle_in=[dec, ra]" or "pixel_in=pix.number"')
            sys.exit()

        #
        print('Number of stars in sight line:', len(ind1))
        Ag = self.Ag[ind1]
        dist = self.dist[ind1]

        Bin_ind = np.searchsorted(self.bin, dist)
        ind_sort = np.argsort(Bin_ind)
        
        Ag = Ag[ind_sort]
        dist = dist[ind_sort]
        Bin_ind = Bin_ind[ind_sort]

        Ag_list = np.zeros(self.Nbins+1)
        R_list = np.zeros(self.Nbins+1)
        stdlist = np.zeros(self.Nbins+1)

        counter = 0
        for i in range(self.Nbins-1):
            ind2 = np.where(Bin_ind == i+1)[0]
            #print(len(ind2))
            if len(ind2) > 0:
                Ag_list[i+1] = np.mean(Ag[ind2])
                R_list[i+1] = np.mean(dist[ind2])
                #stdlist[i+1] = np.std(Ag[ind2])
            else:
                if i == 0:
                    Ag_list[i+1] = 0
                else:
                    ran_Ag = np.random.normal(Ag_list[i], 0.46) # sigma: from Andrae etal 2018 (6.5)
                    Ag_list[i+1] = Ag_list[i] + abs(ran_Ag)
                R_list[i+1] = (self.bin[i+1] + self.bin[i])/2
            #
            #print(Ag_list[i], Ag_list[i+1])
            if (Ag_list[i+1] < Ag_list[i]) and (i >= 2):
                counter += 1
                if counter >= 2:
                    #f = interpolate.InterpolatedUnivariateSpline([R_list[i-1], R_list[i]],\
                    #                                             [Ag_list[i-1],Ag_list[i]], k=1)
                    #f = interpolate.interp1d([R_list[i-1], R_list[i]],[Ag_list[i-1],Ag_list[i]],\
                    #                         fill_value='extrapolate')
                    #print(f(R_list[i+1]))
                    #print((Ag_list[i]-Ag_list[i-1])/(R_list[i]-R_list[i-1])) 
                    #print(i, Ag_list[i-1], Ag_list[i])
                    Ag_list[i+1] = (Ag_list[i]) #Ag_list[i]
                            
        #
        ind3 = np.where(Bin_ind >= self.Nbins)[0]
        Ag_list[-1] = np.mean(Ag[ind3])
        R_list[-1] = np.mean(dist[ind3])

        #print(stdlist)
        #print(Ag_list)
        return(Ag_list, R_list, coord)
        ###


    def Draw_pixel(self, Ndraws, Ag_array, R_array):
        
        pix_draw = np.random.choice(self.gal_pixpos, Ndraws)
        Ag_poly = np.zeros((Ndraws, len(self.x)))
        Ag_der = np.zeros((Ndraws, len(self.x)))
        coords = []
        for i, dp in enumerate(pix_draw):
            print('Drawn pixel:', dp)
            Ag_list, R_list, coord = self.Ag_los_fitcurve(pixel_in=dp)
            Ag_array[i,:] = Ag_list
            R_array[i,:] = R_list

            fit = np.polyfit(R_list, Ag_list, 10)
            fx = np.poly1d(fit)

            Ag_poly[i,:] = fx(self.x)
            dAg = self.derivation(fx, self.x)
            Ag_der[i,:] = dAg
            coords.append(coord)
            
            #print(R_list)
            #print(Ag_list)
            plt.figure(i)
            #plt.plot(self.x, Ag_poly[i,:], label='({},{})'.format(round(coord[0],2),\
            #                                                      round(coord[1],2)))
            plt.plot(self.x, dAg, label='({},{})'.format(round(coord[0],2), round(coord[1],2)))
            #plt.plot(R_list, Ag_list, 'xk', label=r'mean $A_G$ in bin')
            plt.xlabel(r'$r$')
            plt.ylabel(r'$A_G$')
            plt.legend(loc=2)
        #
        return(Ag_poly, Ag_der, coords)
        ###

    def Ag_los(self, coordin, Ag_array, R_array):
        """
        For known spesific directions. Calculate the los Ag for known directions.
        plan to plot in position l(b)
        """
        phi = coordin[1]
        theta = coordin[0]
        Nphi = len(phi)
        Ntheta = len(theta)
        print(phi, len(phi))
        print(theta, len(theta))
        Ag_poly = np.zeros((Nphi, Ntheta, len(self.x)))
        Ag_der = np.zeros((Nphi, Ntheta, len(self.x)))
        c = ['k','purple','b','r','g','y','m','c','grey']
        m = ['.','^','x','*','^','+','d','v','o']
        for i in range(Nphi):
            print('=> phi:', phi[i])
            for j in range(Ntheta):
                print('-', phi[i], theta[j])
                Ag_list, R_list, coord = self.Ag_los_fitcurve(angle_in=[theta[j], phi[i]])
                Ag_array[i,j,:] = Ag_list
                R_array[i, j,:] = R_list
                # Merge last and first phi:
                if (phi[i] == 360):
                    Ag0 = Ag_array[0, j, :]
                    Ag1 = Ag_array[-1,j, :]
                    R0 = R_array[0, j,:]
                    R1 = R_array[-1,j,:]
                    Ag_array[-1,j,:] = (Ag0 + Ag1)/2
                    R_array[-1, j,:] = (R0 + R1)/2
                    
                #
                fit = np.polyfit(R_list, Ag_list, 10)
                fx = np.poly1d(fit)
                dAg = self.derivation(fx, self.x)
                
                Ag_poly[i, j,:] = fx(self.x)
                Ag_der[i, j, :] = dAg
                
                plt.figure(i)
                plt.scatter(R_array[i, j,:], Ag_array[i,j,:], c=c[j], marker=m[j],\
                         label=r'$b={}$'.format(theta[j]))
                #plt.plot(self.x, Ag_poly[i,j,:], c=c[j])
                plt.xlabel('$r$')
                plt.ylabel(r'$A_G$')
                plt.legend(loc=2)
                #plt.ylim(-1, 5)
                plt.savefig('Figures/Ag_points2_los{}.png'.format(phi[i]))
            #

        #
                    
        return(Ag_poly, Ag_der)
        ###

    def derivation(self, f, x):
        """
        Compute the derivative of Ag
        """
        h = 1e-5
        return((f(x+h) - f(x-h))/(2*h))
    


    def Interpolate(self, r, fx, N, A=None):
        # If A is 'spline' then use spline interpolation instead of regular interpolation.
        x, dx = np.linspace(np.min(r), np.max(r), N, retstep=True)
        if A == 'spline':
            splrep = interpolate.splrep(r, fx, k=3)
            splev_der = interpolate.splev(x, splrep, der=1)
            #splrep = interpolate.splrep(r, fx, k=3)
            splev = interpolate.splev(x, splrep, der=0)
            return(x, splev, splev_der)
        else:
            f = interpolate.interp1d(r, fx, kind='kubic')
            y = f(x)
            yder = np.diff(y)/dx
            return(x, y, yder)


    def plot_Ag_Maps(self, maps):
        """
        Make map plots of extinction
        """
        print(self.Nmaps)
        for i in range(self.Nmaps):
            print('map', i+1 )
            # rotate map to galactic coords?
            r = hp.rotator.Rotator(coord=['C', 'G'])
            map = r.rotate_map_pixel(maps[i])
            
            hp.mollview(map, title=r'Mean $A_G$ from {}pc to {}pc'.\
                        format(int(self.bin[i]), int(self.bin[i+1])),unit='mag')
            plt.savefig('Figures/Mean_Ag_{}_Nside{}.png'.format(int(self.bin[i+1]),self.Nside))
        plt.show()

    def Plot_los(self, Fx_list, Fx_los, ylab):
        plt.figure()
        Fx_mean = np.mean(Fx_list, axis=1)
        for j in range(len(Fx_list[0,:])):
            plt.plot(self.bin, Fx_list[:,j])
        
        plt.plot(self.bin, Fx_mean, '-k')
        plt.xlabel(r'Distance $r$ [pc]')
        plt.ylabel(r'{} [mag]'.format(ylab))

        plt.figure()
        los_Fx_mean = np.mean(Fx_los, axis=1)
        for j in range(len(los_Fx_mean)):
            plt.plot(self.bin, Fx_los[:,j])

        plt.plot(self.bin, los_Fx_mean, '-k')
        plt.xlabel(r'Distance $r$ [pc]')
        plt.ylabel(r'{} [mag]'.format(ylab))


    def Plot_2D_polar(self, phi, Ag_list):
        r = self.bin
        r_edge = np.linspace(min(r), max(r), len(r))
        phi_edge = phi
        Ag = Ag_list[:,2,:]
        Ag = Ag.T
        print(np.shape(r), np.shape(phi), np.shape(Ag[:,0]))
        #H, _, _ = np.histogram2d(Ag[0,:], Ag[:,0], [r, phi])
        #Phi, R = np.meshgrid(phi, r)
        ax = plt.subplot(111, polar=True)
        ax.pcolormesh(Ag)

    def Plot_Agerr(self):
        bins = np.arange(0, np.max(self.dist), 250)
        meanR = np.zeros(len(bins)-1)
        meanAg_err = np.zeros((len(bins)-1, 4))
        meanAg = np.zeros(len(bins)-1)
        #print(meanAg_err)
        for i in range(len(bins)-1):
            ind = np.where((self.dist > bins[i]) & (self.dist <= bins[i+1]))[0]
            print(i, bins[i+1], len(ind))
            meanR[i] = np.mean(self.dist[ind])
            meanAg[i] = np.mean(self.Ag[ind])
            mean_up = np.mean(self.Ag_upp[ind])
            mean_low = np.mean(self.Ag_low[ind])
            meanAg_err[i, :] = [mean_low, mean_up, (mean_up+mean_low)/2, mean_up-mean_low] 
                            

        #print(meanAg_err)
        #print(meanR)
        ###
        plt.figure()
        plt.plot(meanR, meanAg, '-r', label=r'$\bar{A_G}$')
        plt.plot(meanR, meanAg_err[:,3], '-b', label=r'$\Delta \bar{A_{G}}_{,error}$')
        plt.plot(meanR, meanAg_err[:,0], '--k', label=r'$\bar{A_{G}}_{,low}$')
        plt.plot(meanR, meanAg_err[:,1], ':k', label=r'$\bar{A_{G}}_{,up}$')
        #plt.plot(meanR, meanAg_err[:,2], '-g', label=r'$(\bar{A_{G,up}}+\bar{A_{G,low}})/2$')
        plt.xlabel(r'$r$, [pc]')
        plt.ylabel(r'mean $A_G$, [mag]')
        plt.legend(loc=2)
        plt.grid(True)
        plt.savefig('Figures/mean_Ag_vs_dist.png')

    def plotMaps_total(self, maps, pixel, fx, i):
        """
        Plot the total extinction at a distance interval (x1, x2).
        Input:
        - pixel, sequence. The pixels to map
        """
        for i in range(self.Nmaps):
            map = maps[i]
            print(np.shape(map))
            #map, bin = np.histogram(pixel, self.Npix, weights=fx)
            hp.mollview(map, coord=['C', 'G'], nest=False,\
                        title='Total extinction from {} pc to {} pc'.format(self.bin[i], self.bin[i+1]))
        


#################################
#       Function calls          #
#################################

A = True

Ext = Extinction(128, 30, Rmax=3000)
#Ext.Extinction_map(A)
#Ext.xyzExtinction(Rmax=300, box_size=2)
#Ext.Extinction_mapslice(slices1=5, A=True)
#Ext.ExtinctionMap2()
Ext.Ag_func(10)
#Ext.Ag_los_fitcurve(pixel_in=12345)
#Ext.Plot_Agerr()


plt.show()
