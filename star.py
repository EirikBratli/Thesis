#........1.........2.........3.........4.........5.........6.........7.........8

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd
from scipy import interpolate 
from math import atan2
import sys, os, time
import h5py

################################################################################


class Star():
    """
    For an input star, find the pixel (in galactic coordinates) at a distance
    interval. Then apply the extinction of the star.
    Contain functions:
    -
    -
    Input:
    - Nside, integer.
    - r, scalar. distance to the star
    - p, integer/scalar.
    - alpha, scalar. The longitude angle of the star, in deg?
    - delta, scalar. The latitude angle of the star, in deg?
    - i, integer. Star number
    """
    def __init__(self, Nside, Rmax, a):
        self.Nside = Nside
        self.Rmax = Rmax
        self.ra = a[2]
        self.dec = np.pi/2 - a[3]
        self.dist = a[1]
        #self.pixpos = a[2]
        self.Bin_ind = a[0]
        
        self.Ag = a[4]
        #self.ind_in = i
        self.Nbins = int(np.max(self.Bin_ind)) + 1
        self.back = np.empty((0, 4)) # shape ? (r, bin, pix, Ag)
        self.bin, self.dx = np.linspace(0, Rmax, self.Nbins, retstep=True)
        print(self.bin, self.Nbins)
        self.Bin_num = np.linspace(0, self.Nbins, self.Nbins+1)
        
        ###


    def get_Ag_tot(self):
        """
        Get the total extinction in a bin per pixel. Calculate also the mean
        extinction in the pixel.
        Return:
        - Ag_tot, list. List of the total extinction in each pixel per distance
                        bin.
        - Ag_mean, list. List of mean extinction in each pixel per distance bin.
        """
        print(self.Nbins)
        Npix = hp.pixelfunc.nside2npix(self.Nside)
        Ag_tot = []
        Ag_mean = []
        N_star = []
        cols = []
        #Ag_data = {'bin': {'pix': ,'tot': , 'mean': , 'N':}}
        for i in range(self.Nbins):
            ind1 = np.where(self.Bin_ind == i+1)[0]
            print('--> Bin {}:'.format(i+1), len(ind1))
            pix = hp.pixelfunc.ang2pix(self.Nside, self.dec[ind1], self.ra[ind1])
            
            Ag = self.Ag[ind1] # Ag in bin
            ind2 = np.argsort(pix)  # sort for pixels
            pix2 = pix[ind2]
            Ag2 = Ag[ind2]
            #i_miss = np.where(np.diff(pix[ind2]) > 1)[0]
            #pix3 = np.insert(pix2, i_miss+1, pix2[i_miss]+1)
            #Ag3 = np.insert(Ag2, i_miss+1, 0)

            d = {'pix': pix2, 'Ag': Ag2}
            data = pd.DataFrame(data=d)
            #groupsum = data.groupby('pix')['Ag'].sum()
            #groupmean = data.groupby('pix')['Ag'].mean()
            #grlen = data.groupby('pix')['Ag'].count()
            Ag_data = data.groupby('pix')['Ag'].agg([np.sum, np.mean, np.size])
            pixel = Ag_data.index.tolist()
            cols.append(i+1)
            
            #print(Ag_tot, np.shape(Ag_tot))
            print(Ag_data)
            
            # plot map
            map_mean, b = np.histogram(pixel, bins=Npix, weights=Ag_data['mean'])
            map_tot, b1 = np.histogram(pixel, bins=Npix, weights=Ag_data['sum'])
            Ag_tot.append(map_tot)
            Ag_mean.append(map_mean)

            #r = hp.rotator.Rotator(coord=['C','G'])
            #map = r.rotate_map_pixel(map_tot)
            #hp.mollview(map)
            #print(len(Ag_data['sum'].fillna(0)))
            #plt.plot(pix3)
        #
        #plt.show()
        return(Ag_tot, Ag_mean)

    def slicemap(self, Ndec, Nra):  # not working!!
        rot = hp.rotator.Rotator(coord=['C', 'G'])
        dec2, ra2 = rot(self.dec, self.ra)
        ra = (ra2+ np.pi)*180/np.pi
        dec = dec2*180/np.pi

        ang_l = np.linspace(0, 360, Nra)
        ang_b = np.linspace(0, 180, Ndec)

        ra_bin = np.searchsorted(ang_l, ra)
        dec_bin = np.searchsorted(ang_b, dec)
        rm, tm = np.meshgrid(self.bin, ang_b)
        Ag_grid0 = np.zeros((len(ang_l)-1, len(ang_b)-1, self.Nbins))
        dAg_grid = np.zeros((len(ang_l)-1, len(ang_b)-1, self.Nbins-1))
        print(np.shape(Ag_grid0[:,:,:]), np.shape(rm))
        ind1 = np.argsort(ra_bin)
        df = pd.DataFrame({'ra_bin': ra_bin[ind1],
                           'ra': ra[ind1],
                           'dec': dec[ind1],
                           'r_bin': self.Bin_ind[ind1],
                           'dist': self.dist[ind1],
                           'Ag': self.Ag[ind1]})
        print(df)
        dist = self.dist
        #sys.exit()
        data = pd.DataFrame({'ra_bin': np.zeros(len(ra_bin)),
                             'dec_bin': np.zeros(len(ra_bin)),
                             'Ag': np.zeros(len(ra_bin))})

        for i in range(self.Nbins-1):
            ind3 = np.where((dist > self.bin[i]) & (dist <= self.bin[i+1]))[0]
            print('-->', i, self.bin[i+1], len(ind3))
                        
            data['ra_bin'][ind3] = ra_bin[ind3]
            data['dec_bin'][ind3] = dec_bin[ind3]
            data['Ag'][ind3] = self.Ag[ind3]
            #print(data)

            dt2 = data.groupby(['ra_bin','dec_bin'])['Ag'].agg([np.mean,np.size])#.replace({'dec_bin': {'tmp': 0}})
            index = np.array(dt2.index.values).astype(int)
            am = np.asarray(dt2['mean'][1:])
            
            am = np.reshape(am, (Nra-1, Ndec-1))
            Ag_grid0[:,:,i+1] = am
            print(Ag_grid0[:,:,i+1])
                        
        # Not working!!!
        ####

    def get_slice_maps(self, Ndec, Nra):
        Npix = hp.pixelfunc.nside2npix(self.Nside)
        #print(self.ra[:100])
        #print(self.dec[:100])
        Ndec = Ndec + 1
        Nra = Nra + 1
        df = pd.DataFrame({'dist': self.dist,
                           'ra': self.ra,
                           'dec': self.dec,
                           'Ag': self.Ag})

        Bin_ind = self.Bin_ind#[ind1]
        dist = self.dist#[ind1]
        Ag = self.Ag#[ind1]
        # rotate to galactic coordinates
        rot = hp.rotator.Rotator(coord=['C','G'])
        dec2, ra2 = rot(self.dec, self.ra) # (0,PI), (-PI,PI)
        ra = (ra2 + np.pi)*180/np.pi  # to degrees
        dec = dec2*180/np.pi          # to degrees
        ang = np.linspace(0, 360, Nra)
        r = np.linspace(self.Rmax/self.Nbins, self.Rmax*(1+1/self.Nbins), self.Nbins)
        print(ra)
        ra_bin = np.searchsorted(ang, ra)
        print(df)
        print(ra_bin, np.max(ra_bin))
        #sys.exit()
        Ndiff = self.Nbins*3
        dec_ang = np.linspace(0, 180, Ndec)
        rm, tm = np.meshgrid(self.bin, dec_ang)
        drm, dtm = np.meshgrid(np.linspace(0, self.Rmax+10, Ndiff+1), dec_ang)
        Ag_grid = np.zeros((len(ang), len(dec_ang), self.Nbins))
        dAg_grid = np.zeros((len(ang), len(dec_ang), Ndiff))
        print(np.shape(Ag_grid[0,:,:]), np.shape(dAg_grid))
        
        for i in range(len(ang)-1):
            
            print('-->', i, i+1, ang[i], ang[i+1])#, len(ind2))
            for j in range(len(dec_ang)-1):
                #print(j, j+1, dec_ang[j], dec_ang[j+1])
                ind2 = np.where((ra >= ang[i]) & (ra < ang[i+1]) &\
                                (dec >= dec_ang[j]) & (dec<dec_ang[j+1]))[0]
                    
                data = pd.DataFrame({'bin': Bin_ind[ind2],
                                     'r': self.dist[ind2],
                                     'Ag': self.Ag[ind2]})

                #print(len(ind2))#print(data)
            
                Ag_data = data.groupby('bin')['Ag'].agg([np.mean,np.std,np.size])
                R_data = data.groupby('bin')['r'].agg([np.mean, np.std])
                temp_dist = np.zeros((self.Nbins))
                #
                #print(Ag_data)
                #print(R_data)
                #print(np.array(Ag_data.index.values).astype(int))
                #print(len(np.array(Ag_data.index.values).astype(int)))
                dist_index = np.array(Ag_data.index.values).astype(int)
                Ag_grid[i,j, dist_index] = Ag_data['mean']
                temp_dist[dist_index] = R_data['mean']
                
                ii = np.where(temp_dist[1:] == 0)[0]
                #print(temp_dist[ii+1])
                if len(ii) > 0:
                    #r = temp_dist[ii] + self.dx
                    for k in ii:
                        r = temp_dist[k] + self.dx
                        temp_dist[k+1] = r
                    print(ii)
                else:
                    pass
                #print(temp_dist[ii+1])
                #print(Ag_grid[i,j,:])
                diff_Ag = self.diff_Ag(Ag_grid[i,j,:], temp_dist, Ndiff)
                dAg_grid[i, j, :] = diff_Ag
                #print(Ag_data['mean'])
            #
            #print(Ag_grid[i,:,:])
            #print(np.mean(dec[np.where((ra >= ang[i]) & (ra < ang[i+1]))]))
            # plot:
            #plt.figure('ra: {}'.format(ang[i+1]))
            #plt.plot(Ag_data['mean'])
            #ax = plt.subplot(111, polar=True)
            #plt.pcolormesh(rm, tm, Ag_grid[i,:,:])
            #cb = plt.colorbar()
            plt.figure('diff Ag: ra {}'.format(ang[i+1]))
            #ax = plt.subplot(111, polar=True)
            plt.pcolormesh(drm[:,:-1], dtm[:,:-1], dAg_grid[i,:,:])
            cb = plt.colorbar()
        #####
        plt.show()
        #####

    def diff_Ag(self, Ag_mean, r, N):
        """
        Calculate the differential extinction along a sight line. Using the 
        info in Chen etal 2018. return also indices. Interpolate between bins 
        as well
        """
        
        #N = 10 * self.Nbins # splining points
        x = np.linspace(np.min(r), np.max(r), N+1)
        #print(r)
        #Ag_f = interpolate.interp1d(r, Ag_mean, kind='cubic')
        #Ag_new = Ag_f(x)
        #dAg_intp = np.diff(Ag_new)
        dAg_splrep = interpolate.splrep(r, Ag_mean, k = 3)
        dAg = interpolate.splev(x, dAg_splrep, der=1)
        #print(np.diff(Ag_mean))
        #index = np.nonzero(diff_Ag)
        #dx = (self.bin[-1] - self.bin[0])/(self.Nbins - 1)
        #print(dAg_intp)
        #return(dAg_intp)
        return(dAg)

        

    def add_star_info(self, array):
        """
        Send in sorted distance, bin num, pixel, Ag after bin number.
        Get the bin number, pixel, extinction of the star into an array
        """
        #bin = self.get_Bin() 
        #pix = self.get_pixel()
        array = np.array([[self.dist, bin, pix, self.Ag]])
        #self.back = np.concatenate((in_prev, input), axis=0 # to much time)
        #data = {'bin': bin,[self.dist, self.Ag]}
        return(array)

    def get_N(self, N_in):
        #N = 'N stars in a 3d position (r, theta, alpha)'
        data = {'N_star': {'N': N_in+1, 'p': self.pix, 'r': self.dist}}
        return(N)

#### class end ###

""" 
Ag_tot = np.zeros((Npix, self.Nbins))
for i in range(Npix):
ind1 = np.where(self.pixpos == i)[0]
print(i, len(ind1))
r = self.dist[ind1]
Ag = self.Ag[ind1]
for j in range(1, self.Nbins+1):
ind2 = np.where(self.Bin_ind[ind1] == j)[0]
Ag1 = np.sum(Ag[ind2]) 
print(len(ind2), j, Ag1)
Ag_tot[i, j] = Ag1
"""

def coord_trans(ra, dec):
    """
    Transform RA and DEC to galactic coordinates, in deg!
    """
    degrees = np.pi/180.
    el0 = 32.93191857 * degrees # a
    r0 = 282.859481208 * degrees # b
    d0 = 62.8717488056 * degrees # c
    
    l1 = np.cos(dec)*np.cos(ra - r0)
    l2 = np.sin(dec)*np.sin(d0) + np.cos(dec)*np.cos(ra - r0)*np.cos(d0)
    l3 = np.sin(dec)*np.cos(d0) - np.cos(dec)*np.sin(ra - r0)*np.sin(d0)
    #print(l1,l2, l3)
    B = np.arcsin(l3)/degrees
    
    L = (np.arctan2(l1, l2) + el0)/degrees
    #print(L, B)
    return(L,B)


#coord_trans(0,0)
