#........1.........2.........3.........4.........5.........6.........7.........8

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd
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
        self.Nbins = int(np.max(self.Bin_ind))
        self.back = np.empty((0, 4)) # shape ? (r, bin, pix, Ag)
        self.bin, self.dx = np.linspace(0, Rmax, self.Nbins, retstep=True)
        print(self.bin)
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

    def get_slice_maps(self, Ndec, Nra):
        """
        Compute 
        """
        Npix = hp.pixelfunc.nside2npix(self.Nside)
        #print(self.ra[:100])
        #print(self.dec[:100])
        df = pd.DataFrame({'dist': self.dist,
                           'ra': self.ra,
                           'dec': self.dec,
                           'Ag': self.Ag})

        print(df)

        Bin_ind = self.Bin_ind#[ind1]
        dist = self.dist#[ind1]
        Ag = self.Ag#[ind1]

        # rotate to galactic coordinates
        rot = hp.rotator.Rotator(coord=['C','G'])
        dec2, ra2 = rot(self.dec, self.ra) # (0,PI), (-PI,PI)
        
        #print(ra2, dec2)
        ra = (ra2 + np.pi)*180/np.pi  # to degrees
        dec = dec2*180/np.pi          # to degrees
        
        ang = np.linspace(0, 360, Nra)
        r = np.linspace(self.Rmax/self.Nbins, self.Rmax*(1+1/self.Nbins), self.Nbins)
        #print(np.min(self.ra), np.max(self.ra), np.mean(self.ra))
        print(np.min(dec), np.max(dec), np.mean(dec))
        
        dec_ang = np.linspace(0, 180, Ndec)
        rm, tm = np.meshgrid(self.bin, dec_ang)
        print(dec_ang[::2])
        print(Bin_ind)
        print(ang)
        #sys.exit()
        Ag_grid = np.zeros((len(ang), len(dec_ang), self.Nbins))
        dAg_grid = np.zeros((len(ang), len(dec_ang), self.Nbins-1))
        print(np.shape(Ag_grid[0,:,:]), np.shape(rm))
        #sys.exit()
        for i in range(len(ang)-1):
            
            print('-->', i, i+1, ang[i], ang[i+1])#, len(ind2))
            for j in range(len(dec_ang)-1):
                #print(j, j+1, dec_ang[j], dec_ang[j+1])
                ind2 = np.where((ra >= ang[i]) & (ra < ang[i+1]) &\
                                (dec >= dec_ang[j]) & (dec<dec_ang[j+1]))[0]
            
                print(np.where(np.diff(Bin_ind[ind2]) > 1)[0], np.diff(Bin_ind[ind2]))
                data = pd.DataFrame({'bin': Bin_ind[ind2],
                                     'r': dist[ind2],
                                     'Ag': Ag[ind2]})

                step = np.diff(Bin_ind[ind2])
                print(np.max(step))
                jump = np.max(step)
                if np.max(step) > 1:
                    # append N steps to data at the appropriate location
                    ii = np.where(step > 1)[0]
                    old = data
                    BI = Bin_ind[ind2]
                    print('----')
                    print(ii, max(step), BI)
                    print(old)
                    bi = [ii + k for k in range(1, int(jump))]
                    rr = [BI[ii + k] for k in range(1, int(jump))]
                    ag = np.zeros(int(jump))
                    print(bi, rr, ag)
                    new = pd.DataFrame({'bin':[i+ii for i in range(1,int(jump))],
                                        'r': [BI[i+ii] for i in range(1,int(jump))],
                                        'Ag': [0 for i in range(1,int(jump))]})
                    print(new)
                    old_ind = old.index[old[0]==ii].tolist()[0]
                    old1 = old[old.index < old_ind]
                    old2 = old[old.index > old_ind]
                    #res = old1.append(new, ignore_index=True).\
                    #     append(old2, ignore_index=True)
                    #print(

                
                print(data)
                Ag_data = data.groupby('bin')['Ag'].agg([np.sum,np.mean,np.size])
                print(Ag_data)
                diff_Ag = self.diff_Ag(Ag_data['mean'])
                Ag_grid[i,j,:] = Ag_data['mean']
                dAg_grid[i,j,:] = diff_Ag
                #print(Ag_data['mean'])
            #
            #print(Ag_grid[i,:,:])
            print(np.mean(dec[np.where((ra >= ang[i]) & (ra < ang[i+1]))]))
            # plot:
            plt.figure('ra: {}'.format(ang[i+1]))
            #plt.plot(Ag_data['mean'])
            #ax = plt.subplot(111, polar=True)
            plt.pcolormesh(rm, tm, Ag_grid[i,:,:])
            cb = plt.colorbar()
            #plt.figure('diff Ag: ra {}'.format(ang[i+1]))
            #ax = plt.subplot(111, polar=True)
            #ax.pcolormesh(rm[:,:-1], tm[:,:-1], dAg_grid[i,:,:])
            #cb = plt.colorbar()
        #####
        """
        rm, pm = np.meshgrid(self.bin, ang)
        for i in range(len(dec_ang)):
            plt.figure('dec: {}'.format(dec_ang[i]))
            plt.pcolormesh(rm, pm, Ag_grid[:,i,:])
            cb = plt.colorbar()            
        """
        plt.show()
        #####

    def diff_Ag(self, Ag_mean):
        #print(Ag_mean)
        diff_Ag = np.diff(Ag_mean)#np.zeros(len(Ag_mean))
        #for i in range(1, len(Ag_mean)):
        #    diff_Ag[i] = Ag_mean['mean'][i] - Ag_mean['mean'][i-1]
        #print(diff_Ag)
        return(diff_Ag)

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
    a = 33. * np.pi/180.
    b = 62.6 * np.pi/180
    c = 282.25 * np.pi/180
    
    l1 = np.cos(dec)*np.cos(ra - c)
    l2 = np.sin(dec)*np.sin(b) + np.cos(dec)*np.cos(ra - c)*np.cos(b)
    l3 = np.sin(dec)*np.cos(b) - np.cos(dec)*np.sin(ra - c)*np.sin(b)
    print(l1,l2, l3)
    B = np.arcsin(l3)
    cosB = np.sqrt(1 - l3**2)
    Lmina = np.arccos(l1/cosB)
    Lmina2 = np.arcsin(l2/cosB)
    print(B, Lmina, Lmina2)

#coord_trans(0., 0.)    
    
