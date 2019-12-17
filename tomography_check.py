"""
Program to compare Planck 353Ghz polarization map to tomagraphy data of Raphael
etal 2019.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import h5py
import sys, time


#######################

def load_tomographydata(file, colname):
    colnum = getCols(file, colname)
    print(colname)
    print(colnum)
    data = np.genfromtxt(file, delimiter=',', skip_header=1, usecols=colnum)
    print(np.shape(data))
    return(data)

def getCols(file, colname):
    a = np.genfromtxt(file, delimiter=',', names=True)
    b = np.asarray(a.dtype.names)
    colnum = []
    for j in range(len(colname)):
        for i in range(len(b)):
            if colname[j] == b[i]:
                colnum.append(i)
    return(colnum)

def tomo_map(data, Nside=2048):
    # make healpix map for the given Nside for the data
    theta = np.pi/2. - data[:,1] * np.pi/180.
    phi = data[:,0] * np.pi/180.

    # get pixel numbers
    pix = hp.pixelfunc.ang2pix(Nside, theta, phi)
    print(pix)
    print(hp.pixelfunc.nside2pixarea(Nside, degrees=True))
    # Create maps
    Npix = hp.nside2npix(Nside)
    p_map = np.zeros(Npix)
    q_map = np.zeros(Npix)
    u_map = np.zeros(Npix)

    print(Npix, np.shape(p_map))
    print(len(np.unique(pix)))
    uniqpix = np.unique(pix)
    #print(uniqpix)
    for i in uniqpix:
        ind = i == pix

        p = np.sum(data[ind, 2])
        q = np.sum(data[ind, 4])
        u = np.sum(data[ind, 6])

        p_map[i] = p
        q_map[i] = q
        u_map[i] = u
        #print(i, p, data[ind, 2], p_map[i])
    #p_map[uniqpix] = hp.UNSEEN
    #p_map[pix] = data[:,2]
    #q_map[pix] = data[:,4]
    #u_map[pix] = data[:,6]
    print(p_map[uniqpix])
    print(q_map[uniqpix])
    print(u_map[uniqpix])
    print(np.sum(p_map==0))
    return(p_map, q_map, u_map, pix)

def plot_q():
    pass

def main(tomofile, colnames):

    data = load_tomographydata(tomofile, colnames)
    print(data[0,:])
    p_map, q_map, u_map, pix = tomo_map(data)

    ind = (np.where(p_map > 0)[0])
    ind1 = (np.where(q_map > 0)[0])
    ind2 = (np.where(u_map != 0)[0])
    print(len(ind), len(ind1), len(ind2))
    #print(p_map[ind])
    #plt.hist(p_map[pix], bins=574)
    hp.mollview(np.log(p_map), title='p')
    plt.savefig('Figures/tomography/test_p.png')
    hp.mollview(np.log(q_map), title='q')
    plt.savefig('Figures/tomography/test_q.png')
    hp.mollview(np.log(u_map), title='u')
    plt.savefig('Figures/tomography/test_u.png')

    plt.show()

file = 'Data/total_tomography.csv'
colnames = ['ra', 'dec', 'p', 'p_er', 'q', 'q_er', 'u', 'u_er', 'dist',\
            'dist_low', 'dist_up', 'Rmag1']
main(file, colnames)
