"""
Main program in tomography analysis of polarisation data in 3D
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import h5py, argparse
import sys, time, glob, os
import scipy.optimize as spo

from astropy import units as u_
from astropy.coordinates import SkyCoord

import convert_units as cu
import tools_mod as tools
import smoothing_mod as smooth
import plotting_mod as plotting
import load_data_mod as load


### Main function ####

def main(planckfile, dustfile, tomofile, colnames, names, pol, res,\
        part='all'): 
    """
    The main function of the program. Do all the calling to the functions used 
    to calculate the comparison between the Tomography data and Planck         
    polarisation data. Want to smooth to uK_cmb.                               
                                                                               
    Parameters:                                                                
    -----------                                                                
    - planckfile, string.   Name of the planck file to compare with.           
    - dustfile, string.     Name of the dust intensity file.                   
    - tomofile, string.     The name of the tomography file.                   
    - colnames, list.       List of the column names of the tomography file.   
    - names, list.          List with the column names in the smoothed planck  
                            maps, with polarisation first then dust intensity. 
    - pol, bool.            Which Stokes parameter to evaluate.                
                                                                               
    Return:                                                                    
    -------
    """
    
    # read smoothed planck maps.                                            
    print('load planck 353GHz data')
    # read_smooth_maps(filename, name, shape)                               
    IQU_smaps = smooth.read_smooth_maps(planckfile, names[0], 3)
    dust_smap = smooth.read_smooth_maps(dustfile, names[1], 1)[0]
    T_smap = IQU_smaps[0]
    Q_smap = IQU_smaps[1]
    U_smap = IQU_smaps[2]

    Nside = hp.get_nside(T_smap)
    print('Using Nside={}'.format(Nside))

    #sys.exit()                                                             
    # load tomography data:                                                 
    data = load.load_tomographydata(tomofile, colnames)
    print('Data loaded, using Nside={}'.format(Nside))

    p, q, u, sigma, r_map, pix = load.pix2star_tomo(data,\
                                                    Nside, part)
    u = -u # to Healpix convention                                  
    mask = pix
    print(len(mask))
    # Modify length of submillimeter polarisation arrays:
    Ts = T_smap[mask]
    Qs = Q_smap[mask]
    Us = U_smap[mask]
    dust = dust_smap[mask]
    # modify the smoothing
    u_smap = smooth.smooth_tomo_map(u, mask, Nside, res)
    q_smap = smooth.smooth_tomo_map(q, mask, Nside, res)
    p_smap = smooth.smooth_tomo_map(p, mask, Nside, res)
    u = u_smap[mask]
    q = q_smap[mask]
    p = p_smap[mask]
    print('Tomography maps smoothed')
    #print(np.mean(q_smap[mask]), np.mean(dust_smap[mask]))
    dPsi = np.full(len(u), hp.UNSEEN)
    #sys.exit()                                                             

    l, b = tools.convert2galactic(data[:,0], data[:,1])
    lon = np.mean(l)
    lat = np.mean(b)
    print(lon, lat)

    # calculate Delta psi:
    print(len(Qs), len(q), len(Us), len(u))
    print('Calculate Delta psi:')
    psi = tools.delta_psi(Qs, q, Us, u)

    # Calculate ratio Ps/pv:
    unit1 = 287.45 # MJy/sr/Kcmb                                               
    unit2 = unit1*1e-6
    print('Calculate Ps/pv:')
    Ps = np.sqrt(Qs**2 + Us**2)
    print('Ps/pv=',np.mean(Ps/p)*unit2, np.std(Ps/p)*unit2)
    print(np.mean(Ts/p)*unit2)
    print('--------')
    
    planck = [Qs, Us, dust]
    tomo = [q, u, sigma]    
    return(planck, tomo, r_map, mask)

##################

dtnames = 'IQU+dust'
colnames = ['ra', 'dec', 'p', 'p_er', 'evpa', 'evpa_er', 'q', 'q_er',\
            'u', 'u_er', 'dist', 'dist_low', 'dist_up', 'Rmag1']
dtnames = dtnames.split('+')
print(dtnames)

tomofile = 'Data/total_tomography.csv'
IQU_file = 'Data/IQU_Nside2048_15arcmin.h5'
dust_file = 'Data/dust_Nside2048_15arcmin.h5'

part = 'all'
part = '1cloud'
part = '2cloud'
save = part[:2]
print(save)

#### Function call ####
planck, tomo, r_map, mask = main(IQU_file, dust_file, tomofile, colnames,\
                                 dtnames, pol='qu', res=15, part=part)

Q, U, dust = planck[:]
q, u, sigma = tomo[:]
#print(sigma[0])
#### Plotting ####
plotting.plot_corr_stars(Q, U, q, u, sigma[1], sigma[2], r_map, mask,\
                         x_lab=r'$q_v, u_v$', y_lab=r'$Q_s^{{353}},U_s^{{353}}$',\
                         save=save, part=part)


plt.show()
