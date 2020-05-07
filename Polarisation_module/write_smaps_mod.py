"""
Module for writing smoothed maps to file.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import h5py
import sys, time, glob, os
import scipy.optimize as spo

from astropy import units as u_
from astropy.coordinates import SkyCoord

import convert_units as cu
import tools_mod as tools
import smoothing_mod as smooth
import plotting_mod as plotting
import load_data_mod as load


planckfile = 'Data/HFI_SkyMap_353-psb-field-IQU_2048_R3.00_full.fits'
dustfile = 'Data/dust_353_commander_temp_n2048_7.5arc.fits'

Nside_new = int(sys.argv[1])
res = sys.argv[2]


print('load planck 353GHz data')
T, P, Q, U = load.load_planck_map(planckfile, p=True)
d353 = load.load_planck_map(dustfile)
dust353 = tools.Krj2Kcmb(d353) # in uK_cmb
T = T*1e6 # in uK_cmb
P = P*1e6 # in uK_cmb
Q = Q*1e6 # in uK_cmb
U = U*1e6 # in uK_cmb
sys.exit()
Nside = hp.get_nside(T)
print(Nside, Nside_new)
if Nside != Nside_new:
    print('Get new resolution of maps to Nside={}'.format(Nside_new))

    new_maps_pl = tools.ud_grade_maps([T, Q, U, dust353], new_Nside=Nside_new)
    T = new_maps_pl[0]
    Q = new_maps_pl[1]
    U = new_maps_pl[2]
    dust353 = new_maps_pl[3]
    print('maps in new Nside')
    Nside = Nside_new
else:
    pass
#sys.exit()
print('Write smoothed maps to file')
smooth.Write_smooth_map([T, Q, U, dust353], ['IQU', 'dust'], Nside=Nside, res=res)
