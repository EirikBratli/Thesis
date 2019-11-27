import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import time
###########################
"""
Program to plot thermal dust map from planck for different Nsides.
"""

def Thermal_map(file, Nside_in=0):
    # read fits data or similar:
    td_map = hp.fitsfunc.read_map(file)
    #print(td_map)
    #print(np.shape(td_map))

    #cl, el = self.power_spectrum(td_map)
    #self.plot_planck_map(td_map, cl, el)
    maps = fix_resolution(td_map, Nside_in)
    #smap = smoothing(maps, Nside_in)
    #plot_planck_map(smap, Nside_in)
    #
    return(maps)

def fix_resolution(map, Nside_in=0):
    Nsides = [8,16,32,64,128]
    maps = []
    smaps = []
    t0 = time.time()
    if Nside_in > 0:
        print('Fix resolution to Nside={}'.format(Nside_in))
        m = hp.pixelfunc.ud_grade(map, Nside_in)
        return(m)

    else:
        for Ns in Nsides:
            t1 = time.time()
            print('Fix resolution to Nside={}'.format(Ns))
            m = hp.pixelfunc.ud_grade(map, Ns)
            maps.append(m)

            smap = smoothing(m, Ns)
            smaps.append(smap)

            cl, el = power_spectrum(smap)
            # plot map with new Nside
            plot_planck_map(m, cl, el)
            t2 = time.time()
            #it_time = t2-t1
            #tot_time = t2-t0
            print('Interation time: {}s, total time: {}s'.format((t2-t1), (t2-t0)))
        #

    return(maps)

def power_spectrum(map):
    print('Calculate power spectrum')
    cl = hp.sphtfunc.anafast(map)
    el = np.arange(len(cl))
    return(cl, el)

def smoothing(map, Nside):

    FWHM = 2.5*64/(Nside)*(np.pi/180) # need radians!
    smap = hp.sphtfunc.smoothing(map, fwhm=FWHM, iter=3)
    return(smap)

def plot_powspec(cl, el):
    Nside = hp.pixelfunc.get_nside(map)
    plt.figure()
    plt.loglog(el, el*(el+1)*cl)
    plt.xlabel(r'$l$')
    plt.ylabel(r'$l(l+1)C_l$')
    plt.savefig('Figures1/Planck_maps/powspec_dust857_ns{}.png'.format(Nside))

def plot_planck_map(map):
    Nside = hp.pixelfunc.get_nside(map)

    #map = np.log10(map)
    hp.mollview(map, title='Planck, thermal dust map, Nside={}'.format(Nside))
    plt.savefig('Figures1/Planck_maps/tdust_map857_ns{}.png'.format(Nside))


    ###


###########
#file = 'HFI_SkyMap_857-field-Int_2048_R3.00_full.fits'
#file = 'COM_CompMap_dust-commander_0256_R2.00.fits'

#Thermal_map(file, 0)
#plt.show()
