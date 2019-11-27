#################################################
#                                               #
#   Program to compare CO and dust maps from    #
#   Planckwith extinction maps from Gaia.       #
#                                               #
#################################################

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import os, sys
import scipy as sp
import h5py

#################################################


def Read_H5(file, name):
    f = h5py.File(file, 'r')
    data = np.asarray(f[name])
    f.close()
    return(data)

def get_Planck_map(filename):
    print('Read file: {}'.format(filename))
    map = hp.fitsfunc.read_map(filename)
    Nside = hp.pixelfunc.get_nside(map)
    return(map, Nside)

def fix_resolution(map, Nside_in=0):
    Nsides = [16,32, 64, 128, 256]
    maps = []
    smaps = []

    if Nside_in > 0:
        print('Fix resolution to Nside={}'.format(Nside_in))
        m = hp.pixelfunc.ud_grade(map, Nside_in)
        sm = smoothing(m, Nside_in)
        return(m, sm)

    else:
        for Ns in Nsides:
            print('Fix resolution to Nside={}'.format(Ns))
            m = hp.pixelfunc.ud_grade(map, Ns)
            smap = smoothing(map, Ns)
            maps.append(m)
            smaps.append(smap)

        return(maps, smap)
    #

def smoothing(map, Nside):
    FWHM = 2.5*(64/Nside) * (np.pi/180)
    smap = hp.sphtfunc.smoothing(map, fwhm=FWHM, iter=3)
    return(smap)

def powerspectrum(map):
    cl = hp.sphtfunc.anafast(map)
    ell = np.arange(len(cl))
    return(cl, ell)

def call_Compare_maps(Ag_file, dust_file=None, CO_file=None):
    """
    Function to compare the maps.
    """

    print('Read extinction map:')
    Ag_map = Read_H5(Ag_file, 'Ag')
    Nside_Ag = hp.pixelfunc.get_nside(Ag_map)
    Ag_smap = smoothing(Ag_map, Nside_Ag)
    print(Nside_Ag)
    # get power spectrum for smoothed Ag
    cl_Ag, ell_Ag = powerspectrum(Ag_smap/np.max(Ag_smap))
    plot_cl(cl_Ag, ell_Ag, 'b', 'Ag')
    if (dust_file != None):
        # compare dust to extinction
        dust_map, Nside_dust = get_Planck_map(dust_file)
        #print(len(dust_map), dust_map)
        #Nside_dust = hp.pixelfunc.get_nside(dust_map)
        print('Dust:', Nside_dust)
        if Nside_dust != Nside_Ag:
            dust_map, dust_smap = fix_resolution(dust_map, Nside_Ag)
        else:
            dust_smap = smoothin(dust_map, Nside_Ag)

        # power spectrum:
        cl_dust, ell_dust = powerspectrum(dust_smap/np.max(dust_smap))
        fx, coeffs1, ind1 = line_fit(Ag_smap, dust_smap, ymax=7.5)
        gx, coeffs2, ind2 = line_fit(Ag_smap, dust_smap, ymax=7.5, upper=True)
        print('plotting dust')
        # plotting
        #map_plots(Ag_map, 'Ag', Nside_Ag)
        #map_plots(dust_map, 'Dust', Nside_Ag)
        #vs_plot(Ag_map, dust_map, 'Dust')

        # smoothed
        map_plots(dust_smap, 'dust', Nside_Ag, '$K_{RJ}$', smooth=True)
        map_plots(np.log(dust_smap), 'Dust', Nside_Ag, '$K_{RJ}$', smooth=True)
        vs_plot(Ag_smap, dust_smap, 'Dust', '$K_{RJ}$', ymax=7.5, smooth=True)

        map_comparison(Ag_smap, dust_smap, Nside_Ag, 'dust', '$K_{RJ}$', coeffs2[0],\
                        coeffs2[0], ind1, ind2)
        plot_line_scatter(Ag_smap, dust_map, fx, gx, 'dust', '$K_{RJ}$')
        plot_cl(cl_dust, ell_dust, 'r', 'dust')

    if (CO_file != None):
        # compare CO to extinction
        CO_map, Nside_CO = get_Planck_map(CO_file)
        #Nside_CO = hp.pixelfunc.get_nside(CO_map)
        print('CO:', Nside_CO)
        if Nside_CO != Nside_Ag:
            CO_map, CO_smap = fix_resolution(CO_map, Nside_Ag)
        else:
            CO_smap = smoothing(CO_map, Nside_Ag)

        # power spectrum:
        cl_co, ell = powerspectrum(CO_smap/np.max(CO_smap))
        fx, coeffs1, ind1 = line_fit(Ag_smap, CO_smap, ymax=5)
        gx, coeffs2, ind2 = line_fit(Ag_smap, CO_smap, ymax=5, upper=True)
        print('Plotting CO')
        # plotting
        #map_plots(Ag_map, 'Ag', Nside_Ag, '$K_{RJ} km s^{-1}$')
        #map_plots(CO_map, 'CO', Nside_Ag, '$K_{RJ} km s^{-1}$')
        #vs_plot(Ag_map, CO_map, 'CO', '$K_{RJ} km s^{-1}$')

        # smoothed
        map_plots(CO_smap, 'CO', Nside_Ag, '$K_{RJ} km s^{-1}$', smooth=True)
        map_plots(np.log(CO_smap), 'CO', Nside_Ag, '$K_{RJ} km s^{-1}$', smooth=True)
        vs_plot(Ag_smap, CO_smap, 'CO', '$K_{RJ} km s^{-1}$', ymax=5, smooth=True)
        map_comparison(Ag_smap, CO_smap, Nside_Ag, 'CO', '$K_{RJ} km s^{-1}$',\
                        coeffs1[0], coeffs2[0], ind1, ind2)
        plot_line_scatter(Ag_smap, CO_map, fx, gx, 'CO', '$K_{RJ} km s^{-1}$')
        plot_cl(cl_co, ell, 'g', 'CO')

    else:
        print('Need to compare with dust and/or CO')
        sys.exit()

def line_fit(x, y, xmax=1, ymax=10, upper=False):
    if upper == True:
        ind = np.where(y > ymax)[0]
        xmod = x[ind]
        ymod = y[ind]
        fit = np.polyfit(xmod, ymod, 1)
    else:
        ind = np.where(y <= ymax)[0]
        xmod = x[ind]
        ymod = y[ind]
        fit = np.polyfit(xmod, ymod, 1)
    return(np.poly1d(fit), fit, ind)

def plot_line_scatter(Ag_map, comp_map, fx, gx, ylab, yunit):
    x = np.arange(int(np.max(Ag_map)+2))
    print(np.min(Ag_map), np.max(Ag_map))
    plt.figure()
    plt.scatter(Ag_map, comp_map, s=0.3, c='b')
    plt.plot(x, fx(x), '-r', label='lower fit')
    plt.plot(x, gx(x), '-g', label='upper fit')
    plt.xlabel(r'$A_G$, [mag]')
    plt.ylabel(r'${}$, [{}]'.format(ylab, yunit))
    plt.legend(loc=2)
    plt.savefig('Figures/Comparison/Ag_vs_{}_fit.png'.format(ylab))

def vs_plot(Ag_map, comp_map, ylab, yunit, ymax=10, smooth=False):
    # Plotting
    fit = np.polyfit(Ag_map, comp_map, 1)
    fx = np.poly1d(fit)
    x = np.arange(int(np.max(Ag_map)+1))
    gx, coeffs, ind = line_fit(Ag_map, comp_map, ymax=ymax)
    print('Plynomial coefficients:', coeffs)
    if smooth==False:
        plt.figure('Ag vs {}'.format(ylab))
        plt.scatter(Ag_map, comp_map, s=0.3, c='b')
        plt.plot(x, fx(x), '-k')
        plt.plot(x, gx(x), '-g')
        plt.xlabel(r'$A_G$, [mag]')
        plt.ylabel(r'{}, [{}]'.format(ylab, yunit))
        #plt.yscale('log')
        plt.savefig('Figures/Comparison/Ag_vs_{}.png'.format(ylab))
    else:
        plt.figure('smoothed Ag vs {}'.format(ylab))
        plt.scatter(Ag_map, comp_map, s=0.3, c='b')
        plt.plot(x, fx(x), '-k')
        plt.plot(x, gx(x), '-g')
        plt.xlabel(r'$A_G$, [mag]')
        plt.ylabel(r'{}, [{}]'.format(ylab, yunit))
        #plt.yscale('log')
        plt.xlim(0.3, 1)
        plt.ylim(-0.5, 20)
        plt.savefig('Figures/Comparison/smoothed_Ag_vs_{}.png'.format(ylab))

def map_comparison(Ag_map, comp_map, Nside, comp, unit, fac1, fac2=None, ind1=None,\
                    ind2=None):
    #
    map = comp_map
    map[ind1] = (comp_map[ind1] - fac1*Ag_map[ind1])
    map[ind2] = (comp_map[ind2] - fac2*Ag_map[ind2])

    hp.mollview(map, title=(r'{} - $b\times A_G$, Nside={}').format(comp, Nside),\
                unit=unit)
    plt.savefig('Figures/Comparison/map_comparison_{}_Ns{}.png'.format(comp, Nside))

    hp.mollview(np.log(map), title=(r'{} - $b\times A_G$, Nside={}').\
                format(comp, Nside), unit=unit)
    plt.savefig('Figures/Comparison/log_map_comparison_{}_Ns{}.png'.format(comp, Nside))

def map_plots(map, lab, Ns, unit, smooth=False):
    #
    if smooth==False:
        hp.mollview(map, title='{} map'.format(lab), unit=unit)
        plt.savefig('Figures/{}_map_Nside{}.png'.format(lab, Ns))

    else:
        hp.mollview(map, title='smoothed {} map'.format(lab), unit=unit)
        plt.savefig('Figures/smooth_{}_map_Nside{}.png'.format(lab, Ns))

def plot_cl(cl, ell, color, lab):
    plt.figure('power spectrum')
    plt.plot(ell, ell*(ell + 1)*cl, c=color, label='{}'.format(lab))
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\ell(\ell+1)C_{{\ell}}$')
    plt.legend(loc='best')
    plt.savefig('Figures/Powerspectrum.png')

    plt.figure('log power spectrum')
    plt.semilogy(ell, ell*(ell + 1)*cl, c=color, label='{}'.format(lab))
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\ell(\ell+1)C_{{\ell}}$')
    plt.legend(loc='best')
    plt.savefig('Figures/Powerspectrum_log.png')




###########################
#      Function calls     #
###########################

Ag_file_dt128 = 'Data/map_3000r_Nside128.h5'
Ag_file_sim16 = 'Data/Ag_map_r3000_Nside16.h5'
dust_file = 'Data/HFI_SkyMap_857-field-Int_2048_R3.00_full.fits'
CO_file = 'Data/co_c0001_k000001.fits'

#call_Compare_maps(Ag_file_dt128, dust_file=dust_file, CO_file=CO_file)
call_Compare_maps(Ag_file_sim16, dust_file=dust_file, CO_file=CO_file)
#plt.show()
