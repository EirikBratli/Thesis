"""      1         2         3          4          5          6         7          80
Program to reduce the parameter files from containing 1.7 billion objects to less 
than 87 mill objects. Considering distance uncertainties < 20%, only positive 
distances, magnitudes in G band between 6 and 18 mag.
"""

import numpy as np
import healpy as hp
import os, sys, glob
import h5py

def ToGaia_dir():
    os.chdir('../Gaia/data_v2/')
    dir = os.path.abspath(os.curdir)
    print(dir)

def FromGaia_dir():
    os.chdir('../../3D_extinction/')
    dir = os.path.abspath(os.curdir)
    #print(dir)

def Read_H5(file, name):
    """
    Read one .h5 file and return an array with the data
    Input:
    - file, string. Name of the file to read with extention
    - name, string. Data name of the .h5 file.
    Return:
    - data, sequence. Array with the data from the .h5 file 
    """
    #print(file, name)
    f = h5py.File(file, 'r')
    data = np.asarray(f[name])
    f.close()
    return(data)

def Write_H5(in_data, name, filename):
    """
    Write .h5 file. Save to a data folder: '../3D_extinction/Data/'
    
    Input:
    - in_data, sequence. The list/array of data to be written to file
    - name, string. Data name of the data
    - filename, string. The name of the output file
    """
    #print(in_data[0].type())
    f = h5py.File('Data/' + filename, 'w')
    if isinstance(in_data[0], (int, np.int64)) == True:
        print('- is integer')
        f.create_dataset(name, data=in_data.astype(int))
    elif isinstance(in_data[0], (float, np.float32)) == True:
        print('- is float')
        f.create_dataset(name, data=in_data.astype(dtype=np.float32))
    elif isinstance(in_data[0], str) == True:
        print('- is string')
        f.create_dataset(name, data=in_data.astype(str))
    else:
        print(in_data.type())
        print('Type of object is not, "int", "float" or "str".')

    f.close()

def ReduceData(files, names):
    """
    Main function: get the relevant parameter files, find the relevant objects and 
    their index. Write new reduced parameter files. Do only full length parameter
    files. Need to know the file names to reduce and type those into a input list.

    Input:
    - files, sequence. List of files names to write, no extension
    - names, sequence. List of parameter names to read. 
    
    """
    if len(files) == len(names):
        pass
    else:
        raise('Length of input list are unequal: ({}), ({})'.format(len(files), len(names)))
        sys.exit()

    # read in data:
    path = 'mn/stornext/d16/cmbco/pasiphae/eiribrat/Gaia/data_v2/'
    #files = glob.glob('{path}*.h5')
    N0 = 1692919135
    names = ['ra', 'dec', 'parallax', 'parallax_error', 'phot_g_mean_mag',\
             'phot_bp_mean_mag', 'phot_rp_mean_mag', 'a_g_val',\
             'a_g_percentile_lower', 'a_g_percentile_upper', 'e_bp_min_rp_val',\
             'e_bp_min_rp_percentile_lower', 'e_bp_min_rp_percentile_upper']
    N1 = len(names)
    datas = np.zeros((N1, N0))
    #for i, fil in enumerate(files):
    #    datas[i, :] = Read_H5(fil, )

    #readpath = 'data_v2/'
    ToGaia_dir()
    for i in range(N1):
        print('Read parameter: {}'.format(names[i]))
        datas[i, :] = Read_H5('{}.h5'.format(files[i]), names[i])
    """
    datas[0, :] = Read_H5(path + 'RightAscension.h5', names[0])
    datas[1, :] = Read_H5(path + 'Declination.h5', names[1])
    datas[2, :] = Read_H5(path + 'Parallax.h5', names[2])
    datas[3, :] = Read_H5(path + 'Parallax_error.h5', names[3])
    datas[4, :] = Read_H5(path + 'Mean_mag_G.h5', names[4])
    datas[5, :] = Read_H5(path + 'Mean_mag_BP.h5', names[5])
    datas[6, :] = Read_H5(path + 'Mean_mag_RP.h5', names[6])
    datas[7, :] = Read_H5(path + 'A_G.h5', names[7])
    datas[8, :] = Read_H5(path + 'A_G_lower.h5', names[8])
    datas[9, :] = Read_H5(path + 'A_G_upper.h5', names[9])
    datas[10,:] = Read_H5(path + 'Reddening.h5', names[10])
    datas[11,:] = Read_H5(path + 'Reddening_low.h5', names[11])
    datas[12,:] = Read_H5(path + 'reddening_upp.h5', names[12])
    """
    FromGaia_dir()
    for i in range(N1):
        if len(datas[i,:]) != N0:
            print('Length of data "{}" is too short/already reduced'.format(names[i]))
            sys.exit()

    # find the relevant objects
    perc = datas[3, :]/datas[2, :]
    perc = np.nan_to_num(perc)
    G = np.nan_to_num(datas[4, :])
    d = np.nan_to_num(datas[2,:])
    Ag = np.nan_to_num(datas[7, :])
    ind = np.where((perc < 0.2) & (Ag != 0) & (G < 18) & (d > 0))
    ind = ind[0]
    data = datas[:, ind]
    print(np.shape(data), type(ind[0]), np.max(d))
    #sys.exit()

    # write new files
    for i in range(N1):
        print('Write file: {}'.format(files[i]))
        Write_H5(data[i, :], names[i], files[i] + '_v2.h5')
        
    print('Write index file')
    Write_H5(ind, 'indices', 'Index_v2.h5')
    
    
    # End reduce data
    #################

def Reduce1file(file, name):
    """
    If need of reduction of one parameter file.
    """
    N0 = 1692919135
    path = ''#'mn/stornext/d16/cmbco/pasiphae/eiribrat/Gaia/data_v2/'

    # Get indices:
    ind = Read_H5('Data/Index_v2.h5', 'indices')
    print(len(ind))

    # Read data file
    ToGaia_dir()
    data = Read_H5(path + file + '.h5', name)
    print(data)
    FromGaia_dir()

    # Check if data is the correct size
    if len(data) != N0:
        print('Length of data "{name}" is too short/ already reduced')
        sys.exit()
    else:
        print('Reduce data')
        data = data[ind]
    # Write redused data to new file
    Write_H5(data, name, file + '_v2.h5')
    
    # end Reduce1file



#######################
#   Function calls    #
#######################

names = ['ra', 'dec', 'parallax', 'parallax_error', 'phot_g_mean_mag',\
             'phot_bp_mean_mag', 'phot_rp_mean_mag', 'a_g_val',\
             'a_g_percentile_lower', 'a_g_percentile_upper', 'e_bp_min_rp_val',\
             'e_bp_min_rp_percentile_lower', 'e_bp_min_rp_percentile_upper']

filenames = ['RightAscension', 'Declination', 'Parallax', 'Parallax_error',\
             'Mean_mag_G', 'Mean_mag_BP', 'Mean_mag_RP',\
             'Extinction', 'Extinction_lower', 'Extinction_upper',\
             'Reddening', 'Reddening_lower','Reddening_upper']

#ReduceData(filenames, names)
Reduce1file('gal_latitude', 'b')


#ToGaia_dir()
#p = Read_H5('Parallax.h5', 'parallax')
#FromGaia_dir()
#print(np.shape(p))
#print(p)
