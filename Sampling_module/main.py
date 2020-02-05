"""
The main program for sampling and fitting models to the planck data.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import sys, time
#import h5py

# import the modules
from comp_intensity_mod import Model
from metropolis_mod import Initialize, MetropolisHastings
from stat_mod import logLikelihood, logPrior, Cov
import planck_map_mod as planck


def main():
    pass
