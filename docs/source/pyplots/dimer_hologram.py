"""
dimer_hologram.py

Plot a dimer hologram to insert into the documentation

Author:
Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

import scipy
import numpy
import os
import sys

import pylab
from scipy.misc.pilutil import imresize
from scipy.ndimage import gaussian_filter
from scipy.signal import resample

sys.path.append('../../../')
import holopy
from holopy.hologram import Hologram
from holopy import optics

holofile = '../../../tests/standarddimer.npy'
optical_train = optics.Optics(wavelen = 658e-9, index = 1.33, 
                              pixel_scale = 1e-7)
h = Hologram(numpy.load(holofile), optics=optical_train)

pylab.gray()
pylab.imshow(h)


    
