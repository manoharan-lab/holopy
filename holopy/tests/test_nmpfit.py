# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca
# W. Perry, Jerome Fung, and Ryan McGorty
#
# This file is part of Holopy.
#
# Holopy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Holopy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Holopy.  If not, see <http://www.gnu.org/licenses/>.
'''
Test fitting a hologram using nmpfit without any wrapping
'''

import numpy as np
import os
import string
import pylab

import holopy
#from scatterpy.theory import mie_fortran as mie
from scatterpy.theory import mie
from scatterpy.theory import Mie
from scatterpy.scatterer import Sphere, SphereCluster
from holopy.third_party import nmpfit
from holopy.process import normalize

# these are the exact values; should correspond to fit results
# in order: real index, imag index, radius , x, y, z, alpha, fnorm, fit status
gold_single = np.array([1.5768, 0.0001, 6.62e-7, 5.54e-6, 5.79e-6,
                        14.2e-6, 0.6398, 7.119, 2]) 

# Particle parameters for input file
medium_index = 1.33

# define optical train
wavelen = 658e-9
polarization = [0., 1.0]
divergence = 0
pixel_scale = [.1151e-6, .1151e-6]
index = 1.33

optics = holopy.optics.Optics(wavelen=wavelen, index=index,
                              pixel_scale=pixel_scale,
                              polarization=polarization,
                              divergence=divergence)

# initial guess
scaling_alpha = .6
radius = .85e-6
n_particle_real = 1.59
n_particle_imag = 1e-4
x = .576e-05
y = .576e-05
z = 15e-6

# parameters for fitter
ftol = 1e-10
xtol = 1e-10
gtol = 1e-10
damp = 0
maxiter = 100
quiet = False

# parinfo to pass to MPFIT
parinfo = [{'parname':'n_particle_real',
          'limited': [True, False],
          'step': 1e-4,
          'limits': [1.0, 0],
          'value': n_particle_real},
          {'parname': 'radius',
           'limited': [True, False],
           'limits': [0.0, 0],
           'step': 100e-9,
           'mpmaxstep': 1e-6,
           'value': radius},
          {'parname': 'x',
           'limited': [False, False],
           'limits': [0, 0],
           'step': 100e-9,
           'mpmaxstep': 1e-6,
           'value': x},
          {'parname': 'y',
           'limited': [False, False],
           'limits': [0, 0],
           'step': 100e-9,
           'mpmaxstep': 1e-6,
           'value': y},
          {'parname':'z',
           'limited': [False, False],
           'limits': [0, 0],
           'step': 100e-9,
           'mpmaxstep': 1e-6,
           'value': z},
          {'parname': 'scaling_alpha',
           'limited': [True, True],
           'limits': [0.0, 1.0],
           'value': scaling_alpha}] 

path = os.path.abspath(holopy.__file__)
path = string.rstrip(path, chars='__init__.pyc')+'tests/exampledata/'

# load the target file
# be sure to normalize, or fit will fail
holo = normalize(holopy.load(path + 'image0001.npy'))

# or uncomment to fit a calculated hologram (residual should be zero
# if fit succeeds)
#holo = mie.forward_holo(100, optics, gold_single[0], gold_single[1],
#                        gold_single[2], gold_single[3],
#                        gold_single[4], gold_single[5], gold_single[6])

theory = Mie(imshape = holo.shape, optics=optics)

# define the residual function
def residfunct(p, fjac = None):
    # nmpfit calls residfunct w/fjac as a kwarg, we ignore

    # uncomment for old interface to mie calculation
    #calculated = mie.forward_holo(holo.shape[0], optics, p[0],
    #                              n_particle_imag, p[1], p[2], p[3],
    #                              p[4], p[5])

    # below uses new class-based interface to mie calculation
    sphere = Sphere(n=p[0]+n_particle_imag*1j, r=p[1], x=p[2], y=p[3], 
                    z=p[4])
    calculated = theory.calc_holo(sphere, alpha=p[5])

    status = 0
    derivates = holo - calculated

    # print sum of squares and param values for debugging
    print "sum of squares: ", np.dot(derivates.ravel(), derivates.ravel())
    print "params: ", p
    return([status, derivates.ravel()])

def test_nmpfit():
    fitresult = nmpfit.mpfit(residfunct, parinfo = parinfo, ftol = ftol,
                             xtol = xtol, gtol = gtol, damp = damp,
                             maxiter = maxiter, quiet = quiet)

    print "Fit finished with status ", fitresult.status
    print "Difference from expected values: ", \
        fitresult.params - gold_single[np.array([0,2,3,4,5,6])] 


