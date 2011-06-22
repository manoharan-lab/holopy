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
Test fitting a dimer hologram using nmpfit without any wrapping.

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Rebecca W. Perry <rperry@seas.harvard.edu>
'''

import numpy as np
import os
import string
import pylab

import holopy
#from holopy.model.theory import mie_fortran as mie
from holopy.model.theory import tmatrix_dimer
from holopy.model.scatterer import SphereDimer
from holopy.third_party import nmpfit
from holopy.process import normalize

# these are the exact values; should correspond to fit results
# in order: real index (2), imag index (2), radius (2) , 
# x (1), y (1), z (1), alpha (1), angles (2), gap (1), fit status (1) 
gold_dimerslow = np.array([1.603,1.603,0.0001,0.0001,6.857e-7,6.964e-7,
    1.700e-5,1.739e-5,2.093e-5,1,-29.78,-13.83,0.000,16.72])

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
scaling_alpha = .635
radius_1 = .65e-6
radius_2 = .65e-6
n_particle_real_1 = 1.59
n_particle_imag_1 = .00001
n_particle_real_2 = 1.59
n_particle_imag_2 = .00001
x_com = 17.3e-6
y_com = 17.3e-6
z_com =  20.7e-6
euler_beta = -28.5
euler_gamma = -14.87
gap_distance = 10e-9


# parameters for fitter
ftol = 1e-10
xtol = 1e-10
gtol = 1e-10
damp = 0
maxiter = 100
quiet = False

# parinfo to pass to MPFIT (fitting for 10 parameters for dimer)
parinfo = [{'parname':'n_particle_real_1',
          'limited': [True, False],
          'step': 1e-4,
          'limits': [1.0, 0],
          'value': n_particle_real_1},
          {'parname':'n_particle_real_2',
          'limited': [True, False],
          'step': 1e-4,
          'limits': [1.0, 0],
          'value': n_particle_real_2},
          {'parname': 'radius_1',
           'limited': [True, False],
           'limits': [0.0, 0],
           'step': 100e-9,
           'mpmaxstep': 1e-6,
           'value': radius_1},
          {'parname': 'radius_2',
           'limited': [True, False],
           'limits': [0.0, 0],
           'step': 100e-9,
           'mpmaxstep': 1e-6,
           'value': radius_2},
          {'parname': 'x_com',
           'limited': [False, False],
           'limits': [0, 0],
           'step': 100e-9,
           'mpmaxstep': 1e-6,
           'value': x_com},
          {'parname': 'y_com',
           'limited': [False, False],
           'limits': [0, 0],
           'step': 100e-9,
           'mpmaxstep': 1e-6,
           'value': y_com},
          {'parname':'z_com',
           'limited': [False, False],
           'limits': [0, 0],
           'step': 100e-9,
           'mpmaxstep': 1e-6,
           'value': z_com},
          {'parname': 'scaling_alpha',
           'limited': [True, True],
           'limits': [0.5, 1.0],
           'value': scaling_alpha},
          {'parname': 'euler_beta',
           'limited': [False, False],
           'limits': [0, 0],
           'step': .001,
           'mpmaxstep': 10,
           'value': euler_beta},
          {'parname':'euler_gamma',
           'limited': [False, False],
           'limits': [0, 0],
           'step': .001,
           'mpmaxstep': 10,
           'value': euler_gamma},
          {'parname':'gap_distance',
           'limited': [False, False],
           'limits': [0, 0],
           'step': 1e-9,
           'mpmaxstep': 10e-9,
           'value': gap_distance},
 ] 

path = os.path.abspath(holopy.__file__)
path = string.rstrip(path, chars='__init__.pyc')+'tests/exampledata/'

# load the target file
# be sure to normalize, or fit will fail
holo = normalize(holopy.load(path + 'image0002.npy'))

# define the residual function
def residfunct(p, fjac = None):
    # nmpfit calls residfunct w/fjac as a kwarg, we ignore
    # syntax:
    # tmatrix_dimer.forward_holo(im.shape[0], optics, n_particle_real_1,
    #                        n_particle_real_2, n_particle_imag_1,
    #                        n_particle_imag_2, radius_1, radius_2, 
    #                        x_com, y_com, z_com, scaling_alpha, 
    #                        euler_beta, euler_gamma, gap_distance, 
    #                        tmat_dict=None, old_coords=False, 
    #                        dimensional=True)


    calculated = tmatrix_dimer.forward_holo(holo.shape[0],optics,p[0],
        p[1], n_particle_imag_1, n_particle_imag_2, p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10])

    status = 0
    derivates = holo - calculated

    # print sum of squares and param values for debugging
    print "sum of squares: ", np.dot(derivates.ravel(), derivates.ravel())
    print "params: ", p
    return([status, derivates.ravel()])



def residfunctoverlapallowed(p, fjac = None):
    # nmpfit calls residfunct w/fjac as a kwarg, we ignore

    if p[10] >= 0:
        # nmpfit calls residfunct w/fjac as a kwarg, we ignore
        # syntax:
        # tmatrix_dimer.forward_holo(im.shape[0], optics, n_particle_real_1,
        #                        n_particle_real_2, n_particle_imag_1,
        #                        n_particle_imag_2, radius_1, radius_2, 
        #                        x_com, y_com, z_com, scaling_alpha, 
        #                        euler_beta, euler_gamma, gap_distance, 
        #                        tmat_dict=None, old_coords=False, 
        #                        dimensional=True)

        calculated = tmatrix_dimer.forward_holo(holo.shape[0],optics,p[0],
            p[1], n_particle_imag_1, n_particle_imag_2, p[2], p[3], 
            p[4], p[5], p[6], p[7], p[8], p[9], p[10])
        status = 0
        derivates = holo - calculated

    else:
        status = 0
        derivates = np.ones([holo.shape[0],holo.shape[0]])
    
    # print sum of squares and param values for debugging
    print "sum of squares: ", np.dot(derivates.ravel(), derivates.ravel())
    print "gold standard sum of squares: 16.72"
    print "params: ", p
    return([status, derivates.ravel()])

fitresult = nmpfit.mpfit(residfunctoverlapallowed, parinfo = parinfo, ftol = ftol,
                         xtol = xtol, gtol = gtol, damp = damp,
                         maxiter = maxiter, quiet = quiet)
# or uncomment this fit to try fitting without hard limits on the gap distance
#fitresult = nmpfit.mpfit(residfunctoverlapallowed, parinfo = parinfo, ftol = ftol,
#                         xtol = xtol, gtol = gtol, damp = damp,
#                         maxiter = maxiter, quiet = quiet)

print "Fit finished with status ", fitresult.status
print "Difference from expected values: ", \
      fitresult.params - gold_dimerslow[np.array([0,1,4,5,6,7,8,9,10,11,12])] 

