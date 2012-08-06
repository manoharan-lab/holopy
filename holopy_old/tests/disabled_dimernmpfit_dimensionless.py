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
from scatterpy.theory import tmatrix_dimer
from scatterpy.scatterer import SphereDimer
from holopy.third_party import nmpfit
from holopy.process import normalize

# these are the exact values; should correspond to fit results
# in order:
gold_dimerslow = np.array([1.20137044036732, 8.73443736434356, 8.87082641747115, 216.523192268157, 221.530284256344, 266.569136122325, 1, -29.7801476788, -13.8296930386, 6.93889390391118E-018, 16.7224924836])

# Particle parameters for input file
medium_index = 1.334

# define optical train
# k = 2*pi*1.334/658e-9
wavelen = 8.3817692
polarization = [0., 1.0]
divergence = 0
pixel_scale = [4.39469662, 4.39469662]
index = 1.334

optics = holopy.optics.Optics(wavelen=wavelen, index=index,
                              pixel_scale=pixel_scale,
                              polarization=polarization,
                              divergence=divergence)

# initial guess
scaling_alpha = 0.63500000000000001
radius_1 = 8.2798631912696354 
radius_2 = 8.2798631912696354
n_particle_real_1 = 1.1919040479760119 
n_particle_imag_1 = .00001
x_com = 220.37174339840723
y_com = 220.37174339840723
z_com = 263.68179701427914 
euler_beta = -28.5
euler_gamma = -14.869999999999999
gap_distance = 0.12738251063491748

# parameters for fitter
ftol = 1e-10
xtol = 1e-10
gtol = 1e-10
damp = 0
maxiter = 60
quiet = False

# parinfo to pass to MPFIT (fitting for 10 parameters for dimer)
parinfo = [{'parname':'n_particle_real_1',
          'limited': [True, False],
          'step': 1e-4,
          'limits': [1.0, 0],
          'value': n_particle_real_1},
          {'parname': 'radius_1',
           'limited': [True, False],
           'limits': [0.0, 0],
           'mpmaxstep': 1e-6,
           'value': radius_1},
          {'parname': 'radius_2',
           'limited': [True, False],
           'limits': [0.0, 0],
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
           'limits': [0.0, 1.0],
           'value': scaling_alpha},
          {'parname': 'euler_beta',
           'limited': [True, True],
           'limits': [-90., 270.],
           'step': .001,
           'mpmaxstep': 90,
           'value': euler_beta},
          {'parname':'euler_gamma',
           'limited': [False, False],
           'limits': [0, 0],
           'mpmaxstep': 90,
           'value': euler_gamma},
          {'parname':'gap_distance',
           'limited': [True, False], # change True to False to allow the gap to go negative
           'limits': [0.0, 0],
           'mpmaxstep': .13,      
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
        p[0], n_particle_imag_1, n_particle_imag_1, p[1], p[2], p[3], 
        p[4], p[5], p[6], p[7], p[8], p[9], dimensional=False)

    status = 0
    derivates = holo - calculated

    # print sum of squares and param values for debugging
    print "sum of squares: ", np.dot(derivates.ravel(), derivates.ravel())
    return([status, derivates.ravel()])

def residfunctoverlapallowed(p, fjac = None):
    # nmpfit calls residfunct w/fjac as a kwarg, we ignore

    if p[9] >= 0:
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
            p[0], n_particle_imag_1, n_particle_imag_1, p[1], p[2], p[3], 
            p[4], p[5], p[6], p[7], p[8], p[9], dimensional=False)
            
        status = 0
        derivates = holo - calculated

    else:
        status = 0
        derivates = np.ones([holo.shape[0],holo.shape[0]])*1000./(holo.shape[0]**2)

    # print sum of squares and param values for debugging
    print "sum of squares: ", np.dot(derivates.ravel(), derivates.ravel())
    return([status, derivates.ravel()])

def test_dimerfitdimensionless():
    fitresult = nmpfit.mpfit(residfunct, parinfo = parinfo, ftol = ftol,
                             xtol = xtol, gtol = gtol, damp = damp,
                             maxiter = maxiter, quiet = quiet)
    # comment the previous three lines and uncomment the three lines below and get 
    # rid of the limits on the gap distance to try fitting without hard limits 
    # on the gap distance
    '''fitresult = nmpfit.mpfit(residfunctoverlapallowed, parinfo = parinfo, ftol = ftol,
                         xtol = xtol, gtol = gtol, damp = damp,
                         maxiter = maxiter, quiet = quiet)'''

    print "gold standard sum of squares: 16.7224924836"
    print "Fit finished with status: ", fitresult.status
    print "final params: ", fitresult.params
    print "Difference from expected values: ", \
        fitresult.params - gold_dimerslow[0:10] 
