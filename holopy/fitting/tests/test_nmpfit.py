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
from __future__ import division

import numpy as np

from ...scattering.theory import Mie
from ...scattering.scatterer import Sphere
from ..third_party import nmpfit
from ...core import Optics
from ...core.process import normalize
from ...core.tests.common import assert_parameters_allclose, get_example_data

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

optics = Optics(wavelen=wavelen, index=index,
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

holo = normalize(get_example_data('image0001.npy', optics))


# define the residual function
def residfunct(p, fjac = None):
    # nmpfit calls residfunct w/fjac as a kwarg, we ignore

    sphere = Sphere(n=p[0]+n_particle_imag*1j, r=p[1], center = p[2:5])
    calculated = Mie.calc_holo(sphere, holo, scaling=p[5])

    status = 0
    derivates = holo - calculated

    return([status, derivates.ravel()])

def test_nmpfit():
    fitresult = nmpfit.mpfit(residfunct, parinfo = parinfo, ftol = ftol,
                             xtol = xtol, gtol = gtol, damp = damp,
                             maxiter = maxiter, quiet = quiet)

    assert_parameters_allclose(fitresult.params,
                               gold_single[np.array([0,2,3,4,5,6])], rtol=1e-3)
