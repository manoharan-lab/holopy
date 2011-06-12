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
Some code to test the refactoring of the mie module.  Compares output
of the functions from the new class structure to output from the
legacy (non-method) functions.  Can delete this file once refactoring
is complete.

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
'''

import numpy as np
import os
import string
import pylab

import holopy
#from holopy.model.theory import mie_fortran as mie
from holopy.model.theory import Mie
from holopy.model.scatterer import Sphere, SphereCluster
from holopy.third_party import nmpfit
from holopy.process import normalize

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

theory = Mie(imshape = 256, optics=optics)

    # mie.forward_holo(im.shape[0], optics, n_particle_real,
    #                        n_particle_imag, radius, x, y,
    #                        z, scaling_alpha)
    #calculated = mie.forward_holo(holo.shape[0], optics, p[0],
    #                              n_particle_imag, p[1], p[2], p[3],
    #                              p[4], p[5])
sphere = Sphere(n=n_particle_real + n_particle_imag*1j, r=radius, 
                x=x, y=y, z=z)
calculated = theory.calc_holo(sphere, alpha=scaling_alpha)
pylab.imshow(calculated)

