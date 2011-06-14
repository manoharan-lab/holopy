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
Test cython Mie calculations and python interface

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
'''

import numpy as np
import os
import string
import pylab

from nose.tools import raises, assert_raises
from numpy.testing import assert_, assert_equal, \
    assert_array_almost_equal, assert_array_equal
from nose.tools import with_setup

import holopy
from holopy.model import mie

# define optical train
wavelen = 658e-9
ypolarization = [0., 1.0] # y-polarized
xpolarization = [1.0, 0.] # x-polarized
divergence = 0
pixel_scale = [.1151e-6, .1151e-6]
index = 1.33

yoptics = holopy.optics.Optics(wavelen=wavelen, index=index,
                              pixel_scale=pixel_scale,
                              polarization=ypolarization,
                              divergence=divergence)

xoptics = holopy.optics.Optics(wavelen=wavelen, index=index,
                              pixel_scale=pixel_scale,
                              polarization=xpolarization,
                              divergence=divergence)

scaling_alpha = .6
radius = .85e-6
n_particle_real = 1.59
n_particle_imag = 1e-4
x = .576e-05
y = .576e-05
z = 15e-6

imshape = 128

def test_mie_polarization():
    xholo = mie.forward_holo(imshape, xoptics, n_particle_real,
                             n_particle_imag, radius, x, y, z,
                             scaling_alpha)
    yholo = mie.forward_holo(imshape, yoptics, n_particle_real,
                             n_particle_imag, radius, x, y, z,
                             scaling_alpha)
    
    # the two arrays should not be equal
    try:
        assert_array_almost_equal(xholo, yholo)
    except AssertionError:
        pass
    else:
        raise AssertionError("Holograms computed for both x- and y-polarized light are too similar.")

    return xholo, yholo
