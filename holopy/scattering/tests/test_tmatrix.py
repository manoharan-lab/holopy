# Copyright 2011-2013, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, and Ryan McGorty, Anna Wang
#
# This file is part of HoloPy.
#
# HoloPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HoloPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HoloPy.  If not, see <http://www.gnu.org/licenses/>.
'''
Tests non-spherical T-matrix code calculations against Mie code

.. moduleauthor:: Anna Wang <annawang@seas.harvard.edu>
'''


from numpy.testing import assert_raises, assert_allclose

import numpy as np
from nose.plugins.attrib import attr
from .. import Sphere, Axisymmetric, Spheroid, Ellipsoid, calc_holo,

from ...core import detector_grid, update_metadata

schema = update_metadata(detector_grid(shape = 200, spacing = .1),
                     illum_wavelen = .660, medium_index = 1.33,
                                     illum_polarization = [1,0])

@attr('medium')
def test_tmat_sphere():
    sc = Sphere(n=1.59, r=0.9, center=(7, 8, 30))
    sct = Axisymmetric(n=1.59, r=(0.9,0.9), center=(7, 8, 30))
    mie_holo = calc_holo(schema, sc)
    tmat_holo = calc_holo(schema, sct)
    assert_allclose(mie_holo, tmat_holo, atol=.06)

'''
def test_spheroid():
    e = Ellipsoid(n=1.5, r=[.4, .4, .1],rotation=(0,-np.pi/2, np.pi/2), center = (5, 5, 25))
    s = Spheroid(n = 1.5, r = [.4, 1.], rotation = (-np.pi/2, np.pi/2), center = (5, 5, 25))
    st = Axisymmetric(n = 1.5, r = [.4, 1.], rotation = (-np.pi/2, np.pi/2),center = (5, 5, 25))
    dda_holo = calc_holo(schema, e)
    tmat_holo = calc_holo(schema, st)
    assert_allclose(mie_holo, DDA_holo, rtol=.0015)
'''
