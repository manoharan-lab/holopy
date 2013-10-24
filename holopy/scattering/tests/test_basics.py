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
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from ...core import Optics, ImageSchema, Grid
from ..scatterer import Sphere
from ..theory import Mie
from ...core.tests.common import assert_obj_close

# small tests against results from the previous version of holopy

def test_positions_kr_theta_phi():
    o = Optics(wavelen=.66, index=1.33, polarization = (1, 0))
    t = ImageSchema(shape = (2,2), spacing = .1, optics = o)
    pos = t.positions.kr_theta_phi((0,0,1), o)
    assert_allclose(pos, np.array([[ 12.66157039,   0.        ,   0.        ],
       [ 12.72472076,   0.09966865,   1.57079633],
       [ 12.72472076,   0.09966865,   0.        ],
       [ 12.78755927,   0.1404897 ,   0.78539816]]))

def test_calc_field():
    o = Optics(wavelen=.66, index=1.33, polarization = (1,0))
    s = Sphere(n=1.59, r=.5, center=(0,0,1))
    t = ImageSchema(shape  = (2,2), spacing = .1, optics = o)
    thry = Mie(False)
    f = thry.calc_field(s, t)
    assert_equal(t.optics, f.optics)
    assert_equal(t.spacing, f.spacing)
    assert_allclose(f, np.array([[[ -3.95866810e-01 +2.47924378e+00j,
                                     0.00000000e+00 +0.00000000e+00j,
                                     0.00000000e+00 -0.00000000e+00j],
                                  [ -4.91260953e-01 +2.32779296e+00j,
                                     9.21716363e-20 -5.72226912e-19j,
                                     2.99878926e-18 -1.41959276e-17j]],

                                 [[ -4.89755627e-01 +2.31844748e+00j,
                                     0.00000000e+00 +0.00000000e+00j,
                                     4.89755627e-02 -2.31844748e-01j],
                                  [ -5.71886751e-01 +2.17145168e+00j,
                                     1.72579090e-03 -8.72241140e-03j,
                                     5.70160960e-02 -2.16272927e-01j]]]),
                    atol=1e-8)

def test_calc_holo():
    o = Optics(wavelen=.66, index=1.33, polarization = (1, 0))
    s = Sphere(n=1.59, r=.5, center=(0,0,1))
    t = ImageSchema(shape  = (2,2), spacing = .1, optics = o)
    thry = Mie(False)
    h = thry.calc_holo(s, t)
    assert_obj_close(h.positions, t.positions)
    assert_allclose(h, np.array([[ 6.51162661,  5.67743548],
                                 [ 5.63554802,  4.89856241]]))

def test_calc_intensity():
    o = Optics(wavelen=.66, index=1.33, polarization = (1,0))
    s = Sphere(n=1.59, r=.5, center=(0,0,1))
    t = ImageSchema(shape  = (2,2), spacing = .1, optics = o)
    thry = Mie(False)
    i = thry.calc_intensity(s, t)
    assert_allclose(i, np.array([[ 6.30336023,  5.65995739],
                                 [ 5.61505927,  5.04233591]]))
