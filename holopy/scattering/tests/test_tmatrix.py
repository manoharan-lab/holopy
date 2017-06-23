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
from __future__ import division


from nose.tools import assert_raises
from numpy.testing import assert_almost_equal
import numpy as np
from nose.tools import with_setup
from nose.plugins.attrib import attr
from ...scattering.errors import InvalidScatterer
from ..scatterer import Sphere, Axisymmetric

from ...core import ImageSchema, Optics
from ..theory import Mie, Tmatrix
from .common import assert_allclose, verify

import os.path

schema = ImageSchema(shape = 200, spacing = .1,
                     optics = Optics(wavelen = .660, index = 1.33,
                                     polarization = [1,0]))

@attr('medium')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_tmat_sphere():
    sc = Sphere(n=1.59, r=0.9, center=(7, 8, 30))
    #assert_raises(InvalidScatterer, Sphere, n=1.59, r=0.9, center=(0, 0))
    sct = Axisymmetric(n=1.59, r=(0.9,0.9), center=(7, 8, 30))
    mie_holo = Mie.calc_holo(sc, schema)
    tmat_holo = Tmatrix.calc_holo(sct, schema)
    assert_allclose(mie_holo, tmat_holo, rtol=.0015)

'''
def test_spheroid():
    schema = ImageSchema(100, .1, Optics(.66, 1.33, (1, 0)))
    s = Spheroid(n = 1.5, r = [.4, 1.], rotation = (-np.pi/2, np.pi/2),
                    center = (5, 5, 25))
    dda_holo = DDA.calc_holo(s, schema)
    st = Axisymmetric(n = 1.5, r = [.4, 1.], rotation = (-np.pi/2, np.pi/2),
                    center = (5, 5, 25))
    tmat_holo = Tmatrix.calc_holo(st, schema)
    assert_allclose(mie_holo, DDA_holo, rtol=.0015)
'''
