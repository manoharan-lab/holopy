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
from nose.plugins.skip import SkipTest
from .. import Tmatrix, DDA, Sphere, Spheroid, Ellipsoid, Cylinder, calc_holo as calc_holo_external
from ..errors import DependencyMissing
from ...core import detector_grid, update_metadata
from ...core.tests.common import verify

def calc_holo(schema, scatterer, medium_index=None, illum_wavelen=None,**kwargs):
    try:
        return calc_holo_external(schema, scatterer, medium_index, illum_wavelen, **kwargs)
    except DependencyMissing:
        raise SkipTest()


schema = update_metadata(detector_grid(shape = 200, spacing = .1),
                     illum_wavelen = .660, medium_index = 1.33,
                                     illum_polarization = [1,0])

@attr('slow')
def test_sphere():
    s = Sphere(n=1.59, r=0.9, center=(2, 2, 80))
    mie_holo = calc_holo(schema, s)
    tmat_holo = calc_holo(schema, s, theory=Tmatrix)
    assert_allclose(mie_holo, tmat_holo, atol=.008)
    
def test_spheroid():
    s = Spheroid(n = 1.5, r = [.4, 1.], rotation = (0, np.pi/2, np.pi/2), center = (5, 5, 15))
    holo = calc_holo(schema, s)
    verify(holo, 'tmatrix_spheroid')

def test_cylinder():
    s = Cylinder(n = 1.5, d=.8, h=2, rotation = (0, np.pi/2, np.pi/2), center = (5, 5, 15))
    holo = calc_holo(schema, s)
    verify(holo, 'tmatrix_cylinder')

def test_vs_dda():
    s = Spheroid(n = 1.5, r = [.4, 1.], rotation = (0, np.pi/2, np.pi/2), center = (5, 5, 50))
    try:
        dda_holo = calc_holo(schema, s, theory=DDA)
    except DependencyMissing:
        raise SkipTest()
    tmat_holo = calc_holo(schema, s, theory=Tmatrix)
    assert_allclose(dda_holo, tmat_holo, atol=.05)
    
