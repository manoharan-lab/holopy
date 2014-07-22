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
Test construction and manipulation of CSG Scatterer objects.

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
'''
import holopy as hp
from holopy.scattering.scatterer import Sphere, Difference
from holopy.scattering.scatterer.csg import CsgScatterer
from holopy.scattering.theory import DDA
from holopy.core import ImageSchema, Optics
from holopy.core.tests.common import verify

import numpy as np
from numpy.testing import assert_allclose


def test_csg_construction():
    s = Sphere(n = 1.6, r=.5, center=(0, 0, 0))
    st = s.translated(.4, 0, 0)
    pacman = Difference(s, st)
    assert_allclose(pacman.bounds, [(-.5, .5), (-.5, .5), (-.5, .5)])

def test_csg_dda():
    s = Sphere(n = 1.6, r=.1, center=(5, 5, 5))
    st = s.translated(.03, 0, 0)
    pacman = Difference(s, st)
    sch = ImageSchema(10, .1, Optics(.66, 1.33, (0, 1)))
    h = DDA.calc_holo(pacman, sch)
    verify(h, 'dda_csg')

    hr = DDA.calc_holo(pacman.rotated(np.pi/2, 0, 0), sch)
    rotated_pac = pacman.rotated(np.pi/2, 0, 0)
    verify(h/hr, 'dda_csg_rotated_div')
