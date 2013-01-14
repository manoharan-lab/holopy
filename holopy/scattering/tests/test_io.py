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
Test file IO of scatterpy objects

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
'''
from __future__ import division

from ..scatterer import Sphere
from ...core.tests.common import assert_read_matches_write
from ..scatterer import Sphere, Spheres

    
def test_scatterer_io():
    s = Sphere()
    assert_read_matches_write(s)

    s1 = Sphere(1.59, .5, [1, 1, 2])
    s2 = Sphere(1.59, .5, [1, 3, 2])
    sc = Spheres([s1, s2])

    assert_read_matches_write(sc)
