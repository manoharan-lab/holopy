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
from numpy.testing import assert_equal
from nose.plugins.attrib import attr

from ..helpers import _ensure_array, coord_grid


@attr('fast')
def test_ensure_array():
    assert_equal(_ensure_array(1.0), np.array([1.0]))
    assert_equal(_ensure_array([1.0]), np.array([1.0]))
    assert_equal(_ensure_array(np.array([1.0])), np.array([1.0]))

@attr('fast')
def test_coord_grid():
    assert_equal(coord_grid(([0,5],[0,5],[0,5])).shape, (5,5,5,3))
    assert_equal(coord_grid(([0,1],[0,1],[0,1]), .2).shape, (5, 5, 5, 3))
    assert_equal(coord_grid(([0,1],[0,1],[0,1]), .5),
                 np.array([[[[ 0. ,  0. ,  0. ],
                             [ 0. ,  0. ,  0.5]],
                            [[ 0. ,  0.5,  0. ],
                             [ 0. ,  0.5,  0.5]]],
                           [[[ 0.5,  0. ,  0. ],
                             [ 0.5,  0. ,  0.5]],
                            [[ 0.5,  0.5,  0. ],
                             [ 0.5,  0.5,  0.5]]]]))
