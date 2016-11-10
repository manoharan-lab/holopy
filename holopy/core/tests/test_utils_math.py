# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, Ryan McGorty, Anna Wang, Solomon Barkley
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
from ..utils import _ensure_array, ensure_listlike, mkdir_p
from ..math import rotate_points, rotation_matrix
from .common import get_example_data, assert_obj_close
from nose.plugins.attrib import attr
import tempfile
import os
import shutil

#Test math
def test_rotate_single_point():
    points = np.array([1.,1.,1.])
    assert_allclose(rotate_points(points, np.pi, np.pi, np.pi),
                    np.array([-1.,  1., -1.]), 1e-5)
    
def test_rotation_matrix_degrees():
    assert_allclose(rotation_matrix(180., 180., 180., radians = False), 
                    rotation_matrix(np.pi, np.pi, np.pi))

#test utils
@attr('fast')
def test_ensure_array():
    assert_equal(_ensure_array(1.0), np.array([1.0]))
    assert_equal(_ensure_array([1.0]), np.array([1.0]))
    assert_equal(_ensure_array(np.array([1.0])), np.array([1.0]))

def test_ensure_listlike():
    assert ensure_listlike(None) == []

def test_mkdir_p():
    tempdir = tempfile.mkdtemp()
    mkdir_p(os.path.join(tempdir, 'a', 'b'))
    mkdir_p(os.path.join(tempdir, 'a', 'b'))
    shutil.rmtree(tempdir)
