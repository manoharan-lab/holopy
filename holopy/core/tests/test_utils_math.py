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

from nose.plugins.attrib import attr
import tempfile
import os
import shutil
from multiprocessing.pool import Pool

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import xarray as xr
from schwimmbad import MultiPool, SerialPool, pool

from holopy.core.utils import (ensure_array, ensure_listlike, 
                                            mkdir_p, choose_pool)
from holopy.core.math import rotate_points, rotation_matrix
from holopy.core.tests.common import assert_obj_close, get_example_data


#Test math
@attr("fast")
def test_rotate_single_point():
    points = np.array([1.,1.,1.])
    assert_allclose(rotate_points(points, np.pi, np.pi, np.pi),
                    np.array([-1.,  1., -1.]), 1e-5)

@attr("fast")
def test_rotation_matrix_degrees():
    assert_allclose(rotation_matrix(180., 180., 180., radians = False),
                    rotation_matrix(np.pi, np.pi, np.pi))

#test utils
@attr('fast')
def test_ensure_array():
    assert_equal(ensure_array(1.0), np.array([1.0]))
    assert_equal(ensure_array([1.0]), np.array([1.0]))
    assert_equal(ensure_array(np.array([1.0])), np.array([1.0]))
    len(ensure_array(1.0))
    len(ensure_array(np.array(1.0)))
    len(ensure_array([1.0]))
    len(ensure_array(False))
    len(ensure_array(xr.DataArray([12],dims='a',coords={'a':['b']})))
    len(ensure_array(xr.DataArray([12],dims='a',coords={'a':['b']}).sel(a='b')))
    len(ensure_array(xr.DataArray(12)))


def test_choose_pool():
    class dummy():
        def map():
            return None
    assert isinstance(choose_pool(None), SerialPool)
    assert isinstance(choose_pool(2), MultiPool)
    assert isinstance(choose_pool('all'), MultiPool)
    assert isinstance(choose_pool('auto'), (pool.BasePool, Pool))
    assert not isinstance(choose_pool(dummy), (pool.BasePool, Pool))


@attr('fast')
def test_ensure_listlike():
    assert ensure_listlike(None) == []


@attr("fast")
def test_mkdir_p():
    tempdir = tempfile.mkdtemp()
    mkdir_p(os.path.join(tempdir, 'a', 'b'))
    mkdir_p(os.path.join(tempdir, 'a', 'b'))
    shutil.rmtree(tempdir)
