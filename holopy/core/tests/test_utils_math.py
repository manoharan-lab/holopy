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

import os
import shutil
import unittest
import tempfile
import multiprocessing as mp
from importlib.util import find_spec

import numpy as np
from numpy.testing import assert_allclose
from nose.plugins.attrib import attr
import xarray as xr
try:
    from schwimmbad import MultiPool, pool, MPIPool
    _has_schwimmbad = True
except ModuleNotFoundError:
    _has_schwimmbad = False


from holopy.core.utils import (
    ensure_array, ensure_listlike, ensure_scalar, mkdir_p, dict_without,
    updated, repeat_sing_dims, choose_pool)
from holopy.core.math import (
    rotate_points, rotation_matrix, transform_cartesian_to_spherical,
    transform_spherical_to_cartesian, transform_cartesian_to_cylindrical,
    transform_cylindrical_to_cartesian, transform_cylindrical_to_spherical,
    transform_spherical_to_cylindrical, find_transformation_function,
    keep_in_same_coordinates)
from holopy.core.tests.common import assert_obj_close, get_example_data


TOLS = {'atol': 1e-14, 'rtol': 1e-14}

class DummyPool():
    def __init__(self, index_val):
        self.index_val = index_val
    def map():
        return None


class TestCoordinateTransformations(unittest.TestCase):
    @attr("fast")
    def test_transform_cartesian_to_spherical_returns_correct_shape(self):
        np.random.seed(12)
        xyz = np.random.randn(3, 10)
        rtp = transform_cartesian_to_spherical(xyz)
        self.assertTrue(rtp.shape == xyz.shape)

    @attr("fast")
    def test_transform_cartesian_to_spherical(self):
        np.random.seed(12)
        xyz = np.random.randn(3, 10)
        rtp = transform_cartesian_to_spherical(xyz)
        r_is_close = np.allclose(
            rtp[0],
            np.sqrt(np.sum(xyz**2, axis=0)),
            **TOLS)
        theta_is_close = np.allclose(
            rtp[1],
            np.arccos(xyz[2] / np.linalg.norm(xyz, axis=0)),
            **TOLS)
        phi_is_close = np.allclose(
            rtp[2],
            np.arctan2(xyz[1], xyz[0]) % (2 * np.pi),
            **TOLS)
        self.assertTrue(r_is_close)
        self.assertTrue(theta_is_close)
        self.assertTrue(phi_is_close)

    @attr("fast")
    def test_transform_cartesian_to_spherical_returns_phi_on_0_2pi(self):
        np.random.seed(12)
        xyz = np.random.randn(3, 10)
        rtp = transform_cartesian_to_spherical(xyz)
        phi = rtp[2]
        self.assertTrue(np.all(phi > 0))

    @attr("fast")
    def test_transform_cartesian_to_spherical_at_origin(self):
        xyz_0 = np.zeros((3, 1))
        rtp = transform_cartesian_to_spherical(xyz_0)
        xyz_1 = transform_spherical_to_cartesian(rtp)
        self.assertTrue(np.allclose(xyz_0, xyz_1, **TOLS))

    @attr("fast")
    def test_transform_spherical_to_cartesian(self):
        # check that spherical_to_cartesian is the inverse of cartesian_to_sph
        np.random.seed(12)
        xyz_0 = np.random.randn(3, 10)
        rtp = transform_cartesian_to_spherical(xyz_0)
        xyz_1 = transform_spherical_to_cartesian(rtp)
        self.assertTrue(np.allclose(xyz_0, xyz_1, **TOLS))

    @attr("fast")
    def test_transform_cartesian_to_cylindrical_returns_correct_shape(self):
        np.random.seed(12)
        xyz = np.random.randn(3, 10)
        rpz = transform_cartesian_to_cylindrical(xyz)
        self.assertTrue(rpz.shape == xyz.shape)

    @attr("fast")
    def test_transform_cartesian_to_cylindrical(self):
        np.random.seed(12)
        xyz = np.random.randn(3, 10)
        rpz = transform_cartesian_to_cylindrical(xyz)
        r_is_close = np.allclose(
            rpz[0],
            np.sqrt(xyz[0]**2 + xyz[1]**2),
            **TOLS)
        phi_is_close = np.allclose(
            rpz[1], np.arctan2(xyz[1], xyz[0]) % (2 * np.pi),
            **TOLS)
        z_is_close = np.allclose(xyz[2], rpz[2])
        self.assertTrue(r_is_close)
        self.assertTrue(phi_is_close)
        self.assertTrue(z_is_close)

    @attr("fast")
    def test_transform_cartesian_to_cylindrical_returns_phi_on_0_2pi(self):
        np.random.seed(12)
        xyz = np.random.randn(3, 10)
        rpz = transform_cartesian_to_cylindrical(xyz)
        phi = rpz[1]
        self.assertTrue(np.all(phi > 0))

    @attr("fast")
    def test_transform_cylindrical_to_cartesian(self):
        # check cylindrical_to_cartesian is the inverse of cartesian_to_cyl
        np.random.seed(12)
        xyz_0 = np.random.randn(3, 10)
        rpz = transform_cartesian_to_cylindrical(xyz_0)
        xyz_1 = transform_cylindrical_to_cartesian(rpz)
        self.assertTrue(np.allclose(xyz_0, xyz_1, **TOLS))

    @attr("fast")
    def test_transform_cylindrical_to_spherical(self):
        # Uses the pre-existing cartesian to cylindrical & spherical functions
        np.random.seed(12)
        xyz = np.random.randn(3, 20)

        rho_phi_z = transform_cartesian_to_cylindrical(xyz)
        r_theta_phi_true = transform_cartesian_to_spherical(xyz)
        r_theta_phi_check = transform_cylindrical_to_spherical(rho_phi_z)
        is_ok = np.allclose(r_theta_phi_true, r_theta_phi_check, **TOLS)
        self.assertTrue(is_ok)

    @attr("fast")
    def test_transform_spherical_to_cylindrical(self):
        # Uses the pre-existing cartesian to cylindrical & spherical functions
        np.random.seed(12)
        xyz = np.random.randn(3, 20)

        r_theta_phi = transform_cartesian_to_spherical(xyz)
        rho_phi_z_true = transform_cartesian_to_cylindrical(xyz)
        rho_phi_z_check = transform_spherical_to_cylindrical(r_theta_phi)
        is_ok = np.allclose(rho_phi_z_true, rho_phi_z_check, **TOLS)
        self.assertTrue(is_ok)

    @attr("fast")
    def test_find_transformation_function_returns_helpful_error(self):
        # This test will have to be changed if someone implements
        # spherical bipolar coordinates.
        self.assertRaises(
            NotImplementedError,
            find_transformation_function,
            'cartesian', 'spherical_bipolar')

    @attr("fast")
    def test_find_transformation_function(self):
        desired = [
            ('cartesian', 'spherical', transform_cartesian_to_spherical),
            ('cartesian', 'cylindrical', transform_cartesian_to_cylindrical),
            ('spherical', 'cartesian', transform_spherical_to_cartesian),
            ('cylindrical', 'cartesian', transform_cylindrical_to_cartesian),
            ('spherical', 'cylindrical', transform_spherical_to_cylindrical),
            ('cylindrical', 'spherical', transform_cylindrical_to_spherical),
            ]
        for initial, final, correct_method in desired:
            self.assertTrue(
                find_transformation_function(initial, final) is correct_method)

    @attr("fast")
    def test_keep_in_same_coordinates(self):
        np.random.seed(12)
        xyz = np.random.randn(3, 10)
        the_same = keep_in_same_coordinates(xyz)
        self.assertTrue(np.allclose(xyz, the_same, **TOLS))

    @attr("fast")
    def test_find_transformation_function_when_same(self):
        np.random.seed(12)
        xyz = np.random.randn(3, 10)
        for which in ['cartesian', 'spherical', 'cylindrical']:
            method = find_transformation_function(which, which)
            self.assertTrue(np.allclose(xyz, method(xyz), **TOLS))

    @attr("fast")
    def test_coordinate_transformations_work_when_z_is_a_scalar(self):
        # This just tests that the transformations work, not that they
        # are in the shape (N, 3), as some of the calculations prefer
        # to leave z as a scalar if it starts as one (e.g. mielens)
        np.random.seed(12)
        x, y = np.random.randn(2, 10)
        z = np.random.randn()

        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        versions_to_check = [
            ('cartesian', 'spherical', [x, y, z]),
            ('cartesian', 'cylindrical', [x, y, z]),
            ('cylindrical', 'cartesian', [rho, phi, z]),
            ('cylindrical', 'spherical', [rho, phi, z]),
            ]
        for *version_to_check, coords in versions_to_check:
            method = find_transformation_function(*version_to_check)
            try:
                result = method(coords)
            except:
                msg = '_to_'.join(version_to_check) + ' failed'
                self.assertTrue(False, msg=msg)
        pass


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

class TestEnsureArray(unittest.TestCase):
    @attr("fast")
    def test_None_is_unchanged(self):
        self.assertTrue(ensure_array(None) is None)

    @attr("fast")
    def test_xarray_is_unchanged(self):
        xr_array = xr.DataArray([2], dims='a', coords={'a':['b']})
        self.assertTrue(xr_array.equals(ensure_array(xr_array)))

    @attr("fast")
    def test_listlike(self):
        self.assertEqual(ensure_array([1]), np.array([1]))
        self.assertEqual(ensure_array((1)), np.array([1]))
        self.assertEqual(ensure_array(np.array([1])), np.array([1]))

    @attr("fast")
    def test_xarrays_without_coords(self):
        self.assertEqual(ensure_array(xr.DataArray(1)), np.array([1]))
        self.assertEqual(ensure_array(xr.DataArray([1])), np.array([1]))

    @attr("fast")
    def test_zero_d_objects(self):
        self.assertEqual(ensure_array(1), np.array([1]))
        self.assertEqual(ensure_array(np.array(1)), np.array([1]))
        zero_d_xarray = xr.DataArray(2, coords={'a':'b'})
        xr_array = xr.DataArray([2], dims='a', coords={'a':['b']})
        self.assertTrue(xr_array.equals(ensure_array(zero_d_xarray)))


class TestListUtils(unittest.TestCase):
    @attr('fast')
    def test_ensure_listlike(self):
        self.assertEqual(ensure_listlike(None), [])
        self.assertEqual(ensure_listlike(1), [1])
        self.assertEqual(ensure_listlike([1]), [1])

    @attr('fast')
    def test_ensure_scalar(self):
        self.assertEqual(ensure_scalar(1), 1)
        self.assertEqual(ensure_scalar(np.array(1)), 1)
        self.assertEqual(ensure_scalar(np.array([1])), 1)

@attr("fast")
def test_mkdir_p():
    tempdir = tempfile.mkdtemp()
    mkdir_p(os.path.join(tempdir, 'a', 'b'))
    mkdir_p(os.path.join(tempdir, 'a', 'b'))
    shutil.rmtree(tempdir)


class TestDictionaryUtils(unittest.TestCase):
    @attr("fast")
    def test_dict_without(self):
        input_dict = {'a':1, 'b':2, 'c':3, 'd':4}
        output_dict = dict_without(input_dict, ['a','d','e'])
        self.assertEqual(input_dict, {'a':1, 'b':2, 'c':3, 'd':4})
        self.assertEqual(output_dict, {'b':2, 'c':3})

    @attr("fast")
    def test_updated_basic(self):
        input_dict = {'a':1, 'b':2, 'c':3, 'd':4}
        update_dict = {'c':5, 'd':None, 'e':6}
        output_dict = updated(input_dict, update_dict)
        self.assertEqual(input_dict, {'a':1, 'b':2, 'c':3, 'd':4})
        self.assertEqual(output_dict, {'a':1, 'b':2, 'c':5, 'd':4, 'e':6})

    @attr("fast")
    def test_updated_keep_None(self):
        input_dict = {'a':1, 'b':2, 'c':3, 'd':4}
        update_dict = {'c':5, 'd':None, 'e':6}
        output_dict = updated(input_dict, update_dict, False)
        self.assertEqual(input_dict, {'a':1, 'b':2, 'c':3, 'd':4})
        self.assertEqual(output_dict, {'a':1, 'b':2, 'c':5, 'd':None, 'e':6})

    @attr("fast")
    def test_updated_from_kw(self):
        input_dict = {'a':1, 'b':2, 'c':3, 'd':4}
        output_dict = updated(input_dict, b=7, c=None, e=8)
        self.assertEqual(input_dict, {'a':1, 'b':2, 'c':3, 'd':4})
        self.assertEqual(output_dict, {'a':1, 'b':7, 'c':3, 'd':4, 'e':8})

    @attr("fast")
    def test_kw_takes_priority(self):
        input_dict = {'a':1, 'b':2, 'c':3, 'd':4}
        update_dict = {'c':5, 'd':None, 'e':6}
        output_dict = updated(input_dict, update_dict, b=7, e=8)
        self.assertEqual(input_dict, {'a':1, 'b':2, 'c':3, 'd':4})
        self.assertEqual(output_dict, {'a':1, 'b':7, 'c':5, 'd':4, 'e':8})


class TestRepeatSingDims(unittest.TestCase):
    # these tests compare dictionaries containing numpy arrays
    # using np.testing.assert_equal to avoid errors.
    @attr("fast")
    def test_all_keys(self):
        input_dict = {'x':[0], 'y':[1], 'z':[0,1,2]}
        output_dict = {'x':np.array([0, 0, 0]), 'y':np.array([1, 1, 1]),
                      'z':[0, 1, 2]}
        np.testing.assert_equal(repeat_sing_dims(input_dict), output_dict)

    @attr("fast")
    def test_input_isnt_modified(self):
        input_dict = {'x':[0], 'y':[1], 'z':[0,1,2]}
        repeat_sing_dims(input_dict)
        self.assertEqual(input_dict, {'x':[0], 'y':[1], 'z':[0,1,2]})

    @attr("fast")
    def test_repeat_some_keys(self):
        input_dict = {'x':[0], 'y':[1], 'z':[0,1,2]}
        output_dict ={'x':np.array([0,0,0]), 'y':[1], 'z':[0, 1, 2]}
        repeated = repeat_sing_dims(input_dict, ['x', 'z'])
        np.testing.assert_equal(repeated, output_dict)

    @attr("fast")
    def test_nothing_to_repeat(self):
        input_dict = {'x':[0], 'y':[1], 'z':[0,1,2]}
        repeated = repeat_sing_dims(input_dict, ['x', 'y'])
        self.assertEqual(repeated, input_dict)


class TestChoosePool(unittest.TestCase):
    @attr("fast")
    def test_custom_pool(self):
        custom_pool = DummyPool(17)
        chosen_pool = choose_pool(custom_pool)
        self.assertTrue(choose_pool(custom_pool) is custom_pool)

    @attr("fast")
    def test_multiprocessing_pool(self):
        mp_pool = mp.pool.Pool(5)
        self.assertTrue(choose_pool(mp_pool) is mp_pool)

    @attr("fast")
    @unittest.skipIf(not _has_schwimmbad, "schwimmbad not installed")
    def test_nonepool(self):
        none_pool = choose_pool(None)
        self.assertFalse(isinstance(none_pool, (pool.BasePool, mp.pool.Pool)))
        self.assertEqual(list(none_pool.map(len, [[0,1,2],'asdf'])), [3, 4])
        self.assertTrue(hasattr(none_pool, "close"))

    @attr("fast")
    @unittest.skipIf(not _has_schwimmbad, "schwimmbad not installed")
    def test_counting_all_cores(self):
        all_pool = choose_pool('all')
        self.assertTrue(isinstance(all_pool, MultiPool))
        self.assertEqual(all_pool._processes, mp.cpu_count())

    @attr("fast")
    @unittest.skipIf(not _has_schwimmbad, "schwimmbad not installed")
    def test_schwimmbad_multipool(self):
        multi_pool = choose_pool(5)
        self.assertTrue(isinstance(multi_pool, MultiPool))
        self.assertEqual(multi_pool._processes, 5)

    @attr("fast")
    @unittest.skipIf(not _has_schwimmbad, "schwimmbad not installed")
    def test_MPI(self):
        if MPIPool.enabled():
            # mpi should work
            mpi_pool = choose_pool('mpi')
            self.assertTrue(isinstance(mpi_pool, MPIPool))
        elif find_spec('mpi4py') is None:
            # mpi4py not installed
            self.assertRaises(ImportError, choose_pool, 'mpi')
        else:
            # mpi4py installed but only one process available
            self.assertRaises(ValueError, choose_pool, 'mpi')

    @attr("fast")
    @unittest.skipIf(not _has_schwimmbad, "schwimmbad not installed")
    def test_auto(self):
        auto_pool = choose_pool('auto')
        self.assertTrue(isinstance(auto_pool, (pool.BasePool, mp.pool.Pool)))


if __name__ == '__main__':
    unittest.main()
