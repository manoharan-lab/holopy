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

import warnings
import unittest
import tempfile
from collections import OrderedDict

import numpy as np
from numpy.testing import assert_allclose, assert_raises, assert_equal

from holopy.core.metadata import data_grid, clean_concat, illumination as ILLUM
from holopy.core.io.vis import display_image, show
from holopy.core.io.io import get_example_data
from holopy.core.tests.common import assert_obj_close
from holopy.core.errors import BadImage


# Creating some d-dimensional arrays for testing visualization:
ARRAY_2D = np.arange(20).reshape(5, 4)
ARRAY_3D = np.arange(60).reshape(3, 5, 4)
ARRAY_4D = np.transpose(
    [ARRAY_3D, ARRAY_3D + 0.5, 0 * ARRAY_3D],
    axes=(1, 2, 3, 0))
ARRAY_5D = np.reshape(ARRAY_4D, ARRAY_4D.shape + (1,))


with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import matplotlib.pyplot as plt
plt.ioff()


def convert_ndarray_to_xarray(array, extra_dims=None):
    # FIXME extra_dims needs to be an OrderedDict, since the creation of
    # an xarray assumes that iteration over extra_dims occurs in the
    # insertion order.
    # However FIXME passing ``extra_dims`` as an OrderedDict does not
    # let the tests pass, as holopy.core.metadata.data_grid and
    # holopy.core.metadata.make_coords both assume that dicts iterate
    # in a fixed order.
    if array.ndim > 2:
        z = range(len(array))
    else:
        z = 0
    array = data_grid(array, spacing=1, z=z, extra_dims=extra_dims)
    array.attrs['_image_scaling'] = None
    return array


class TestDisplayImage(unittest.TestCase):
    def test_basic_case_with_real_values(self):
        xarray_real = convert_ndarray_to_xarray(ARRAY_3D)
        displayed = display_image(xarray_real, scaling=None)
        displayed_transposed = display_image(
            xarray_real.transpose(), scaling=None)
        assert_obj_close(displayed, xarray_real)
        assert_obj_close(displayed_transposed, xarray_real)

    def test_basic_case_with_complex_values(self):
        xarray_real = convert_ndarray_to_xarray(ARRAY_3D)
        xarray_complex = xarray_real + 0j
        xarray_complex[0, 0, :] = (
            xarray_complex[0, 0, :] / np.sqrt(2) * (1 + 1j))  # /= ??
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            displayed = display_image(xarray_complex, scaling=None)
            assert_obj_close(displayed, xarray_real)

    def test_custom_dimension_names(self):
        xarray_real = convert_ndarray_to_xarray(ARRAY_3D)
        dims = xarray_real.assign_coords(
            dim1=xarray_real['x'],
            dim2=xarray_real['y'],
            dim3=xarray_real['z'])
        dims = dims.swap_dims({'x': 'dim1', 'y': 'dim2', 'z': 'dim3'})
        dims = display_image(dims, vert_axis='dim1', horiz_axis='dim2',
                             depth_axis='dim3', scaling=None)
        is_ok = np.allclose(dims.values, xarray_real.values)
        self.assertTrue(is_ok)

    def test_custom_dimension_names_extra_dims(self):
        xarray_real = convert_ndarray_to_xarray(ARRAY_3D)
        extra_dims = OrderedDict([["t", [0, 1, 2]], [ILLUM, [0, 1, 2]]])
        xarray_5d = convert_ndarray_to_xarray(
            ARRAY_5D.transpose([4, 1, 2, 0, 3]),
            extra_dims=extra_dims)
        displayed = display_image(xarray_5d, depth_axis='t', scaling=None)

        xarray_4d = convert_ndarray_to_xarray(
            ARRAY_4D, extra_dims={ILLUM: [0, 1, 2]})

        # There is no arithemtic operations on these, so the numbers
        # should be exactly the same and not just close:
        is_ok = np.all(displayed.values == xarray_4d.values)
        self.assertTrue(is_ok)

    def test_interpet_axes_for_numpy_arrays(self):
        xr2 = convert_ndarray_to_xarray(ARRAY_2D)
        xr3 = convert_ndarray_to_xarray(ARRAY_3D)
        displayed_2d = display_image(ARRAY_2D, scaling=None)
        displayed_3d = display_image(ARRAY_3D, scaling=None)
        displayed_transposed = display_image(
            np.transpose(ARRAY_3D, [1, 0, 2]),
            scaling=None)
        assert_obj_close(displayed_2d, xr2)
        assert_obj_close(display_image(displayed_3d, scaling=None), xr3)
        assert_obj_close(displayed_transposed, xr3)

    def test_specify_axes_for_numpy_arrays(self):
        transposed = np.transpose(ARRAY_3D, [1, 0, 2])
        displayed_transposed = display_image(transposed, scaling=None)
        xr_transposed = convert_ndarray_to_xarray(transposed)
        assert_obj_close(
            display_image(ARRAY_3D, depth_axis=1, scaling=None), xr_transposed)
        assert_obj_close(
            display_image(ARRAY_3D, vert_axis=0, horiz_axis=2, scaling=None),
            xr_transposed)

    def test_raises_error_if_image_is_1d(self):
        self.assertRaises(BadImage, display_image, ARRAY_2D[0])

    def test_raises_error_if_image_is_4d(self):
        self.assertRaises(BadImage, display_image, ARRAY_4D)

    def test_excess_dims(self):
        # FIXME what are these tests testing for? more specific names
        # would be great.
        xr4 = convert_ndarray_to_xarray(ARRAY_4D, extra_dims={'t': [0, 1, 2]})
        assert_raises(BadImage, display_image, xr4)

        extra_dims = OrderedDict([[ILLUM, [0, 1, 2]], ["t", [0]]])
        xr5 = convert_ndarray_to_xarray(ARRAY_5D, extra_dims=extra_dims)
        assert_raises(BadImage, display_image, xr5)

        col1 = convert_ndarray_to_xarray(
            ARRAY_4D, extra_dims={ILLUM: [0, 1, 2]})
        col2 = convert_ndarray_to_xarray(
            ARRAY_4D, extra_dims={ILLUM: [3, 4, 5]})
        xr6cols = clean_concat([col1, col2], dim=ILLUM)
        assert_raises(BadImage, display_image, xr6cols)

    def test_scaling_exceeds_intensity_bounds(self):
        scale = (-5, 100)
        xr3 = (convert_ndarray_to_xarray(ARRAY_3D) + 5) / 105
        displayed = display_image(ARRAY_3D, scaling=scale)
        assert_allclose(displayed.values, xr3.values)
        assert_equal(displayed.attrs['_image_scaling'], scale)

    def test_scaling_constricts_intensity_bounds(self):
        scale = (-5, 100)
        wide3 = ARRAY_3D.copy()
        wide3[0, 0, 0] = -5
        wide3[-1, -1, -1] = 100
        xr3 = convert_ndarray_to_xarray(ARRAY_3D)/59
        assert_equal(display_image(wide3).attrs['_image_scaling'], scale)
        assert_obj_close(display_image(wide3, (0, 59)).values, xr3.values)

    def test_flat_colour_dimension(self):
        xr3 = convert_ndarray_to_xarray(
            ARRAY_4D[:, :, :, 0:1], extra_dims={ILLUM: [0]})
        displayed_xr = display_image(xr3)
        displayed_np = display_image(ARRAY_3D)
        assert_obj_close(displayed_xr, displayed_np)

    def test_colour_name_formats(self):
        base = convert_ndarray_to_xarray(
            ARRAY_4D, extra_dims={ILLUM: ['red', 'green', 'blue']})
        cols = [['Red', 'Green', 'Blue'],
                ['r', 'g', 'b'],
                [0, 1, 2],
                ['a', 's', 'd']]
        for collist in cols:
            xr4 = convert_ndarray_to_xarray(
                ARRAY_4D, extra_dims={ILLUM: collist})
            assert_obj_close(display_image(xr4, scaling=None), base)

    def test_colours_in_wrong_order(self):
        base = convert_ndarray_to_xarray(
            ARRAY_4D, extra_dims={ILLUM: ['red', 'green', 'blue']})
        xr4 = convert_ndarray_to_xarray(
            ARRAY_4D[:, :, :, [0, 2, 1]],
            extra_dims={ILLUM: ['red', 'blue', 'green']})
        assert_allclose(display_image(xr4, scaling=None).values, base.values)

    def test_missing_colours(self):
        base = convert_ndarray_to_xarray(
            ARRAY_4D, extra_dims={ILLUM: ['red', 'green', 'blue']})
        slices = [[0, 2, 1], [1, 0], [0, 1], [0, 1]]
        possible_valid_colors = [
            ['red', 'blue', 'green'],
            ['green', 'red'],
            [0, 1],
            ['x-pol', 'y-pol']]
        dummy_channel = [None, 2, -1, -1]
        for i, c, d in zip(slices, possible_valid_colors, dummy_channel):
            xr4 = convert_ndarray_to_xarray(
                ARRAY_4D[:, :, :, i], extra_dims={ILLUM: c})
            xr4 = display_image(xr4, scaling=None)
            if d is not None:
                assert_equal(xr4.attrs['_dummy_channel'], d)
                del xr4.attrs['_dummy_channel']
            assert_obj_close(xr4, base)


class ShowTest(unittest.TestCase):
    def test_show(self):
        d = get_example_data('image0001')
        try:
            show(d)
        except RuntimeError:
            # this occurs on travis since there is no display
            raise SkipTest()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', (DeprecationWarning, UserWarning))
            plt.savefig(tempfile.TemporaryFile(suffix='.pdf'))


if __name__ == '__main__':
    unittest.main()
