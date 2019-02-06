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

import numpy as np
from numpy.testing import assert_allclose, assert_raises, assert_equal

from holopy.core.metadata import data_grid, clean_concat, illumination as illum
from holopy.core.io.vis import display_image, show
from holopy.core.io.io import get_example_data
from holopy.core.tests.common import assert_obj_close
from holopy.core.errors import BadImage

d2 = np.array([range(i, i+4) for i in range(0,19,4)])
d3 = np.array([[range(i, i+4) for i in range(j, j+19,4)] for j in [0,20,40]])
d4 = np.array([[[[c, c+0.5, 0] for c in range(i, i+4)]
                              for i in range(j, j+19,4)] for j in [0,20,40]])
d5 = np.array([[[[[c], [c+0.5], [0]] for c in range(i, i+4)] 
                              for i in range(j, j+19,4)] for j in [0,20,40]])

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import matplotlib.pyplot as plt
plt.ioff()

def ndarray2xr(array, extra_dims=None):
    if array.ndim > 2:
        z = range(len(array))
    else:
        z = 0
    array = data_grid(array, spacing=1, z=z, extra_dims=extra_dims)
    array.attrs['_image_scaling'] = None
    return array

class TestDisplayImage(unittest.TestCase):
    def test_basics(self):
        # test simplest cases
        basic = ndarray2xr(d3)
        assert_obj_close(display_image(basic, scaling=None), basic)
        assert_obj_close(display_image(basic.transpose(), scaling=None), basic)

        # test complex values
        cplx = basic.copy()+0j
        cplx[0,0,:] = cplx[0,0,:] / np.sqrt(2) * (1 + 1j)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert_obj_close(display_image(cplx, scaling=None), basic)

        # test custom dim names
        dims = basic.assign_coords(dim1=basic['x'], dim2=basic['y'], dim3=basic['z'])
        dims = dims.swap_dims({'x':'dim1', 'y':'dim2', 'z':'dim3'})
        dims = display_image(dims, vert_axis = 'dim1', horiz_axis = 'dim2', 
                                        depth_axis = 'dim3', scaling=None)
        assert_allclose(dims.values, basic.values)
        t5 = d5.transpose([4, 1, 2, 0, 3])
        t5 = ndarray2xr(t5, extra_dims={"t":[0,1,2],illum:[0,1,2]})
        t5 = display_image(t5, depth_axis='t', scaling=None)
        xr4 = ndarray2xr(d4, extra_dims={illum:[0,1,2]})
        assert_obj_close(t5.values, xr4.values)

    def test_np_arrays(self):
        # test interpret axes
        xr2 = ndarray2xr(d2)
        assert_obj_close(display_image(d2, scaling=None), xr2)
        xr3 = ndarray2xr(d3)
        assert_obj_close(display_image(d3, scaling=None), xr3)
        transposed3 = np.transpose(d3, [1, 0, 2])
        assert_obj_close(display_image(transposed3, scaling=None), xr3)

        # test specify axes
        xr3trans = ndarray2xr(transposed3)
        assert_obj_close(display_image(d3, depth_axis=1, scaling=None), xr3trans)
        assert_obj_close(display_image(d3, vert_axis=0, horiz_axis=2, scaling=None), xr3trans)

    def test_excess_dims(self):
        assert_raises(BadImage, display_image, d2[0])
        assert_raises(BadImage, display_image, d4)
        xr4 = ndarray2xr(d4, extra_dims={'t':[0,1,2]})
        assert_raises(BadImage, display_image, xr4)
        xr5 = ndarray2xr(np.array(d5), extra_dims = {illum:[0,1,2], 't':[0]})
        assert_raises(BadImage, display_image, xr5)
        col1 = ndarray2xr(d4, extra_dims={illum:[0,1,2]})
        col2 = ndarray2xr(d4, extra_dims={illum:[3,4,5]})
        xr6cols = clean_concat([col1, col2], dim=illum)
        assert_raises(BadImage, display_image, xr6cols)

    def test_scaling(self):
        # test scaling exceeds intensity bounds
        my_scale = (-5, 100)
        xr3 = (ndarray2xr(d3)+5)/105
        disp = display_image(d3, scaling=my_scale)
        assert_allclose(disp.values, xr3.values)
        assert_equal(disp.attrs['_image_scaling'], my_scale)

        # test scaling constricts intensity bounds
        wide3 = d3.copy()
        wide3[0, 0, 0] = -5
        wide3[-1, -1, -1] = 100
        xr3 = ndarray2xr(d3)/59
        assert_equal(display_image(wide3).attrs['_image_scaling'], my_scale)
        assert_obj_close(display_image(wide3, (0, 59)).values, xr3.values)

    def test_colours(self):
        # test flat colour dim
        xr3 = ndarray2xr(d4[:,:,:,0:1], extra_dims={illum:[0]})
        assert_obj_close(display_image(xr3), display_image(d3))

        # test colour name formats
        base = ndarray2xr(d4, extra_dims={illum:['red','green','blue']})
        cols = [['Red','Green','Blue'],['r','g','b'], [0,1,2], ['a', 's', 'd']]
        for collist in cols:
            xr4 = ndarray2xr(d4, extra_dims={illum:collist})
            assert_obj_close(display_image(xr4, scaling=None), base)

        # test colours in wrong order
        xr4 = ndarray2xr(d4[:,:,:,[0,2,1]], 
                        extra_dims={illum:['red','blue','green']})
        assert_allclose(display_image(xr4, scaling=None).values, base.values)

        # test missing colours
        slices = [[0,2,1],[1,0],[0,1],[0,1]]
        cols = [['red','blue','green'],['green','red'],[0,1],['x-pol','y-pol']]
        dummy_channel = [None, 2, -1, -1]
        for i, c, d in zip(slices, cols, dummy_channel):
            xr4=ndarray2xr(d4[:,:,:,i], extra_dims={illum:c})
            xr4 = display_image(xr4, scaling=None)
            if d is not None:
                assert_equal(xr4.attrs['_dummy_channel'], d)
                del xr4.attrs['_dummy_channel']
            assert_obj_close(xr4, base)

def test_show():
    d = get_example_data('image0001')
    try:
        show(d)
    except RuntimeError:
        # this occurs on travis since there is no display
        raise SkipTest()    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', (DeprecationWarning, UserWarning))
        plt.savefig(tempfile.TemporaryFile(suffix='.pdf'))
