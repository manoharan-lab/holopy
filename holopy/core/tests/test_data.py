# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca
# W. Perry, Jerome Fung, and Ryan McGorty
#
# This file is part of Holopy.
#
# Holopy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Holopy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Holopy.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import division
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from ..marray import ImageSchema, Grid, VectorImage, VectorImageSchema, Image
from .common import assert_obj_close


def test_VectorImage():
    schema = ImageSchema(shape = (100,100), spacing = .1)
    vs = VectorImageSchema.from_ImageSchema(schema)
    vd = VectorImage.zeros_like(vs)
    assert_equal(vd.shape, (100,100,3))

    vd2 = VectorImage(np.zeros((10, 10, 3)))
    assert_equal(vd2.components, ('x', 'y', 'z'))


def test_positions_in_spherical():
    schema = ImageSchema(shape = (2,2), spacing = 1)
    spherical = schema.positions_r_theta_phi((0,0,1))
    assert_allclose(spherical, np.array([[ 1.        ,  0.        ,  0.        ],
       [ 1.41421356,  0.78539816,  1.57079633],
       [ 1.41421356,  0.78539816,  0.        ],
       [ 1.73205081,  0.95531662,  0.78539816]]))

def test_from1d():
    schema = ImageSchema(shape = (2,2), spacing = 1)
    im = VectorImage([[1, 0, 0],
                       [0, 1, 2],
                       [1, 2, 3],
                       [5, 8, 9]])
    assembled = VectorImage([[[1, 0, 0], [0, 1, 2]], [[1, 2, 3], [5, 8, 9]]])
    
    vf = VectorImageSchema.from_ImageSchema(schema).interpret_1d(im)
    assert_equal(vf, assembled)

def test_Image():
    i = Image(np.arange(16).reshape((4,4)), spacing = 1)
    assert_equal(i.positions.spacing, 1)
    assert_equal(i.shape, [4, 4])

def test_resample():
    i = Image(np.arange(16).reshape((4,4)), spacing = 1)
    assert_obj_close(i.resample((2, 2)),
                     Image([[  5,   6], [  9,  10]],
                           spacing=np.array([ 2, 2])), context = 'image') 

    assert_obj_close(i, Image(np.arange(16).reshape((4,4)),
                              spacing = 1), context='image') 
