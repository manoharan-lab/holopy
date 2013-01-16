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
from __future__ import division
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
from ..marray import (ImageSchema, VectorGrid, VectorGridSchema, Image,
                      zeros_like, subimage, resize, Volume, Schema)
from .common import assert_obj_close, get_example_data
from ..errors import UnspecifiedPosition
from .. import Angles

def test_VectorGrid():
    schema = ImageSchema(shape = (100,100), spacing = .1)
    vs = VectorGridSchema.from_schema(schema)
    vd = zeros_like(vs)
    assert_equal(vd.shape, (100,100,3))

    vd2 = VectorGrid(np.zeros((10, 10, 3)))
    assert_equal(vd2.components, ('x', 'y', 'z'))

    holo = get_example_data('image0001.yaml')
    vh = zeros_like(holo)
    assert_equal(vh.shape, holo.shape)
    assert_equal(vh.optics, holo.optics)

    assert_raises(UnspecifiedPosition, holo.positions_theta_phi)


def test_positions_in_spherical():
    schema = ImageSchema(shape = (2,2), spacing = 1)
    spherical = schema.positions_r_theta_phi((0,0,1))
    assert_allclose(spherical, np.array([[ 1. , 0. , 0.],
       [ 1.41421356,  0.78539816,  1.57079633],
       [ 1.41421356,  0.78539816,  0.        ],
       [ 1.73205081,  0.95531662,  0.78539816]]))

def test_from1d():
    schema = ImageSchema(shape = (2,2), spacing = 1)
    im = VectorGrid([[1, 0, 0],
                       [0, 1, 2],
                       [1, 2, 3],
                       [5, 8, 9]])
    assembled = VectorGrid([[[1, 0, 0], [0, 1, 2]], [[1, 2, 3], [5, 8, 9]]])

    vf = VectorGridSchema.from_schema(schema).interpret_1d(im)
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

def test_subimage():
    i = np.zeros((10,10))
    s = subimage(i, (5,5), 2)
    assert s.shape == (2,2)

    assert_raises(IndexError, subimage, i, (2,2), 10)

def test_resize():
    i = Image(np.zeros((100, 100)), .1)
    r = resize(i, (5, 5), (8, 8))
    assert_equal(r.center, (5, 5, 0))
    assert_equal(r.extent, (8, 8))

    v = Volume(np.zeros((10, 10, 10)), .1)
    r = resize(v, extent =  (.8, .8, .8))
    assert_equal(r.center, (.5, .5, .5))
    assert_equal(r.extent, (.8, .8, .8))

    r = resize(v, spacing = (.2, .2, .2))
    assert_equal(r.spacing, (.2, .2, .2))
    assert_equal(r.shape, (5, 5, 5))

def test_positions_theta_phi():
    theta = np.linspace(0, np.pi)
    phi = np.linspace(0, 2*np.pi)
    a = Angles(theta, phi)
    s = Schema((50, 2), a)
    ptp = s.positions_theta_phi()
    assert_equal(ptp.shape, (2500, 2))
    assert_equal(ptp[-1], (np.pi, np.pi*2))
    assert_allclose(ptp[182], np.array([ 0.19234241,  4.10330469]))
