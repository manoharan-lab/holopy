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
from numpy.testing import assert_allclose, assert_equal, assert_raises
from ..tools import center_find, subimage, resize, fft, ifft, math
from ..tools import _ensure_array, coord_grid, ensure_listlike, mkdir_p, ensure_3d, squeeze
from .. import Image, Volume, Optics
from .common import get_example_data
from scipy import fftpack
from nose.plugins.attrib import attr
import tempfile
import os
import shutil

#Test centerfinder
gold_location = np.array([ 48.5729142,  50.23217416])

def test_FoundLocation():
    holo = get_example_data('image0001.yaml')
    location = center_find(holo, threshold=.25)
    assert_allclose(location, gold_location)

#Test img_proc
def test_subimage():
    i = np.zeros((10,10))
    s = subimage(i, (5,5), 2)
    assert s.shape == (2,2)

    i2 = Image(i, 1)
    s2 = subimage(i2, (5, 5), 2)

    assert_equal(s2.positions, subimage(i2.positions, (5,5), 2))

    assert_raises(IndexError, subimage, i, (2,2), 10)

def test_subimage_floats():
    i = Image(np.zeros((100, 100)), .1)
    s1 = subimage(i, (5.2,5.6), 2)
    s2 = subimage(i, (5,6), 2)
    assert_equal(s1, s2)

def test_resize():
    i = Image(np.zeros((100, 100)), .1)
    r = resize(i, (5, 5), (8, 8))
    assert_equal(r.center, (5, 5, 0))
    assert_equal(r.extent[:2], (8, 8))

    v = Volume(np.zeros((10, 10, 10)), .1)
    r = resize(v, extent =  (.8, .8, .8))
    assert_equal(r.center, (.5, .5, .5))
    assert_equal(r.extent, (.8, .8, .8))

    r = resize(v, spacing = (.2, .2, .2))
    assert_equal(r.spacing, (.2, .2, .2))
    assert_equal(r.shape, (5, 5, 5))

#Test fourier
@attr('fast')
def test_fft_1d_no_shift():
    a = np.array([ 0.03939436,  0.69091932,  0.10291701,  0.92518768,  0.99788634,
                   0.70251321,  0.3351499 ,  0.69498738,  0.67824007,  0.374988  ,
                   0.46005945,  0.09685981,  0.29764051,  0.2803798 ,  0.26017554,
                   0.34394677,  0.45096598,  0.17818282,  0.61982928,
                   0.60430196])

    ft = np.array([ 9.13452519+0.j        ,  0.49661245-1.92237764j,
       -0.88865249+0.37262022j, -0.67073165+0.19970474j,
       -0.33229828+1.35390738j,  0.68599608+0.43830045j,
       -0.15385878-0.59887569j, -1.39518843-0.79065853j,
       -0.37017984-0.77974293j, -1.22001389-0.89970879j,
       -0.65000831+0.j        , -1.22001389+0.89970879j,
       -0.37017984+0.77974293j, -1.39518843+0.79065853j,
       -0.15385878+0.59887569j,  0.68599608-0.43830045j,
       -0.33229828-1.35390738j, -0.67073165-0.19970474j,
       -0.88865249-0.37262022j,  0.49661245+1.92237764j])

    assert_allclose(fft(a, shift=False), ft)

@attr('fast')
def test_ifft_1d_no_shift():
    a = np.array([ 0.03939436,  0.69091932,  0.10291701,  0.92518768,  0.99788634,
                   0.70251321,  0.3351499 ,  0.69498738,  0.67824007,  0.374988  ,
                   0.46005945,  0.09685981,  0.29764051,  0.2803798 ,  0.26017554,
                   0.34394677,  0.45096598,  0.17818282,  0.61982928,
                   0.60430196])

    ift = np.array([ 0.45672626+0.j        ,  0.02483062+0.09611888j,
       -0.04443262-0.01863101j, -0.03353658-0.00998524j,
       -0.01661491-0.06769537j,  0.03429980-0.02191502j,
       -0.00769294+0.02994378j, -0.06975942+0.03953293j,
       -0.01850899+0.03898715j, -0.06100069+0.04498544j,
       -0.03250042+0.j        , -0.06100069-0.04498544j,
       -0.01850899-0.03898715j, -0.06975942-0.03953293j,
       -0.00769294-0.02994378j,  0.03429980+0.02191502j,
       -0.01661491+0.06769537j, -0.03353658+0.00998524j,
       -0.04443262+0.01863101j,  0.02483062-0.09611888j])

    assert_allclose(ifft(a, shift=False), fftpack.ifft(a))

#Test math
def test_rotate_single_point():
    points = np.array([1.,1.,1.])
    assert_allclose(math.rotate_points(points, np.pi, np.pi, np.pi),
                    np.array([-1.,  1., -1.]), 1e-5)
    
def test_rotation_matrix_degrees():
    assert_allclose(math.rotation_matrix(180., 180., 180., radians = False), 
                    math.rotation_matrix(np.pi, np.pi, np.pi))

@attr('fast')
def test_ensure_array():
    assert_equal(_ensure_array(1.0), np.array([1.0]))
    assert_equal(_ensure_array([1.0]), np.array([1.0]))
    assert_equal(_ensure_array(np.array([1.0])), np.array([1.0]))

@attr('fast')
def test_coord_grid():
    assert_equal(coord_grid(([0, 5], [0, 5], [0, 5])).shape, (5, 5, 5, 3))
    assert_equal(coord_grid((5, 5, 5)).shape, (5, 5, 5, 3))
    assert_equal(coord_grid(([0, 1], [0, 1], [0, 1]), .2).shape, (5, 5, 5, 3))
    assert_equal(coord_grid(([0, 1], [0, 1], [0, 1]), .5),
                 np.array([[[[ 0. ,  0. ,  0. ],
                             [ 0. ,  0. ,  0.5]],
                            [[ 0. ,  0.5,  0. ],
                             [ 0. ,  0.5,  0.5]]],
                           [[[ 0.5,  0. ,  0. ],
                             [ 0.5,  0. ,  0.5]],
                            [[ 0.5,  0.5,  0. ],
                             [ 0.5,  0.5,  0.5]]]]))

#test helpers
def test_ensure_listlike():
    assert ensure_listlike(None) == []

def test_ensure3d():
    assert_raises(Exception, ensure_3d, [1])

def test_mkdir_p():
    tempdir = tempfile.mkdtemp()
    mkdir_p(os.path.join(tempdir, 'a', 'b'))
    mkdir_p(os.path.join(tempdir, 'a', 'b'))
    shutil.rmtree(tempdir)

def test_squeeze():
    v = Volume(np.ones((10, 1, 10)), spacing = (1, 2, 3),
                       optics = Optics(.66, 1, (1, 0)))
    s = squeeze(v)
    assert_equal(s.shape, (10, 10))
    assert_equal(s.optics, v.optics)
    assert_equal(s.spacing, (1, 3))
