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
import unittest

import numpy as np
import xarray as xr
from numpy.testing import assert_allclose
from nose.plugins.attrib import attr

from holopy.core.process import center_find, subimage, fft, ifft
from holopy.core.metadata import data_grid, detector_grid
from holopy.core.tests.common import get_example_data, assert_obj_close

#Test centerfinder
gold_location = np.array([ 48.5729142,  50.23217416])

@attr("medium")
def test_FoundLocation():
    holo = get_example_data('image0001')
    location = center_find(holo, threshold=.25)
    assert_allclose(location, gold_location, atol=0.01)


#Test img_proc
@attr("fast")
def test_subimage():
    i = detector_grid(shape=(10, 10), spacing=1)
    s = subimage(i, (5,5), 2)
    assert s.shape == (1,2,2)

    i2 = data_grid(i, 1)
    s2 = subimage(i2, (5, 5), 2)


@attr("fast")
def test_subimage_floats():
    i = data_grid(np.zeros((100, 100)), .1)
    s1 = subimage(i, (5.2,5.6), 2)
    s2 = subimage(i, (5,6), 2)
    assert_obj_close(s1, s2)


class TestFourier(unittest.TestCase):
    @attr('fast')
    def test_fft_1d_no_shift(self):
        a = np.array([
            0.03939436, 0.69091932, 0.10291701, 0.92518768, 0.99788634,
            0.70251321, 0.3351499 , 0.69498738, 0.67824007, 0.374988  ,
            0.46005945, 0.09685981, 0.29764051, 0.2803798 , 0.26017554,
            0.34394677, 0.45096598, 0.17818282, 0.61982928,
            0.60430196])

        ft = np.array([
            +9.13452519+0.j        ,  0.49661245-1.92237764j,
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
    def test_ifft_1d_no_shift(self):
        a = np.array([
            0.03939436, 0.69091932, 0.10291701, 0.92518768, 0.99788634,
            0.70251321, 0.3351499 , 0.69498738, 0.67824007, 0.374988  ,
            0.46005945, 0.09685981, 0.29764051, 0.2803798 , 0.26017554,
            0.34394677, 0.45096598, 0.17818282, 0.61982928,
            0.60430196])

        ift = np.array([
            +0.45672626+0.j        ,  0.02483062+0.09611888j,
            -0.04443262-0.01863101j, -0.03353658-0.00998524j,
            -0.01661491-0.06769537j,  0.03429980-0.02191502j,
            -0.00769294+0.02994378j, -0.06975942+0.03953293j,
            -0.01850899+0.03898715j, -0.06100069+0.04498544j,
            -0.03250042+0.j        , -0.06100069-0.04498544j,
            -0.01850899-0.03898715j, -0.06975942-0.03953293j,
            -0.00769294-0.02994378j,  0.03429980+0.02191502j,
            -0.01661491+0.06769537j, -0.03353658+0.00998524j,
            -0.04443262+0.01863101j,  0.02483062-0.09611888j])

        assert_allclose(ifft(a, shift=False), np.fft.ifft(a))

    @attr('fast')
    def test_fft(self):
        holo = get_example_data('image0001')
        assert_obj_close(holo, ifft(fft(holo)))

    @attr("fast")
    def test_fft_of_xarray_returns_xarray(self):
        xarray = get_example_data('image0001')
        after_fft = fft(xarray)
        self.assertIsInstance(after_fft, xr.DataArray)

    @attr("fast")
    def test_ifft_of_xarray_returns_xarray(self):
        xarray = get_example_data('image0001')
        forward = fft(xarray)
        backward = ifft(forward)
        self.assertIsInstance(backward, xr.DataArray)

    @attr("fast")
    def test_fft_ifft_2d_are_inverses(self):
        xarray = get_example_data('image0001')
        forward = fft(xarray)
        backward = ifft(forward)
        data_is_same = np.allclose(
            xarray.values,
            backward.values,
            atol=1e-13, rtol=1e-13)
        self.assertTrue(data_is_same)


if __name__ == '__main__':
    unittest.main()
