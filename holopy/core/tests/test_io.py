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

from .common import (assert_obj_close, assert_read_matches_write,
                     get_example_data_path)
from .. import Optics, Marray, load
from .. process import normalize
import tempfile
from nose.plugins.attrib import attr
import numpy as np


@attr('fast')
def test_hologram_io():
    o = Optics(wavelen=.66, index=1.33, pixel_scale=.1)

    holo = normalize(load(get_example_data_path('image0001.npy'),
                                        optics=o))

    assert_read_matches_write(holo)

@attr('fast')
def test_load_optics():
    optics_yaml = """wavelen: 785e-9
polarization: [1.0, 0]
divergence: 0
pixel_size: [6.8e-6, 6.8e-6]
pixel_scale: [3.3e-7, 3.3e-7]"""
    t = tempfile.TemporaryFile()
    t.write(optics_yaml)
    t.seek(0)

    o = Optics(**load(t))

    assert_obj_close(o, Optics(wavelen=7.85e-07, polarization=[1.0, 0.0], divergence=0, pixel_size=[6.8e-06, 6.8e-06], pixel_scale=[3.3e-07, 3.3e-07]))

def test_marray_io():
    d = Marray(np.random.random((10, 10)))
    assert_read_matches_write(d)
