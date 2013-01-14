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

from .common import (assert_obj_close, assert_read_matches_write,
                     get_example_data)
from .. import Optics, Marray, load, save
from .. process import normalize
import tempfile
import os
import shutil
from nose.plugins.attrib import attr
from numpy.testing import assert_raises
import numpy as np
from ..errors import LoadError
from ..io import save_image, load_image


@attr('fast')
def test_hologram_io():
    holo = normalize(get_example_data('image0001.yaml'))

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

def test_tif_io():
    holo = get_example_data('image0001.yaml')
    t = tempfile.mkdtemp()

    filename = os.path.join(t, 'image0001.tif')
    save(filename, holo)
    l = load(filename)
    assert_obj_close(l, holo)

    # check that it defaults to saving as tif
    filename = os.path.join(t, 'image0002')
    save_image(filename, holo)
    l = load(filename+'.tif')
    assert_obj_close(l, holo)

    shutil.rmtree(t)

def test_non_tiff():
    # test loading a few other image formats.  We have some in the docs
    # director, so just use them
    import holopy
    root = os.path.split(os.path.split(holopy.__file__)[0])[0]
    doc_images = os.path.join(root, 'docs', 'source', 'images')

    load(os.path.join(doc_images, 'image_5Particle_Hologram.jpg'))
    load(os.path.join(doc_images, 'ReconVolume_mlab_5Particle_Hologram.png'))

    assert_raises(LoadError, load_image, os.path.join(doc_images,
                                                      'image_5Particle_Hologram.jpg'), 4)
