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

from holopy.core.tests.common import (assert_obj_close,
                                      assert_read_matches_write,
                                      get_example_data,
                                      get_example_data_path)
from holopy.core import Optics, Marray, load, save, load_image
from holopy.core.process import normalize
import tempfile
import os
import shutil
import warnings
from nose.plugins.attrib import attr
from numpy.testing import assert_raises, assert_equal
import numpy as np
from holopy.core.io import save_image
from holopy.core.holopy_object import Serializable
import yaml

@attr('fast')
def test_hologram_io():
    holo = normalize(get_example_data('image0001.yaml'))

    assert_read_matches_write(holo)

def test_marray_io():
    d = Marray(np.random.random((10, 10)))
    assert_read_matches_write(d)

def test_image_io():
    holo = get_example_data('image0001.yaml')
    t = tempfile.mkdtemp()

    filename = os.path.join(t, 'image0001.tif')
    save(filename, holo)
    l = load_image(filename)
    assert_obj_close(l, holo)

    # check that it defaults to saving as tif
    filename = os.path.join(t, 'image0002')
    save_image(filename, holo)
    l = load_image(filename+'.tif')
    assert_obj_close(l, holo)

    # check saving 16 bit
    filename = os.path.join(t, 'image0003')
    save_image(filename, holo, scaling=None, depth=16)
    l = load_image(filename+'.tif')
    assert_obj_close(l, holo)

    # test that yaml save works corretly with a string instead of a file
    filename = os.path.join(t, 'image0001.yaml')
    save(filename, holo)
    loaded = load(filename)
    assert_obj_close(loaded, holo)

    shutil.rmtree(t)

def test_non_tiff():
    # test loading a few other image formats.  We have some in the docs
    # director, so just use them
    import holopy
    root = os.path.split(os.path.split(holopy.__file__)[0])[0]
    doc_images = os.path.join(root, 'docs', 'source', 'images')

    load_image(os.path.join(doc_images, 'image_5Particle_Hologram.jpg'))
    load_image(os.path.join(doc_images, 'ReconVolume_mlab_5Particle_Hologram.png'))


# test a number of little prettying up of yaml output that we do for
# numpy types
def test_yaml_output():
    # test that numpy types get cleaned up into python types for clean printing
    a = np.ones(10, 'int')
    assert_equal(yaml.dump(a.std()), '0.0\n...\n')

    assert_equal(yaml.dump(np.dtype('float')),"!dtype 'float64'\n")
    assert_equal(yaml.load(yaml.dump(np.dtype('float'))), np.dtype('float64'))

    assert_equal(yaml.dump(Optics), "!class 'holopy.core.metadata.Optics'\n")
    assert_equal(yaml.load(yaml.dump(Optics)), Optics)

    # this should fail on Windows64 because int and long are both
    # int32
    try:
        assert_equal(yaml.dump(a.max()), '1\n...\n')
    except AssertionError as err:
        if err.args[0] == r"""
Items are not equal:
 ACTUAL: '!!python/object/apply:numpy.core.multiarray.scalar [!dtype \'int32\', "\\x01\\0\\0\\0"]\n'
 DESIRED: '1\n...\n'""":
            raise AssertionError("You're probably running a 32 bit OS.  Writing and reading files with integers migth be buggy on 32 bit OS's, we don't think it will lead to data loss, but we make no guarantees'. If you see this on 64 bit operating systems, please let us know by filing a bug.")
        else:
            raise err

    class S(Serializable):
        def __init__(self, a):
            self.a = a

    assert yaml.dump(S('a')) == '!S {a: a}\n'
