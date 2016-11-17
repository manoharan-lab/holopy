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

import yaml
import tempfile
import os
import shutil
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from nose.plugins.attrib import attr

from .. import load, save, load_image, save_image
from ..process import normalize
from ..metadata import get_spacing
from ..holopy_object import Serializable
from .common import (assert_obj_close, assert_read_matches_write, get_example_data)

@attr('fast')
def test_hologram_io():
    holo = normalize(get_example_data('image0001'))
    assert_read_matches_write(holo)

def test_image_io():
    holo = get_example_data('image0001')

    t = tempfile.mkdtemp()

    filename = os.path.join(t, 'image0001.tif')
    save_image(filename, holo, scaling=None)
    l = load(filename)
    assert_obj_close(l, holo)

    # check that it defaults to saving as tif
    filename = os.path.join(t, 'image0002')
    save_image(filename, holo, scaling=None)
    l = load_image(filename+'.tif', name=holo.name, medium_index=holo.medium_index, spacing=get_spacing(holo), illum_wavelen=holo.illum_wavelen, illum_polarization=holo.illum_polarization, normals=holo.normals)
    assert_obj_close(l, holo)

    ##check saving/loading non-tif
    filename = os.path.join(t, 'image0001.bmp')
    save_image(filename, holo, scaling=None)
    # For now we don't support writing metadata to image formats other
    # than tiff, so we have to specify the metadata here
    l=load_image(filename, name=holo.name, medium_index=holo.medium_index, spacing=get_spacing(holo), illum_wavelen=holo.illum_wavelen, illum_polarization=holo.illum_polarization, normals=holo.normals)
    assert_obj_close(l, holo)    

    #check specify scaling
    filename = os.path.join(t, 'image0001.tif')
    save_image(filename, holo, scaling=(0,255))
    l=load_image(filename, name=holo.name, medium_index=holo.medium_index, spacing=get_spacing(holo), illum_wavelen=holo.illum_wavelen, illum_polarization=holo.illum_polarization, normals=holo.normals)
    assert_obj_close(l, holo)

    #check auto scaling
    filename = os.path.join(t, 'image0001.tif')
    save_image(filename, holo, depth='float')
    l=load_image(filename, name=holo.name)
    # skip checking full DataArray attrs because it is akward to keep them through arithmatic. Ideally we would figure out a way to preserve them and switch back to testing fully
    assert_allclose(l, (holo-holo.min())/(holo.max()-holo.min()))    

    # check saving 16 bit
    filename = os.path.join(t, 'image0003')
    save_image(filename, holo, scaling=None, depth=16)
    l = load_image(filename+'.tif', name=holo.name, medium_index=holo.medium_index, spacing=get_spacing(holo), illum_wavelen=holo.illum_wavelen, illum_polarization=holo.illum_polarization, normals=holo.normals)
    assert_obj_close(l, holo)

    # test that yaml save works corretly with a string instead of a file
    filename = os.path.join(t, 'image0001')
    save(filename, holo)
    loaded = load(filename)
    assert_obj_close(loaded, holo)
    shutil.rmtree(t)

# test a number of little prettying up of yaml output that we do for
# numpy types
def test_yaml_output():
    # test that numpy types get cleaned up into python types for clean printing
    a = np.ones(10, 'int')
    assert_equal(yaml.dump(a.std()), '0.0\n...\n')

    assert_equal(yaml.dump(np.dtype('float')),"!dtype 'float64'\n")
    assert_equal(yaml.load(yaml.dump(np.dtype('float'))), np.dtype('float64'))

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
