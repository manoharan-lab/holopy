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


import tempfile
import os
import shutil
import warnings
import unittest

import yaml
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from nose.plugins.attrib import attr
from PIL import Image as pilimage
from PIL.TiffImagePlugin import ImageFileDirectory_v2 as ifd2

from holopy.core import load, save, load_image, save_image
from holopy.core.errors import NoMetadata
from holopy.core.process import normalize
from holopy.core.metadata import get_spacing, copy_metadata
from holopy.core.holopy_object import Serializable
from holopy.core.tests.common import (
    assert_obj_close, assert_read_matches_write, get_example_data)


class test_loading_and_saving(unittest.TestCase):
    def setUp(self):
        self.holo = get_example_data('image0001')
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def load_image_with_metadata(self, filename):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded = load_image(filename,
                                name=self.holo.name, medium_index=self.holo.medium_index,
                                spacing=get_spacing(self.holo),
                                illum_wavelen=self.holo.illum_wavelen,
                                illum_polarization=self.holo.illum_polarization,
                                normals=self.holo.normals, noise_sd=self.holo.noise_sd)
        return loaded

    @attr('fast')
    def test_hologram_io(self):
        assert_read_matches_write(normalize(self.holo))

    @attr("fast")
    def test_image_io(self):
        filename = os.path.join(self.tempdir, 'image0001.tif')
        save_image(filename, self.holo, scaling=None)
        l = self.load_image_with_metadata(filename)
        assert_obj_close(l, self.holo)

    @attr("fast")
    def test_default_save_is_tif(self):
        filename = os.path.join(self.tempdir, 'image0002')
        save_image(filename, self.holo, scaling=None)
        l = self.load_image_with_metadata(filename + '.tif')
        assert_obj_close(l, self.holo)

    @attr("fast")
    def test_non_tif_image(self):
        filename = os.path.join(self.tempdir, 'image0001.bmp')
        save_image(filename, self.holo, scaling=None)
        l = self.load_image_with_metadata(filename)
        assert_obj_close(l, self.holo)

    @attr("fast")
    def test_specify_scaling(self):
        filename = os.path.join(self.tempdir, 'image0001.tif')
        save_image(filename, self.holo, scaling=(0, 255))
        l = self.load_image_with_metadata(filename)
        assert_obj_close(l, self.holo)

    @attr("fast")
    def test_auto_scaling(self):
        filename = os.path.join(self.tempdir, 'image0001.tif')
        save_image(filename, self.holo, depth='float')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            l = load_image(filename, name=self.holo.name, spacing=get_spacing(self.holo))
        # skip checking full DataArray attrs because it is akward to keep
        # them through arithmetic. Ideally we would figure out a way to
        # preserve them and switch back to testing fully
        assert_allclose(l, (self.holo-self.holo.min())/(self.holo.max()-self.holo.min()))

    @attr("fast")
    def test_saving_16_bit(self):
        filename = os.path.join(self.tempdir, 'image0003')
        save_image(filename, self.holo, scaling=None, depth=16)
        l = self.load_image_with_metadata(filename + '.tif')
        assert_obj_close(l, self.holo)

    @attr("fast")
    def test_save_load_h5(self):
        filename = os.path.join(self.tempdir, 'image0001')
        save(filename, self.holo)
        loaded = load(filename)
        assert_obj_close(loaded, self.holo)

    @attr("fast")
    def test_load_func_from_save_image_func(self):
        filename = os.path.join(self.tempdir, 'image0006')
        save_image(filename, self.holo, scaling=None)
        loaded = load(filename + '.tif')
        assert_obj_close(loaded, self.holo)

    @attr("fast")
    def test_load_func_from_save_image_func_with_scaling(self):
        filename = os.path.join(self.tempdir, 'image0006')
        save_image(filename, self.holo, scaling='auto')
        loaded = load(filename + '.tif')
        assert_obj_close(np.around(loaded), self.holo)

    @attr("fast")
    def test_ignoring_metadata_warning(self):
        filename = os.path.join(self.tempdir, 'image0005')
        save_image(filename, self.holo)
        warn_msg = "Metadata detected but ignored. Use hp.load to read it."
        with self.assertWarns(UserWarning) as cm:
            load_image(filename + '.tif')
        self.assertTrue(str(cm.warning) == warn_msg)

    @attr("fast")
    def test_no_metadata(self):
        filename = os.path.join(self.tempdir, 'image0007.tif')
        header = ifd2()
        header[270] = 'Dummy String'
        pilimage.fromarray(self.holo.values[0]).save(filename, tiffinfo=header)
        # load doesn't work
        self.assertRaises(NoMetadata, load, filename)
        # load_image does
        l = load_image(filename, spacing=get_spacing(self.holo))
        assert_obj_close(l, copy_metadata(l, self.holo))


class test_custom_yaml_output(unittest.TestCase):
    @attr("fast")
    def test_yaml_output_of_numpy_types(self):
        a = np.ones(10, 'int')
        assert_equal(yaml.dump(a.std()), '0.0\n...\n')
        assert_equal(yaml.dump(np.dtype('float')), "!dtype 'float64'\n")
        assert_equal(yaml.load(yaml.dump(np.dtype('float'))), np.dtype('float64'))
        try:
            assert_equal(yaml.dump(a.max()), '1\n...\n')
        except AssertionError as err:
            if err.args[0] == r"""
    Items are not equal:
     ACTUAL: '!!python/object/apply:numpy.core.multiarray.scalar [!dtype \'int32\', "\\x01\\0\\0\\0"]\n'
     DESIRED: '1\n...\n'""":
                raise AssertionError("You're probably running a 32 bit OS.  Writing and reading files with integers might be buggy on 32 bit OS's. We don't think it will lead to data loss, but we make no guarantees. If you see this on 64 bit operating systems, please let us know by filing a bug.")
            else:
                raise err

    @attr("fast")
    def test_yaml_output_of_serializable(self):
        class S(Serializable):
            def __init__(self, a):
                self.a = a
        assert yaml.dump(S('a')) == '!S {a: a}\n'
