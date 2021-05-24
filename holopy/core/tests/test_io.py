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


import importlib
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

import holopy as hp
from holopy.core import load, save, load_image, save_image, save_images
from holopy.core.errors import NoMetadata
from holopy.core.io import load_average, get_example_data_path
from holopy.core.io.io import Accumulator
from holopy.core.process import normalize
from holopy.core.metadata import get_spacing, copy_metadata
from holopy.core.holopy_object import HoloPyObject
from holopy.core.tests.common import (
    assert_obj_close, assert_read_matches_write, get_example_data)


IMAGE01_METADATA = {'spacing': 0.0851, 'medium_index': 1.33,
                    'illum_wavelen': 0.66, 'illum_polarization':  (1,0)}

class TestLoadingAndSaving(unittest.TestCase):
    def setUp(self):
        self.holo = get_example_data('image0001')
        self.holograms = [get_example_data('image0001'),
                          get_example_data('image0002')]
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def load_image_with_metadata(self, filename):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded = load_image(filename,
                                name=self.holo.name,
                                medium_index=self.holo.medium_index,
                                spacing=get_spacing(self.holo),
                                illum_wavelen=self.holo.illum_wavelen,
                                illum_polarization=self.holo.illum_polarization,
                                noise_sd=self.holo.noise_sd)
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
    def test_save_images_checks_names_and_holograms_are_same_length(self):
        filenames_too_long = [
            os.path.join(self.tempdir, 'dummy_filename_{}.tif'.format(i))
            for i in range(len(self.holograms) + 1)]

        self.assertRaises(
            ValueError,
            save_images,
            filenames_too_long,
            self.holograms)

    @attr("fast")
    def test_save_images(self):
        filenames = [
            os.path.join(self.tempdir, f)
            for f in ['dummy_filename_1.tif', 'dummy_filename_2.tif']]

        save_images(filenames, self.holograms, scaling=None)

        for ground_truth, filename in zip(self.holograms, filenames):
            loaded = load(filename)
            self.assertTrue(np.all(loaded.values ==  ground_truth.values))

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
        assert_equal(yaml.dump(a.max()), '1\n...\n')

    @attr("fast")
    def test_yaml_output_of_serializable(self):
        class S(HoloPyObject):
            def __init__(self, a, b, c=3):
                self.a = a
                self.b = b
                self.c = c
        instantiated = S(1, None)
        setattr(instantiated, 'd', 'e')
        # only a & c appear because b is None, d is not in __init__ signature
        assert yaml.dump(instantiated) == '!S\na: 1\nc: 3\n'

    @attr("fast")
    def test_custom_treatment_of_length_zero_ndarray(self):
        zero_length_array = np.array(3)
        self.assertEqual(yaml.dump(zero_length_array), yaml.dump(3))

    @attr("fast")
    def test_custom_treatment_of_length_one_ndarray(self):
        one_length_array = np.array([3])
        self.assertEqual(yaml.dump(one_length_array), yaml.dump([3]))

    @attr("fast")
    def test_custom_treatment_of_longer_ndarray(self):
        longer_array = 3 * np.ones(10, dtype=int)
        self.assertEqual(yaml.dump(longer_array), yaml.dump([3] * 10))

    @attr("fast")
    def test_custom_treatment_of_multidim_ndarray(self):
        multi_D_array = 3 * np.ones((2, 2), dtype=int)
        self.assertEqual(yaml.dump(multi_D_array), yaml.dump([[3, 3], [3, 3]]))

    @attr("fast")
    def test_custom_yaml_dump_of_numpy_ufunc(self):
        self.assertEqual(yaml.dump(np.sqrt), "!ufunc \'sqrt\'\n")

    @attr("fast")
    def test_custom_yaml_load_of_numpy_ufunc(self):
        yaml_text = "!ufunc \'sqrt\'\n"
        self.assertEqual(yaml.load(yaml_text, Loader=yaml.FullLoader), np.sqrt)


class TestMemoryUsage(unittest.TestCase):
    @unittest.skipIf(not importlib.util.find_spec('memory_profiler'),
                     'memory_profiler is reqruired for this test')
    @unittest.expectedFailure
    def test_load_average_doesnt_use_excess_mem(self):
        # TODO: Why does load_average use so much memory?
        # See manoharan-lab/holopy#267
        import memory_profiler
        refimg = _load_raw_example_data()
        paths = get_example_data_path(['bg01.jpg', 'bg02.jpg', 'bg03.jpg'])
        usage = memory_profiler.memory_usage((load_average, (paths, refimg,)),
                                             interval=1e-5)
        peak_usage = np.ptp(usage)
        images = _load_example_data_backgrounds()
        expected_usage = sum([im.nbytes / 1e6 for im in images]) # Size in MB
        self.assertTrue(peak_usage < expected_usage * 1.1)


class TestAccumulator(unittest.TestCase):
    @attr("fast")
    def test_push(self):
        accumulator = Accumulator()
        data  = np.arange(10)
        for point in data: accumulator.push(point)
        self.assertTrue(accumulator._n == 10)

    @attr("fast")
    def test_push_hologram(self):
        accumulator = Accumulator()
        data = _load_example_data_backgrounds()
        for holo in data: accumulator.push(holo)
        self.assertTrue(accumulator._n == 3)

    @attr("fast")
    def test_mean(self):
        accumulator = Accumulator()
        data = np.arange(10)
        for point in data: accumulator.push(point)
        self.assertTrue(accumulator.mean() == np.mean(data))

    @attr("fast")
    def test_mean_hologram_value(self):
        accumulator = Accumulator()
        data = _load_example_data_backgrounds()
        for holo in data: accumulator.push(holo)
        numpy_mean = np.mean([holo.values for holo in data], axis=0)
        self.assertTrue(np.allclose(numpy_mean, accumulator.mean().values))

    @attr("fast")
    def test_mean_hologram_type(self):
        import xarray
        expected_type = xarray.core.dataarray.DataArray
        accumulator = Accumulator()
        data = _load_example_data_backgrounds()
        for holo in data: accumulator.push(holo)
        self.assertTrue(isinstance(accumulator.mean(), expected_type))

    @attr("fast")
    def test_std(self):
        accumulator = Accumulator()
        data = np.arange(10)
        for point in data: accumulator.push(point)
        self.assertTrue(accumulator._std() == np.std(data))

    @attr("fast")
    def test_std_no_data(self):
        accumulator = Accumulator()
        self.assertTrue(accumulator._std() is None)

    @attr("fast")
    def test_cv(self):
        accumulator = Accumulator()
        data = np.arange(10)
        for point in data: accumulator.push(point)
        self.assertTrue(accumulator.cv() == np.std(data) / np.mean(data))

    @attr("fast")
    def test_cv_no_data(self):
        accumulator = Accumulator()
        self.assertTrue(accumulator.cv() is None)

    @attr("medium")
    def test_calculate_hologram_noise_sd(self):
        accumulator = Accumulator()
        refimg = _load_raw_example_data()
        paths = get_example_data_path(['bg01.jpg', 'bg02.jpg', 'bg03.jpg'])
        bg = load_average(paths, refimg)
        # This value is from the legacy version of load_average
        self.assertTrue(np.allclose(bg.noise_sd, 0.00709834))

    @attr('fast')
    def test_2_colour_noise_sd(self):
        paths = get_example_data_path(['2colourbg0.jpg', '2colourbg1.jpg',
                                       '2colourbg2.jpg', '2colourbg3.jpg'])
        image = load_average(paths, spacing=1, channel=[0,1])
        gold_noise = [0.06864433355667054, 0.04913377621162473]
        noise = [image.noise_sd.loc[colour].item()
                 for colour in ['green', 'red']]
        self.assertTrue(np.allclose(gold_noise, noise))


def _load_raw_example_data():
    imagepath = get_example_data_path('image01.jpg')
    return load_image(imagepath, **IMAGE01_METADATA)

def _load_example_data_backgrounds():
    bgpath = get_example_data_path(['bg01.jpg', 'bg02.jpg', 'bg03.jpg'])
    return [load_image(path, **IMAGE01_METADATA) for path in bgpath]

if __name__ == '__main__':
    unittest.main()
