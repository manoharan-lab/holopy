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


import warnings
import unittest

import numpy as np
from numpy.testing import assert_equal
from nose.plugins.attrib import attr

from holopy.inference import prior, sample
from holopy.core.process import normalize
from holopy.core.tests.common import assert_obj_close, get_example_data
from holopy.scattering import Sphere, Mie
from holopy.fitting import model
from holopy.inference.noise_model import (
    AlphaModel, NoiseModel, PerfectLensModel)

# GOLD: log(sqrt(0.5/pi))-1/2
gold_sigma = -1.4189385332

# GOLD: inference result - depends on both seeds
gold_alpha = np.array([0.649683])
gold_lens_angle = np.array([0.675643])


@attr("fast")
def test_NoiseModel_lnprior():
    scat = Sphere(r=prior.Gaussian(1, 1), n=prior.Gaussian(1, 1),
                  center=[10, 10, 10])
    mod = NoiseModel(scat, noise_sd=.1)
    assert_obj_close(mod.lnprior([0, 0]), gold_sigma * 2)


class TestSubsetTempering(unittest.TestCase):
    def test_alpha_subset_tempering(self):
        holo = normalize(get_example_data('image0001'))
        scat = Sphere(r=0.65e-6, n=1.58, center=[5.5e-6, 5.8e-6, 14e-6])
        mod = AlphaModel(scat, noise_sd=.1, alpha=prior.Gaussian(0.7, 0.1))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            inf = sample.tempered_sample(
                mod, holo, nwalkers=4, samples=10, stages=1, stage_len=10,
                threads=None, seed=40)
        assert_obj_close(inf.MAP, gold_alpha, rtol=1e-3)

    def test_perfectlens_subset_tempering(self):
        data = normalize(get_example_data('image0001'))
        scatterer = Sphere(r=0.65e-6, n=1.58, center=[5.5e-6, 5.8e-6, 14e-6])
        model = PerfectLensModel(
            scatterer, noise_sd=.1, lens_angle=prior.Gaussian(0.7, 0.1))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            inference_result = sample.tempered_sample(
                model, data, nwalkers=4, samples=10, stages=1, stage_len=10,
                threads=None, seed=40)
        assert_obj_close(inference_result.MAP, gold_lens_angle, rtol=1e-3)

