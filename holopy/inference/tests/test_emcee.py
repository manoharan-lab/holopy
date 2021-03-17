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
from numpy.testing import assert_equal, assert_allclose
from nose.plugins.attrib import attr

from holopy.inference import prior, TemperedStrategy
from holopy.core.process import normalize
from holopy.core.tests.common import assert_obj_close, get_example_data
from holopy.scattering import Sphere, Mie, MieLens
from holopy.inference import prior
from holopy.inference.model import AlphaModel, Model
from holopy.inference.emcee import sample_emcee, EmceeStrategy
from holopy.inference.tests.common import SimpleModel


class testEmcee(unittest.TestCase):
    @attr("fast")
    def test_BaseModel_lnprior(self):
        scat = Sphere(r=prior.Gaussian(1, 1), n=prior.Gaussian(1, 1),
                      center=[10, 10, 10])
        mod = Model(scat, noise_sd=0.1)
        # Desired: log(sqrt(0.5/pi))-1/2
        desired_sigma = -1.4189385332
        assert_obj_close(mod.lnprior([0, 0]), desired_sigma * 2)

    @attr("medium")
    def test_sample_emcee(self):
        data = np.array(.5)
        nwalkers = 10
        ndim = 1
        mod = SimpleModel(1)
        p0 = np.linspace(0, 1, nwalkers*ndim).reshape((nwalkers, ndim))
        sampler = sample_emcee(
            mod, data, nwalkers, 500, p0, parallel=None, seed=40)
        try:
            chain = sampler.get_chain()
            lnprob = sampler.get_log_prob()
        except AttributeError:
            # emcee version < 3.0.0
            chain = sampler.chain
            lnprob = sampler.lnprobability
        should_be_onehalf = chain[lnprob == lnprob.max()]
        assert_allclose(should_be_onehalf, .5, rtol=.001)

    @attr("fast")
    def test_EmceeStrategy(self):
        data = np.array(.5)
        mod = SimpleModel(1)
        strat = EmceeStrategy(10, 15, None, None, seed=48)
        r = strat.sample(mod, data)
        assert_allclose(r._parameters, .5, rtol=.001)


class TestSubsetTempering(unittest.TestCase):
    @attr("slow")
    def test_alpha_subset_tempering(self):
        holo = normalize(get_example_data('image0001'))
        scat = Sphere(r=0.65e-6, n=1.58, center=[5.5e-6, 5.8e-6, 14e-6])
        mod = AlphaModel(scat, noise_sd=.1, alpha=prior.Gaussian(0.7, 0.1))
        strat = TemperedStrategy(nwalkers=4, nsamples=10, stages=1,
                                 stage_len=10, parallel=None, seed=40)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            inference_result = strat.sample(mod, holo)
        desired_alpha = np.array([0.650348])
        assert_allclose(inference_result._parameters, desired_alpha, rtol=5e-3)

    @attr("slow")
    def test_perfectlens_subset_tempering(self):
        data = normalize(get_example_data('image0001'))
        scatterer = Sphere(r=0.65e-6, n=1.58, center=[5.5e-6, 5.8e-6, 14e-6])
        model = AlphaModel(
            scatterer, noise_sd=.1,
            theory=MieLens(lens_angle=prior.Gaussian(0.7, 0.1)))
        strat = TemperedStrategy(nwalkers=4, nsamples=10, stages=1,
                                 stage_len=10, parallel=None, seed=40)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            inference_result = strat.sample(model, data)
        desired_lens_angle = np.array([0.7197887])
        is_ok = np.allclose(
            inference_result._parameters, desired_lens_angle, rtol=1e-3)
        self.assertTrue(is_ok)


if __name__ == '__main__':
    unittest.main()
