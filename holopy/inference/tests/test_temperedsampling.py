import unittest

import numpy as np
from nose.plugins.attrib import attr

from holopy.inference import prior
from holopy.core.process import normalize
from holopy.scattering import Sphere, Mie
from holopy.inference import prior, TemperedStrategy
from holopy.inference.model import AlphaModel, PerfectLensModel
from holopy.core.tests.common import assert_obj_close, get_example_data


class TestTemperedStrategy(unittest.TestCase):
    @unittest.skip("Incomplete test")
    def test_alpha_model(self):
        data = _get_data()
        scatterer = _make_scatterer(data)
        model = AlphaModel(scatterer, alpha=1., noise_sd=0.05)
        inference_result = _sample_model_on_data(model, data)
        has_correct_number_of_params = inference_result.MAP.size == 5
        self.assertTrue(has_correct_number_of_params)

    @unittest.skip("Incomplete test")
    def test_perfect_lens_model(self):
        data = _get_data()
        scatterer = _make_scatterer(data)
        model = PerfectLensModel(scatterer, lens_angle=0.8, noise_sd=0.05)
        inference_result = _sample_model_on_data(model, data)
        has_correct_number_of_params = inference_result.MAP.size == 5
        self.assertTrue(has_correct_number_of_params)

    @unittest.skip("Incomplete test")
    def test_exact_model(self):
        data = _get_data()
        scatterer = _make_scatterer(data)
        model = ExactModel(scatterer, noise_sd=0.05)
        inference_result = _sample_model_on_data(model, data)
        has_correct_number_of_params = inference_result.MAP.size == 5
        self.assertTrue(has_correct_number_of_params)


def _get_data():
    data = get_example_data('image0001')
    return normalize(data)


def _make_scatterer(data):
    index = prior.Gaussian(1.5, .1),
    radius = prior.BoundedGaussian(.5, .05, 0, np.inf)
    center = prior.make_center_priors(data)
    return Sphere(n=index, r=radius, center=center)


def _sample_model_on_data(model, data):
    strategy = TemperedStrategy(nwalkers=10, max_pixels=100)
    result = strategy.optimize(model, data, nsamples=5)
    return result


if __name__ == "__main__":
    unittest.main()
