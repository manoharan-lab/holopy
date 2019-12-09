import unittest
import warnings

import numpy as np
from nose.plugins.attrib import attr

import holopy
from holopy.scattering import Sphere, Mie, calc_holo
from holopy.core.process import normalize
from holopy.core.tests.common import get_example_data
from holopy.inference import (
    AlphaModel, LeastSquaresScipyStrategy, NmpfitStrategy)
from holopy.inference.prior import Uniform


CORRECT_ALPHA = .6497
CORRECT_SPHERE = Sphere(
    n=1.582+1e-4j, r=6.484e-7, center=(5.534e-6, 5.792e-6, 1.415e-5))


class TestLeastSquaresScipyStrategy(unittest.TestCase):
    @attr("slow")
    def test_fit_mie_par_scatterer(self):
        holo = normalize(get_example_data('image0001'))
        model = make_model()

        fitter = LeastSquaresScipyStrategy(max_nfev=25)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = fitter.fit(model, holo)
        fitted = result.scatterer

        self.assertTrue(np.isclose(fitted.n, CORRECT_SPHERE.n, rtol=1e-3))
        self.assertTrue(np.isclose(fitted.r, CORRECT_SPHERE.r, rtol=1e-3))
        self.assertTrue(
            np.allclose(fitted.center, CORRECT_SPHERE.center, rtol=1e-3))
        self.assertTrue(
            np.isclose(result.parameters['alpha'], CORRECT_ALPHA, rtol=0.1))
        self.assertEqual(model, result.model)

    @attr('medium')
    def test_fit_random_subset(self):
        holo = normalize(get_example_data('image0001'))
        model = make_model()

        np.random.seed(40)
        fitter = LeastSquaresScipyStrategy(npixels=1000, max_nfev=25)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = fitter.fit(model, holo)
        fitted = result.scatterer

        self.assertTrue(np.isclose(fitted.n, CORRECT_SPHERE.n, rtol=1e-2))
        self.assertTrue(np.isclose(fitted.r, CORRECT_SPHERE.r, rtol=1e-2))
        self.assertTrue(
            np.allclose(fitted.center, CORRECT_SPHERE.center, rtol=1e-2))
        self.assertTrue(
            np.isclose(result.parameters['alpha'], CORRECT_ALPHA, rtol=0.1))
        self.assertEqual(model, result.model)

    @attr('medium')
    def test_fitted_parameters_similar_to_nmpfit(self):
        holo = normalize(get_example_data('image0001'))
        model = make_model()

        np.random.seed(40)
        fitter_scipy = LeastSquaresScipyStrategy(npixels=500)
        result_scipy = fitter_scipy.fit(model, holo)
        params_scipy = result_scipy.parameters

        np.random.seed(40)
        fitter_nmp = NmpfitStrategy(npixels=500)
        result_nmp = fitter_nmp.fit(model, holo)
        params_nmp = result_nmp.parameters

        for key in params_scipy.keys():
            self.assertAlmostEqual(
                params_scipy[key],
                params_nmp[key],
                places=4)

    @attr('medium')
    def test_fitted_uncertainties_scale_with_number_of_points(self):
        holo = normalize(get_example_data('image0001'))
        model = make_model()

        np.random.seed(40)
        fitter_100 = LeastSquaresScipyStrategy(npixels=100)
        result_100 = fitter_100.fit(model, holo)
        uncertainties_100 = pack_uncertainties_into_dict(result_100)

        np.random.seed(40)
        fitter_900 = LeastSquaresScipyStrategy(npixels=900)
        result_900 = fitter_900.fit(model, holo)
        uncertainties_900 = pack_uncertainties_into_dict(result_900)

        for key in uncertainties_100.keys():
            self.assertTrue(
                np.isclose(
                    uncertainties_100[key],
                    3 * uncertainties_900[key],
                    rtol=0.3, atol=0))

    @attr('medium')
    def test_1_sigma_uncertainty_increases_logpost_by_half(self):
        data, model = make_fake_data_and_1_parameter_model()

        fitter = LeastSquaresScipyStrategy()
        result = fitter.fit(model, data)
        uncertainties = pack_uncertainties_into_dict(result)

        parameters_best = result.parameters
        parameters_1sig = {'alpha': result.parameters['alpha'] +
                                    uncertainties['alpha']}
        loglikelihood_best = model.lnposterior(parameters_best, data)
        loglikelihood_1sig = model.lnposterior(parameters_1sig, data)
        delta_loglikelihood = loglikelihood_best - loglikelihood_1sig
        self.assertAlmostEqual(delta_loglikelihood, 0.5, places=2)

    @attr('medium')
    def test_fitted_uncertainties_similar_to_nmpfit(self):
        data, model = make_fake_data_and_1_parameter_model()

        fitter_scipy = LeastSquaresScipyStrategy()
        result_scipy = fitter_scipy.fit(model, data)
        uncertainties_scipy = pack_uncertainties_into_dict(result_scipy)

        fitter_nmp = NmpfitStrategy()
        result_nmp = fitter_nmp.fit(model, data)
        uncertainties_nmp = pack_uncertainties_into_dict(result_nmp)

        for key in uncertainties_scipy.keys():
            self.assertTrue(
                np.isclose(
                    uncertainties_scipy[key],
                    uncertainties_nmp[key],
                    rtol=0.1, atol=0))


def make_model():
    center_guess = [
        Uniform(0, 1e-5, name='x', guess=.567e-5),
        Uniform(0, 1e-5, name='y', guess=.576e-5),
        Uniform(1e-5, 2e-5, name='z', guess=15e-6),
        ]
    scatterer = Sphere(
        n=Uniform(1, 2, name='n', guess=1.59),
        r=Uniform(1e-8, 1e-5, name='r', guess=8.5e-7),
        center=center_guess)
    alpha = Uniform(0.1, 1, name='alpha', guess=0.6)
    theory = Mie(compute_escat_radial=False)
    model = AlphaModel(scatterer, theory=theory, alpha=alpha)
    return model


def pack_uncertainties_into_dict(fit_result):
    intervals = fit_result.intervals
    return {v.name: v.plus for v in intervals}


def make_fake_data_and_1_parameter_model():
    # the same extent as the example data, but 10x10 instead of 100x100
    # The only parameter is alpha
    detector = holopy.detector_grid([100, 100], spacing=1.151e-7)
    sphere = Sphere(n=1.59, r=8e-7, center=(5.7e-6, 5.7e-6, 15e-6))
    fake_data = calc_holo(
        detector,
        sphere,
        medium_index=1.33,
        illum_wavelen=6.58e-7,
        illum_polarization=(1, 0),
        theory='auto',
        scaling=0.7,)

    alpha = Uniform(0.1, 1, name='alpha', guess=0.6)
    theory = Mie(compute_escat_radial=False)
    model = AlphaModel(sphere, theory=theory, alpha=alpha)
    return fake_data, model


if __name__ == '__main__':
    unittest.main()

