import unittest
import warnings

import numpy as np
from nose.plugins.attrib import attr

import holopy
from holopy.scattering import Sphere, Mie, calc_holo
from holopy.core.process import normalize
from holopy.inference import (
    AlphaModel, LeastSquaresScipyStrategy, NmpfitStrategy)
from holopy.inference import prior


SPHERE = Sphere(n=1.59, r=8e-7, center=(5.7e-6, 5.7e-6, 15e-6))
CORRECT_ALPHA = 0.7


class TestLeastSquaresScipyStrategy(unittest.TestCase):
    @attr("slow")
    def test_fit_complete_model_on_complete_data(self):
        data = make_fake_data()
        model = make_model()

        fitter = LeastSquaresScipyStrategy(max_nfev=25)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = fitter.fit(model, data)
        fitted = result.scatterer

        self.assertTrue(np.isclose(fitted.n, SPHERE.n, rtol=1e-3))
        self.assertTrue(np.isclose(fitted.r, SPHERE.r, rtol=1e-3))
        self.assertTrue(
            np.allclose(fitted.center, SPHERE.center, rtol=1e-3))
        self.assertTrue(
            np.isclose(result.parameters['alpha'], CORRECT_ALPHA, rtol=0.1))
        self.assertEqual(model, result.model)

    @attr('medium')
    def test_fit_random_subset(self):
        data = make_fake_data()
        model = make_model()

        np.random.seed(40)
        fitter = LeastSquaresScipyStrategy(npixels=1000, max_nfev=25)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = fitter.fit(model, data)
        fitted = result.scatterer

        self.assertTrue(np.isclose(fitted.n, SPHERE.n, rtol=1e-2))
        self.assertTrue(np.isclose(fitted.r, SPHERE.r, rtol=1e-2))
        self.assertTrue(
            np.allclose(fitted.center, SPHERE.center, rtol=1e-2))
        self.assertTrue(
            np.isclose(result.parameters['alpha'], CORRECT_ALPHA, rtol=0.1))
        self.assertEqual(model, result.model)

    @attr('slow')
    def test_fitted_parameters_similar_to_nmpfit(self):
        data = make_fake_data()
        model = make_model()

        np.random.seed(40)
        fitter_scipy = LeastSquaresScipyStrategy(npixels=300, ftol=1e-6)
        result_scipy = fitter_scipy.fit(model, data)
        params_scipy = result_scipy.parameters

        np.random.seed(40)
        fitter_nmp = NmpfitStrategy(npixels=300, ftol=1e-6)
        result_nmp = fitter_nmp.fit(model, data)
        params_nmp = result_nmp.parameters

        for key in params_scipy.keys():
            self.assertAlmostEqual(
                params_scipy[key],
                params_nmp[key],
                places=4)

    @attr('slow')
    def test_fitted_uncertainties_scale_with_number_of_points(self):
        data = make_fake_data()
        model = make_model()

        np.random.seed(40)
        fitter_100 = LeastSquaresScipyStrategy(npixels=100, ftol=1e-4)
        result_100 = fitter_100.fit(model, data)
        uncertainties_100 = pack_uncertainties_into_dict(result_100)

        np.random.seed(40)
        fitter_900 = LeastSquaresScipyStrategy(npixels=900, ftol=1e-4)
        result_900 = fitter_900.fit(model, data)
        uncertainties_900 = pack_uncertainties_into_dict(result_900)

        for key in uncertainties_100.keys():
            self.assertTrue(
                np.isclose(
                    uncertainties_100[key],
                    3 * uncertainties_900[key],
                    rtol=0.3, atol=0))

    @attr('medium')
    def test_1_sigma_uncertainty_increases_logpost_by_half(self):
        data = make_fake_data()
        model = make_1_parameter_model()

        fitter = LeastSquaresScipyStrategy()
        result = fitter.fit(model, data)
        uncertainties = pack_uncertainties_into_dict(result)

        parameters_best = result._parameters
        parameters_1sig = [result._parameters[0] + uncertainties['alpha']]
        loglikelihood_best = model.lnposterior(parameters_best, data)
        loglikelihood_1sig = model.lnposterior(parameters_1sig, data)
        delta_loglikelihood = loglikelihood_best - loglikelihood_1sig
        self.assertAlmostEqual(delta_loglikelihood, 0.5, places=2)

    @attr('medium')
    @unittest.skip('Nmpfit does not unscale uncertainties')  # expectedFailure
    def test_fitted_uncertainties_similar_to_nmpfit(self):
        data = make_fake_data()
        model = make_model()

        fitter_scipy = LeastSquaresScipyStrategy(npixels=300)
        result_scipy = fitter_scipy.fit(model, data)
        uncertainties_scipy = pack_uncertainties_into_dict(result_scipy)

        fitter_nmp = NmpfitStrategy(npixels=300)
        result_nmp = fitter_nmp.fit(model, data)
        uncertainties_nmp = pack_uncertainties_into_dict(result_nmp)

        for key in uncertainties_scipy.keys():
            self.assertTrue(
                np.isclose(
                    uncertainties_scipy[key],
                    uncertainties_nmp[key],
                    rtol=0.1, atol=0))


def make_fake_data():
    detector = holopy.detector_grid([40, 40], spacing=2.878e-7)
    fake_data = calc_holo(
        detector,
        SPHERE,
        medium_index=1.33,
        illum_wavelen=6.58e-7,
        illum_polarization=(1, 0),
        theory='auto',
        scaling=CORRECT_ALPHA,)
    return fake_data


def make_1_parameter_model():
    alpha = prior.Uniform(0.1, 1, name='alpha', guess=0.6)
    theory = Mie(compute_escat_radial=False)
    model = AlphaModel(SPHERE, theory=theory, alpha=alpha)
    return model


def make_model():
    # Makes a model with all the paramters used in make_fake_data(),
    # but with the guesses slightly off from the true values
    center_guess = [
        prior.Uniform(0, 1e-5, name='x', guess=5.6e-6),
        prior.Uniform(0, 1e-5, name='y', guess=5.8e-6),
        prior.Uniform(1e-5, 2e-5, name='z', guess=14e-6),
        ]
    scatterer = Sphere(
        n=prior.Uniform(1, 2, name='n', guess=1.55),
        r=prior.Uniform(1e-8, 1e-5, name='r', guess=8.5e-7),
        center=center_guess)
    alpha = prior.Uniform(0.1, 1, name='alpha', guess=0.6)
    theory = Mie(compute_escat_radial=False)
    model = AlphaModel(scatterer, theory=theory, alpha=alpha)
    return model


def pack_uncertainties_into_dict(fit_result):
    intervals = fit_result.intervals
    return {v.name: v.plus for v in intervals}


if __name__ == '__main__':
    unittest.main()

