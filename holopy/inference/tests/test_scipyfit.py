import unittest
import warnings

import numpy as np
from nose.plugins.attrib import attr

from holopy.scattering import Sphere, Mie, calc_holo
from holopy.core.process import normalize
from holopy.core.tests.common import get_example_data
from holopy.inference import AlphaModel, LeastSquaresScipyStrategy
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


if __name__ == '__main__':
    unittest.main()

