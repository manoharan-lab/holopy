# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, and Ryan McGorty, Anna Wang, Solomon Barkley
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


import unittest
import warnings

import numpy as np
from nose.plugins.attrib import attr

from holopy.core.metadata import data_grid
from holopy.scattering import Sphere, Spheres
from holopy.inference import (sample, fit, prior, AlphaModel, EmceeStrategy,
                              NmpfitStrategy, CmaStrategy, TemperedStrategy)
from holopy.inference.interface import (
    make_default_model, parameterize_scatterer, make_uniform,
    validate_strategy, available_sampling_strategies, available_fit_strategies)
from holopy.inference.result import SamplingResult
from holopy.inference.tests.common import SimpleModel

DATA = data_grid(np.ones((2, 2)), spacing=1, medium_index=1,
                 illum_wavelen=0.5, illum_polarization=[0, 1])
SPHERE = Sphere(n=1, center=[2, 2, 2])
GUESSES = {'n': 1, 'r': 2, 'center.0': 3}


class TestUserFacingFunctions(unittest.TestCase):
    @attr('fast')
    def test_cannot_sample_without_model(self):
        self.assertRaises(ValueError, sample, DATA, Sphere())

    @attr('fast')
    def test_sample_function_calls_model_sample(self):
        strategy = EmceeStrategy(nsamples=1)
        result = sample(DATA, SimpleModel(), strategy=strategy)
        self.assertTrue(isinstance(result, SamplingResult))
        self.assertTrue(hasattr(result, 'samples'))

    @attr('medium')
    def test_fit_works_with_scatterer(self):
        function_result = fit(DATA, SPHERE)
        function_result.time = None
        model = make_default_model(SPHERE, None)
        object_result = fit(DATA, model)
        object_result.time = None
        self.assertEqual(function_result, object_result)

    @attr('fast')
    def test_fit_does_not_work_with_arbitrary_object(self):
        self.assertRaises(AttributeError, fit, DATA, 'string')

    @attr('medium')
    def test_fit_works_with_scatterer_and_parameters(self):
        function_result = fit(DATA, SPHERE, ['n', 'x'])
        function_result.time = None
        model = make_default_model(SPHERE, ['n', 'x'])
        object_result = fit(DATA, model)
        object_result.time = None
        self.assertEqual(function_result, object_result)

    @attr('medium')
    def test_model_takes_precendence_over_parameters(self):
        model = make_default_model(SPHERE, ['n', 'x'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            result = fit(DATA, model, ['r', 'y'])
        self.assertEqual(result._names, ['n', 'x', 'alpha'])

    @attr('medium')
    def test_passing_model_and_parameters_gives_warning(self):
        model = make_default_model(SPHERE, None)
        self.assertWarns(UserWarning, fit, DATA, model, ['r', 'y'])

    @attr('medium')
    def test_fit_function_identical_to_strategy_method(self):
        model = SimpleModel()
        strategy = NmpfitStrategy(seed=123)
        strategy_result = strategy.fit(model, DATA)
        strategy_result.time = None
        model_result = fit(DATA, model, strategy=strategy)
        model_result.time = None
        self.assertEqual(strategy_result, model_result)

    @attr('medium')
    def test_sample_function_identical_to_strategy_method(self):
        model = SimpleModel()
        strategy = EmceeStrategy(nwalkers=6, nsamples=10, seed=123)
        strategy_result = strategy.sample(model, DATA)
        strategy_result.time = None
        model_result = sample(DATA, model, strategy)
        model_result.time = None
        self.assertEqual(strategy_result, model_result)


class TestStrategyHandling(unittest.TestCase):
    @attr('medium')
    def test_default_fit_strategy_is_Nmpfit(self):
        result = fit(DATA, SimpleModel())
        self.assertEqual(result.strategy, NmpfitStrategy())

    @attr('slow')
    def test_default_sampling_strategy_is_emcee(self):
        # for speed, we monkey-patch emcee.default_nsamples
        # FIXME this is maybe not the best way to do this.
        put_back = EmceeStrategy._default_nsamples * 1
        EmceeStrategy._default_nsamples = 1
        result = sample(DATA, SimpleModel())
        self.assertTrue(isinstance(result.strategy, EmceeStrategy))
        # and put it back!!
        EmceeStrategy._default_nsamples = put_back
        self.assertNotEqual(EmceeStrategy._default_nsamples, 1)

    @attr('fast')
    def test_fit_strategy_names(self):
        for name, strategy in available_fit_strategies.items():
            strategy_by_name = validate_strategy(name, 'fit')
            self.assertEqual(strategy(), strategy_by_name)

    @attr('fast')
    def test_sample_strategy_names(self):
        for name, strategy in available_sampling_strategies.items():
            if strategy is not NotImplemented:
                strategy_by_name = validate_strategy(name, 'sample')
                self.assertEqual(strategy(), strategy_by_name)

    @attr('fast')
    def test_parallel_tempering_not_implemented(self):
        self.assertRaises(ValueError, validate_strategy,
                          'parallel tempering', 'sample')

    @attr('medium')
    def test_fit_takes_strategy_object(self):
        strategy = NmpfitStrategy(npixels=2, maxiter=2)
        result = fit(DATA, SimpleModel(), strategy=strategy)
        self.assertEqual(result.strategy, strategy)

    @attr('medium')
    def test_sample_takes_strategy_object(self):
        strategy = EmceeStrategy(nsamples=2)
        result = sample(DATA, SimpleModel(), strategy)
        self.assertEqual(result.strategy, strategy)

    @attr('medium')
    def test_fit_takes_strategy_by_name(self):
        result = fit(DATA, SimpleModel(), strategy='cma')
        self.assertTrue(isinstance(result.strategy, CmaStrategy))

    @attr('fast')
    def test_fit_fails_with_sampling_strategy(self):
        self.assertRaises(ValueError, fit,
                          DATA, SimpleModel(), strategy=EmceeStrategy)

    @attr('fast')
    def test_sample_fails_with_fitting_strategy(self):
        self.assertRaises(ValueError, sample,
                          DATA, SimpleModel(), NmpfitStrategy)


class TestHelperFunctions(unittest.TestCase):
    @attr('fast')
    def test_make_default_model_with_no_parameters(self):
        model = make_default_model(SPHERE, None)
        expected = {'n', 'r', 'alpha', 'x', 'y', 'z'}
        self.assertEqual(set(model.parameters.keys()), expected)

    @attr('fast')
    def test_make_default_model_with_parameters(self):
        model = make_default_model(SPHERE, ['r'])
        self.assertEqual(set(model.parameters.keys()), {'r', 'alpha'})

    @attr('fast')
    def test_make_default_model_construction(self):
        model = make_default_model(Sphere(), None)
        self.assertEqual(model.noise_sd, 1)
        self.assertTrue(isinstance(model, AlphaModel))

    @attr('fast')
    def test_parameterize_scatterer_makes_priors(self):
        scatterer = parameterize_scatterer(Sphere(), ['r'])
        self.assertTrue(isinstance(scatterer.r, prior.Prior))

    @attr('fast')
    def test_parameterize_scatterer_xyz(self):
        scatterer = parameterize_scatterer(Sphere(center=[0, 0, 0]), ['x'])
        self.assertTrue(isinstance(scatterer.center[0], prior.Prior))

    @attr('fast')
    def test_parameterize_scatterer_center(self):
        fit_pars = ['center']
        scatterer = parameterize_scatterer(Sphere(center=[0, 0, 0]), fit_pars)
        for i, coord in enumerate('xyz'):
            expected = prior.Uniform(-np.inf, np.inf, 0, coord)
        self.assertEqual(scatterer.center[i], expected)

    @attr('fast')
    def test_parameterize_scatterer_spheres(self):
        sphere = Sphere(r=0.5, n=1, center=[0, 0, 0])
        model = make_default_model(Spheres([sphere, sphere], warn=False))
        expected = {'0:n', '1:n', '0:r', '1:r', 'alpha',
                    '0:x', '0:y', '0:z', '1:x', '1:y', '1:z'}
        self.assertEqual(set(model.parameters.keys()), expected)

    @attr('fast')
    def test_parameterize_scatterer_spheres_minval(self):
        sphere = Sphere(r=0.5, n=1, center=[0, 0, 0])
        model = make_default_model(Spheres([sphere, sphere], warn=False))
        self.assertEqual(model.parameters['0:n'].lower_bound, 0)
        self.assertEqual(model.parameters['1:n'].lower_bound, 0)

    @attr('fast')
    def test_parameterize_scatterer_spheres_by_given(self):
        s1 = Sphere(r=0.5, n=1, center=[0, 0, 0])
        s2 = Sphere(r=0.5, n=1, center=[1, 1, 1])
        model = make_default_model(Spheres([s1, s2]), ['0:r', '1:x'])
        e1 = Sphere(r=prior.Uniform(0, np.inf, 0.5, '0:r'),
                    n=1, center=[0, 0, 0])
        e2 = Sphere(r=0.5, n=1,
                    center=[prior.Uniform(-np.inf, np.inf, 1, '1:x'), 1, 1])
        self.assertEqual(model.scatterer, Spheres([e1, e2]))

    @attr('fast')
    def test_make_uniform_fails_with_unexpected_parameters(self):
        self.assertRaises(ValueError, make_uniform, GUESSES, 'q')

    @attr('fast')
    def test_make_uniform_with_positive_variables(self):
        for variable in ['r', 'n']:
            generated = make_uniform(GUESSES, variable)
            expected = prior.Uniform(0, np.inf, GUESSES[variable], variable)
            self.assertEqual(generated, expected)

    @attr('fast')
    def test_make_uniform_unbounded_variable(self):
        generated = make_uniform(GUESSES, 'center.0')
        expected = prior.Uniform(-np.inf, np.inf, 3, 'center.0')
        self.assertEqual(generated, expected)


if __name__ == '__main__':
    unittest.main()
