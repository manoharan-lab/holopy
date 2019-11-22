# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, and Ryan McGorty, Anna Wang
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
from holopy.scattering import Sphere
from holopy.inference import sample, fit, prior, AlphaModel, EmceeStrategy
from holopy.inference.interface import (
    make_default_model, parameterize_scatterer, rename_xyz, make_uniform)
from holopy.inference.result import SamplingResult
from holopy.inference.tests.common import SimpleModel

DATA = data_grid(np.ones((2, 2)), spacing=1, medium_index=1,
    illum_wavelen=0.5, illum_polarization = [0,1])
SPHERE = Sphere(n=1, center=[2, 2, 2])
GUESSES = {'n': 1, 'r': 2, 'center.0': 3}

class TestUserFacingFunctions(unittest.TestCase):
    @attr('fast')
    def test_cannot_sample_without_model(self):
        self.assertRaises(ValueError, sample, DATA, Sphere())

    @attr('medium')
    def test_sample_function_samples(self):
        result = sample(DATA, SimpleModel())
        self.assertTrue(isinstance(result, SamplingResult))
        self.assertTrue(isinstance(result.strategy, EmceeStrategy))
        self.assertTrue(hasattr(result, 'samples'))

    @attr('medium')
    def test_fit_works_with_model(self):
        function_result = fit(DATA, SimpleModel())
        function_result.time = None
        object_result = SimpleModel().fit(DATA)
        object_result.time = None
        self.assertEqual(function_result, object_result)

    @attr('medium')
    def test_fit_works_with_scatterer(self):
        function_result = fit(DATA, SPHERE)
        function_result.time = None
        model = make_default_model(SPHERE, None)
        object_result = model.fit(DATA)
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
        object_result = model.fit(DATA)
        object_result.time = None
        self.assertEqual(function_result, object_result)

    @attr('medium')
    def test_model_takes_precendence_over_parameters(self):
        model = make_default_model(SPHERE, ['n', 'x'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            result = fit(DATA, model, ['r', 'y'])
        self.assertEqual(result._names, ['n', 'center.0', 'alpha'])

    @attr('medium')
    def test_passing_model_and_parameters_gives_warning(self):
        model = make_default_model(SPHERE, None)
        self.assertWarns(UserWarning, fit, DATA, model, ['r', 'y'])


class TestHelperFunctions(unittest.TestCase):
    @attr('fast')
    def test_make_default_model_with_no_parameters(self):
        model = make_default_model(SPHERE, None)
        expected_parameters = SPHERE.parameters
        expected_parameters['alpha'] = None
        self.assertEqual(model.parameters.keys(), expected_parameters.keys())

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
        scatterer = parameterize_scatterer(Sphere(), 'r')
        self.assertTrue(isinstance(scatterer.r, prior.Prior))

    @attr('fast')
    def test_rename_xyz(self):
        parameters_list = ['x', 'p', 'y']
        parameters_list = rename_xyz(parameters_list)
        self.assertEqual(parameters_list, ['center.0', 'p', 'center.1'])

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

