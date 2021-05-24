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

import xarray as xr
import numpy as np
from nose.plugins.attrib import attr

from holopy.inference import prior
from holopy.inference.result import UncertainValue, FitResult, SamplingResult
from holopy.inference import (
    AlphaModel, CmaStrategy, EmceeStrategy, NmpfitStrategy)
from holopy.scattering import Sphere, Mie
from holopy.scattering.errors import MissingParameter
from holopy.core.metadata import detector_grid, update_metadata
from holopy.core.errors import DeprecationError
from holopy.core.tests.common import (
    assert_read_matches_write, get_example_data)


DATA = update_metadata(detector_grid(shape=10, spacing=2), 1.33, 0.660, (0, 1))
par_s = Sphere(n=prior.Uniform(1.5, 1.65),
               r=prior.Uniform(0.5, 0.7), center=[10, 10, 10])
MODEL = AlphaModel(par_s, alpha=prior.Uniform(0.6, 0.9, guess=0.8))
INTERVALS = [UncertainValue(1.6, 0.1, name='n'), UncertainValue(0.6,
                        0.1, name='r'), UncertainValue(0.7, 0.1, name='alpha')]

def generate_fit_result():
    return FitResult(DATA, MODEL, CmaStrategy(), 10, {'intervals': INTERVALS})


def generate_sampling_result():
    samples = np.array([[[1, 2], [11, 12], [3, 3]], [[0, 3], [1, 3], [5, 6]]])
    samples = xr.DataArray(samples,
                           dims = ['walker', 'chain', 'parameter'],
                            coords={'parameter':['p1','p2']})
    lnprobs = xr.DataArray([[10, 9, 8], [7, 6, 5]], dims = ['walker', 'chain'])
    result = SamplingResult(DATA, MODEL, EmceeStrategy(), 10,
                            kwargs={'samples':samples, 'lnprobs':lnprobs})
    return result


class TestUncertainValue(unittest.TestCase):
    @attr("fast")
    def test_optional_assymetric_uncertainty(self):
        uncval1 = UncertainValue([10], np.array(2))
        uncval2 = UncertainValue(10, 2, 2)
        self.assertEqual(uncval1, uncval2)


class TestFitResult(unittest.TestCase):
    def _make_model(self):
        sphere = Sphere(
            center=(prior.Uniform(0, 1e-5, guess=.567e-5),
                    prior.Uniform(0, 1e-5, .567e-5),
                    prior.Uniform(1e-5, 2e-5)),
            r=prior.Uniform(1e-8, 1e-5, 8.5e-7),
            n=prior.Uniform(1, 2, 1.59))

        model = AlphaModel(
            sphere, theory=Mie(False), alpha=prior.Uniform(0.1, 1, 0.6))
        return model

    @attr("fast")
    def test_failure_if_no_intervals(self):
        self.assertRaises(MissingParameter,
                          FitResult, DATA, MODEL, CmaStrategy(), 10)

    @attr("fast")
    def test_properties(self):
        result = generate_fit_result()
        self.assertEqual(result._parameters, [1.6, 0.6, 0.7])
        self.assertEqual(result._names, ['n', 'r', 'alpha'])
        self.assertEqual(result.parameters,
                         {'r': 0.6, 'n': 1.6, 'alpha': 0.7})
        self.assertEqual(result.scatterer,
                         Sphere(n=1.6, r=0.6, center=[10, 10, 10]))

    @attr("medium")
    def test_hologram(self):
        result = generate_fit_result()
        self.assertAlmostEqual(
            result.hologram.mean().item(), 1.005387, places=6)
        self.assertTrue(hasattr(result, '_hologram'))

    @attr("medium")
    def test_best_fit_throws_exception(self):
        result = generate_fit_result()
        self.assertRaises(DeprecationError, result.best_fit)

    @attr("medium")
    def test_max_lnprob(self):
        result = generate_fit_result()
        self.assertAlmostEqual(result.max_lnprob, -138.17557, places=5)
        self.assertTrue(hasattr(result, '_max_lnprob'))

    @attr("fast")
    def test_calculate_first_time(self):
        result = generate_fit_result()
        dummy_string = "a dummy string"
        dummy_name = "dummy_attr_name"
        def dummy_function():
            return dummy_string
        result._calculate_first_time(dummy_name, dummy_function)
        self.assertEqual(getattr(result, dummy_name), dummy_string)
        self.assertEqual(result._kwargs_keys[-1], dummy_name)

    @attr("fast")
    def test_values_only_calculated_once(self):
        result = generate_fit_result()
        calculations = ['max_lnprob', 'hologram', 'guess_hologram']
        for calculation in calculations:
            random_val = np.random.rand()
            setattr(result, '_' + calculation, random_val)
            self.assertEqual(getattr(result, calculation), random_val)

    @attr("fast")
    def test_add_attr(self):
        result = generate_fit_result()
        result.add_attr({'foo':'bar', 'foobar':7})
        self.assertEqual(result.foo, 'bar')

    @attr("medium")
    def test_guesses_match_model(self):
        result = generate_fit_result()
        model = result.model
        guess_parameters = {key: val.guess
                            for key, val in model.parameters.items()}
        guess_scatterer = model.scatterer_from_parameters(model.initial_guess)
        guess_hologram = model.forward(model.initial_guess, result.data)
        self.assertEqual(result.guess_parameters, guess_parameters)
        self.assertEqual(result.guess_scatterer, guess_scatterer)
        np.testing.assert_equal(result.guess_hologram.values,
                                guess_hologram.values)
        self.assertEqual(result.guess_hologram.attrs, guess_hologram.attrs)

    @attr('medium')
    def test_subset_data_fit_result_is_saveable(self):
        model = self._make_model()
        holo = get_example_data('image0001')

        np.random.seed(40)
        fitter = NmpfitStrategy(npixels=100, maxiter=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            # ignore not-converged warnings since we only do 2 iterations
            fitted = fitter.fit(model, holo)

        result = fitted  # was fix_flat(fitted)
        assert_read_matches_write(result)

    @attr('medium')
    def test_subset_data_fit_result_stores_model(self):
        model = self._make_model()
        holo = get_example_data('image0001')

        np.random.seed(40)
        fitter = NmpfitStrategy(npixels=100, maxiter=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            # ignore not-converged warnings since we only do 2 iterations
            fitted = fitter.fit(model, holo)

        self.assertEqual(model, fitted.model)


class TestIO(unittest.TestCase):
    @attr("fast")
    def test_base_io(self):
        result = FitResult(DATA, MODEL, CmaStrategy(),
                           10, {'intervals':INTERVALS})
        assert_read_matches_write(result)

    @attr("medium")
    def test_dataarray_attr(self):
        result = generate_fit_result()
        samples = xr.DataArray([[1,2,3], [4,5,6]], dims=['dim1', 'dim2'],
                coords={'dim1': ['left', 'right'], 'dim2': ['r', 'b', 'g']})
        result.add_attr({'samples': samples})
        assert_read_matches_write(result)

    @attr("medium")
    def test_saved_calculations(self):
        result = generate_fit_result()
        result.best_fit
        result.max_lnprob
        assert_read_matches_write(result)


class TestSamplingResult(unittest.TestCase):
    @attr("fast")
    def test_intervals(self):
        result = generate_sampling_result()
        self.assertEqual(result.intervals[0].guess, 1)
        self.assertEqual(result.intervals[1].guess, 2)
        self.assertEqual(result.intervals[0].name, 'p1')
        self.assertEqual(result.intervals[1].name, 'p2')

    @attr("fast")
    def test_burn_in(self):
        result = generate_sampling_result().burn_in(1)
        self.assertEqual(result.intervals[0].guess, 11)
        self.assertEqual(result.intervals[1].guess, 12)
        self.assertEqual(result.intervals[0].name, 'p1')
        self.assertEqual(result.intervals[1].name, 'p2')
        self.assertEqual(result.samples.shape, (2, 2, 2))


if __name__ == '__main__':
    unittest.main()
