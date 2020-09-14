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
import tempfile
import warnings

import yaml
import numpy as np
import xarray as xr
from collections import OrderedDict

from nose.plugins.attrib import attr
from numpy.testing import assert_raises

from holopy.core import detector_grid, update_metadata, holopy_object
from holopy.core.tests.common import assert_equal, assert_obj_close
from holopy.scattering import Sphere, Spheres, Mie, calc_holo
from holopy.scattering.errors import MissingParameter
from holopy.core.tests.common import assert_read_matches_write
from holopy.inference import (prior, AlphaModel, ExactModel,
                              NmpfitStrategy, EmceeStrategy,
                              available_fit_strategies,
                              available_sampling_strategies)
from holopy.inference.model import (Model, PerfectLensModel,
                                    make_xarray, make_complex, read_map)
from holopy.inference.tests.common import SimpleModel
from holopy.scattering.tests.common import (
    xschema_lens, sphere as SPHERE_IN_METERS)
from holopy.fitting import fit_warning


class TestModel(unittest.TestCase):
    model_keywords = [
        'noise_sd',
        'medium_index',
        'illum_wavelen',
        'illum_polarization',
        ]

    @attr('fast')
    def test_initializable(self):
        scatterer = make_sphere()
        model = Model(scatterer)
        self.assertTrue(model is not None)

    @attr('fast')
    def test_yaml_round_trip_with_dict(self):
        sphere = make_sphere()
        for key in self.model_keywords:
            value = {'red': 1, 'green': 0}
            kwargs = {key: value}
            model = Model(sphere, **kwargs)
            with self.subTest(key=key):
                reloaded = take_yaml_round_trip(model)
                self.assertEqual(reloaded, model)

    @attr('fast')
    def test_yaml_round_trip_with_xarray(self):
        sphere = make_sphere()
        for key in self.model_keywords:
            value = xr.DataArray(
                [1, 0.],
                dims=['illumination'],
                coords={'illumination': ['red', 'green']})
            kwargs = {key: value}
            model = Model(sphere, **kwargs)
            with self.subTest(key=key):
                reloaded = take_yaml_round_trip(model)
                self.assertEqual(reloaded, model)

    @attr('fast')
    def test_scatterer_is_parameterized(self):
        sphere = Sphere(n=prior.Uniform(1, 2), r=prior.Uniform(0, 1))
        model = AlphaModel(sphere)
        self.assertEqual(model.scatterer, sphere)

    @attr('fast')
    def test_scatterers_maintain_attrs(self):
        spheres = Spheres([Sphere(), Sphere()], warn=False)
        model = AlphaModel(spheres)
        self.assertEqual(model.scatterer.warn, False)

    @attr('fast')
    def test_scatterer_from_parameters_dict(self):
        sphere = Sphere(n=prior.Uniform(1, 2), r=prior.Uniform(0, 1))
        model = AlphaModel(sphere)
        pars = {'r': 0.8, 'n': 1.6}
        expected = Sphere(n=1.6, r=0.8)
        out_scatterer = model.scatterer_from_parameters(pars)
        self.assertEqual(out_scatterer, expected)

    @attr('fast')
    def test_scatterer_from_parameters_list(self):
        sphere = Sphere(n=prior.Uniform(1, 2), r=prior.Uniform(0, 1))
        model = AlphaModel(sphere)
        pars = [1.6, 0.8]
        expected = Sphere(n=1.6, r=0.8)
        self.assertEqual(model.scatterer_from_parameters(pars), expected)

    @attr('fast')
    def test_internal_scatterer_from_parameters_dict_fails(self):
        sphere = Sphere(n=prior.Uniform(1, 2), r=prior.Uniform(0, 1))
        model = AlphaModel(sphere)
        pars = {'r': 0.8, 'n': 1.6}
        self.assertRaises(KeyError, model._scatterer_from_parameters, pars)

    @attr('fast')
    def test_internal_scatterer_from_parameters_list(self):
        sphere = Sphere(n=prior.Uniform(1, 2), r=prior.Uniform(0, 1))
        model = AlphaModel(sphere)
        pars = [1.6, 0.8]
        expected = Sphere(n=1.6, r=0.8)
        self.assertEqual(model._scatterer_from_parameters(pars), expected)

    @attr('fast')
    def test_initial_guess(self):
        sphere = Sphere(n=prior.Uniform(1, 2),
                        r=prior.Uniform(0, 1, guess=0.8))
        model = AlphaModel(sphere)
        self.assertEqual(model.initial_guess, {'n': 1.5, 'r': 0.8})

    @attr('fast')
    def test_yaml_preserves_parameter_names(self):
        n = prior.ComplexPrior(prior.Uniform(1, 2), prior.Uniform(0, 0.1))
        sphere = Sphere(n=n, r=prior.Uniform(0, 1.5, name='radius'),
                        center=[1, 1, prior.Uniform(10, 20)])
        alpha = {'r': 0.6, 'g': prior.Uniform(0.6, 1.0)}
        model = AlphaModel(sphere, alpha=alpha)
        pre_names = model._parameter_names
        post_names = take_yaml_round_trip(model)._parameter_names
        self.assertEqual(pre_names, post_names)

    @attr('fast')
    def test_yaml_preserves_parameter_ties(self):
        tied = prior.Uniform(0, 1)
        sphere = Sphere(n=tied, r=prior.Uniform(0.6, 1, name='radius'),
                        center=[prior.Uniform(0.6, 1), tied, 10])
        alpha = {'r': 0.6, 'g': prior.Uniform(0.8, 0.9)}
        model = AlphaModel(sphere, alpha=alpha)
        model.add_tie(['radius', 'center.0'])
        post_model = take_yaml_round_trip(model)
        self.assertEqual(model.parameters, post_model.parameters)

    def test_yaml_preserves_parameter_names(self):
        sphere = Sphere(r=prior.Uniform(0, 1), n=prior.Uniform(1, 2, name='a'))
        model = AlphaModel(sphere)
        model._parameter_names = ['b', 'c']
        post_model = take_yaml_round_trip(model)
        self.assertEqual(post_model._parameter_names, ['b', 'c'])
        self.assertEqual(post_model._parameters[0].name, 'a')

    @attr('fast')
    def test_ensure_parameters_are_listlike(self):
        sphere = Sphere(r=prior.Uniform(0, 1), n=prior.Uniform(1, 2))
        model = AlphaModel(sphere, alpha=prior.Uniform(0.5, 1))
        as_dict = {'alpha': 0.8, 'r': 1.2, 'n': 1.5}
        as_list = [1.5, 1.2, 0.8]
        self.assertEqual(model.ensure_parameters_are_listlike(as_dict), as_list)
        self.assertEqual(model.ensure_parameters_are_listlike(as_list), as_list)


class TestParameterMapping(unittest.TestCase):
    @attr("fast")
    def test_map_value(self):
        model = SimpleModel()
        parameter = 14
        parameter_map = model._convert_to_map(parameter)
        expected = parameter
        self.assertEqual(parameter_map, expected)

    @attr("fast")
    def test_map_prior(self):
        model = SimpleModel()
        parameter = prior.Uniform(0, 1)
        position = len(model._parameters)
        parameter_map = model._convert_to_map(parameter, 'new name')
        expected = '_parameter_{}'.format(position)
        self.assertEqual(parameter_map, expected)

    @attr("fast")
    def test_mapping_adds_to_model(self):
        model = SimpleModel()
        parameter = prior.Uniform(0, 1)
        model._convert_to_map(parameter, 'new name')
        self.assertEqual(model._parameters[-1], parameter)
        self.assertEqual(model._parameter_names[-1], "new name")

    @attr("fast")
    def test_map_list(self):
        model = SimpleModel()
        parameter = [0, prior.Uniform(0, 1), prior.Uniform(2, 3)]
        position = len(model._parameters)
        parameter_map = model._convert_to_map(parameter)
        expected = [0, "_parameter_{}".format(position),
                    "_parameter_{}".format(position + 1)]
        self.assertEqual(parameter_map, expected)

    @attr("fast")
    def test_list_compound_name(self):
        model = SimpleModel()
        parameter = [0, prior.Uniform(0, 1), prior.Uniform(2, 3)]
        model._convert_to_map(parameter, 'prefix')
        self.assertEqual(model._parameter_names[-2], 'prefix.1')
        self.assertEqual(model._parameter_names[-1], 'prefix.2')

    @attr("fast")
    def test_map_dictionary(self):
        model = SimpleModel()
        parameter = {'a': 0, 'b': 1, 'c': prior.Uniform(0, 1)}
        position = len(model._parameters)
        parameter_map = model._convert_to_map(parameter)
        expected_placeholder = "_parameter_{}".format(position)
        expected = [dict, [[['a', 0], ['b', 1], ['c', expected_placeholder]]]]
        self.assertEqual(parameter_map, expected)

    @attr("fast")
    def test_map_dictionary_ignores_none(self):
        model = SimpleModel()
        parameter = {'a': 0, 'b': 1, 'c': None}
        parameter_map = model._convert_to_map(parameter)
        expected = [dict, [[['a', 0], ['b', 1]]]]
        self.assertEqual(parameter_map, expected)

    @attr("fast")
    def test_dict_compound_name(self):
        model = SimpleModel()
        parameter = {'a': 0, 'b': 1, 'c': prior.Uniform(0, 1)}
        model._convert_to_map(parameter, 'prefix')
        self.assertEqual(model._parameter_names[-1], 'prefix.c')

    @attr("fast")
    def test_map_xarray(self):
        model = SimpleModel()
        parameter = xr.DataArray(np.zeros((3, 3)),
                                 coords=[[10, 20, 30], ['a', 'b', 'c']],
                                 dims=('tens', 'letters'))
        parameter_map = model._convert_to_map(parameter)
        expected_1D = [make_xarray, ['letters', ['a', 'b', 'c'], [0, 0, 0]]]
        expected = [make_xarray, ['tens', [10, 20, 30],
                                  [expected_1D, expected_1D, expected_1D]]]
        self.assertEqual(parameter_map, expected)

    @attr("fast")
    def test_xarray_compound_name(self):
        model = SimpleModel()
        parameter = xr.DataArray(np.zeros((3, 3)),
                                 coords=[[10, 20, 30], ['a', 'b', 'c']],
                                 dims=('tens', 'letters')).astype('object')
        parameter[-1, -1] = prior.Uniform(0, 1)
        model._convert_to_map(parameter, 'prefix')
        self.assertEqual(model._parameter_names[-1], 'prefix.30.c')

    @attr("fast")
    def test_map_complex(self):
        model = SimpleModel()
        parameter = prior.ComplexPrior(1, prior.Uniform(2, 3))
        position = len(model._parameters)
        parameter_map = model._convert_to_map(parameter)
        expected = [make_complex, [1, "_parameter_{}".format(position)]]
        self.assertEqual(parameter_map, expected)

    @attr("fast")
    def test_complex_compound_name(self):
        model = SimpleModel()
        parameter = prior.ComplexPrior(prior.Uniform(0, 1),
                                       prior.Uniform(2, 3))
        model._convert_to_map(parameter, 'prefix')
        self.assertEqual(model._parameter_names[-2], 'prefix.real')
        self.assertEqual(model._parameter_names[-1], 'prefix.imag')

    @attr("fast")
    def test_map_composite_object(self):
        model = SimpleModel()
        parameter = [prior.ComplexPrior(0, 1), {'a': 2, 'b': [4, 5]}, 6]
        parameter_map = model._convert_to_map(parameter)
        expected = [[make_complex, [0, 1]],
                    [dict, [[['a', 2], ['b', [4, 5]]]]], 6]
        self.assertEqual(parameter_map, expected)

    @attr("fast")
    def test_read_func_map(self):
        parameter_map = [dict, [[['a', 0], ['b', 1], ['c', 2]]]]
        expected = {'a': 0, 'b': 1, 'c': 2}
        self.assertEqual(read_map(parameter_map, []), expected)

    @attr("fast")
    def test_read_placeholder_map(self):
        parameter_map = [0, 1, "_parameter_2"]
        placeholders = [3, 4, 5]
        expected = [0, 1, 5]
        self.assertEqual(read_map(parameter_map, placeholders), expected)

    @attr("fast")
    def test_read_complex_map_values(self):
        parameter_map = [make_complex, ['_parameter_0', '_parameter_1']]
        values = [0, 1]
        self.assertEqual(read_map(parameter_map, values), complex(0, 1))

    @attr("fast")
    def test_read_complex_map_priors(self):
        parameter_map = [make_complex, ['_parameter_0', '_parameter_1']]
        priors = [prior.Uniform(0, 1), prior.Uniform(1, 2)]
        expected = prior.ComplexPrior(priors[0], priors[1])
        self.assertEqual(read_map(parameter_map, priors), expected)

    @attr("fast")
    def test_read_composite_map(self):
        n_map = [dict, [[['red', [[make_complex, [1.5, "_parameter_2"]],
                                  [make_complex, [1.6, "_parameter_3"]]]],
                         ['green', [[make_complex, [1.7, "_parameter_4"]],
                                    [make_complex, [1.8, "_parameter_5"]]]]]]]
        parameter_map = [dict, [[['r', ["_parameter_0", "_parameter_1"]],
                                 ['n', n_map],
                                 ['center', [10, 20, "_parameter_6"]]]]]
        placeholders = [0.5, 0.7, 0.01, 0.02, 0.03, 0.04, 30]
        n_expected = {'red': [complex(1.5, 0.01), complex(1.6, 0.02)],
                      'green': [complex(1.7, 0.03), complex(1.8, 0.04)]}
        expected = {'r': [0.5, 0.7], 'n': n_expected, 'center': [10, 20, 30]}
        self.assertEqual(read_map(parameter_map, placeholders), expected)

    @attr("fast")
    def test_make_xarray_1D(self):
        values = [1, 2, 3, 4, 5]
        coords = [10, 20, 30, 40, 50]
        dims = 'dimname'
        constructed = make_xarray(dims, coords, values)
        expected = xr.DataArray(values, coords=[coords], dims=[dims])
        xr.testing.assert_equal(constructed, expected)

    @attr("fast")
    def test_make_xarray_slices(self):
        shared_coords = [['a', 'b', 'c'], [10, 20, 30]]
        shared_dims = ['letters', 'tens']
        slice1 = xr.DataArray(np.ones((3, 3)), shared_coords, shared_dims)
        slice2 = xr.DataArray(np.zeros((3, 3)), shared_coords, shared_dims)
        new_coords = ['ones', 'zeros']
        join_dim = xr.DataArray(new_coords, dims=['new'], name='new')
        constructed = make_xarray('new', new_coords, [slice1, slice2])
        expected = xr.concat([slice1, slice2], dim=join_dim)
        xr.testing.assert_equal(constructed, expected)


class TestParameterTying(unittest.TestCase):
    @attr('fast')
    def test_parameters_list(self):
        tied = prior.Uniform(0, 1)
        scatterer = Sphere(n=tied, r=prior.Uniform(0.5, 1.5),
                           center=[tied, 10, prior.Uniform(0, 10)])
        model = AlphaModel(scatterer)
        expected = [prior.Uniform(0, 1),
                    prior.Uniform(0.5, 1.5),
                    prior.Uniform(0, 10)]
        self.assertEqual(model._parameters, expected)

    @attr('fast')
    def test_parameters_names(self):
        tied = prior.Uniform(0, 1)
        scatterer = Sphere(n=tied, r=prior.Uniform(0.5, 1.5),
                           center=[tied, 10, prior.Uniform(0, 10)])
        model = AlphaModel(scatterer)
        expected = ['n', 'r', 'center.2']
        self.assertEqual(model._parameter_names, expected)

    @attr('fast')
    def test_parameters_map(self):
        tied = prior.Uniform(0, 1)
        scatterer = Sphere(n=tied, r=prior.Uniform(0.5, 1.5),
                           center=[tied, 10, prior.Uniform(0, 10)])
        model = AlphaModel(scatterer)
        expected = [dict, [[['n', '_parameter_0'], ['r', '_parameter_1'],
                            ['center', ['_parameter_0', 10, '_parameter_2']]]]]
        self.assertEqual(model._maps['scatterer'], expected)

    @attr('fast')
    def test_equal_not_identical_do_not_tie(self):
        scatterer = Sphere(n=prior.Uniform(1, 2), r=prior.Uniform(1, 2),
                           center=[10, 10, prior.Uniform(1, 2)])
        model = AlphaModel(scatterer)
        expected_priors = [prior.Uniform(1, 2),
                           prior.Uniform(1, 2),
                           prior.Uniform(1, 2)]
        expected_names = ['n', 'r', 'center.2']
        self.assertEqual(model._parameters, expected_priors)
        self.assertEqual(model._parameter_names, expected_names)

    @attr('fast')
    def test_tied_name(self):
        tied = prior.Uniform(0, 1)
        sphere1 = Sphere(n=prior.Uniform(1, 2), r=tied, center=[1, 1, 1])
        sphere2 = Sphere(n=prior.Uniform(1, 2), r=tied, center=[1, 1, 1])
        model = AlphaModel(Spheres([sphere1, sphere2]))
        expected_names = ['0:n', 'r', '1:n']
        self.assertEqual(model._parameter_names, expected_names)

    @attr('fast')
    def test_prior_name(self):
        tied = prior.Uniform(-5, 5, name='xy')
        sphere = Sphere(n=prior.Uniform(1, 2, name='index'), r=0.5,
                        center=[tied, tied, prior.Uniform(0, 10, name='z')])
        model = AlphaModel(sphere)
        expected_names = ['index', tied.name, 'z']
        self.assertEqual(model._parameter_names, expected_names)

    @attr('fast')
    def test_duplicate_name(self):
        tied = prior.Uniform(-5, 5, name='dummy')
        sphere = Sphere(n=prior.Uniform(1, 2, name='dummy'), r=0.5,
                        center=[tied, tied, prior.Uniform(0, 10, name='z')])
        model = AlphaModel(sphere)
        expected = ['dummy', 'dummy_0', 'z']
        self.assertEqual(model._parameter_names, expected)

    @attr('fast')
    def test_triplicate_name(self):
        tied = prior.Uniform(-5, 5, name='dummy')
        sphere = Sphere(n=prior.Uniform(1, 2, name='dummy'),
                        r=prior.Uniform(1, 2, name='dummy'),
                        center=[tied, tied, prior.Uniform(0, 10, name='z')])
        model = AlphaModel(sphere)
        expected = ['dummy', 'dummy_0', 'dummy_1', 'z']
        self.assertEqual(model._parameter_names, expected)

    @attr('fast')
    def test_add_missing_tie_fails(self):
        sphere = Sphere(n=prior.Uniform(1, 2), r=0.5, center=[10, 10, 10])
        model = AlphaModel(sphere)
        self.assertRaises(ValueError, model.add_tie, ['r', 'n'])

    @attr('fast')
    def test_add_unequal_tie_fails(self):
        sphere = Sphere(n=prior.Uniform(1, 2), r=prior.Uniform(0, 1),
                        center=[10, 10, 10])
        model = AlphaModel(sphere)
        self.assertRaises(ValueError, model.add_tie, ['r', 'n'])

    @attr('fast')
    def test_add_tie_updates_parameters(self):
        tied = prior.Uniform(-5, 5)
        sphere = Sphere(n=prior.Uniform(1, 2), r=prior.Uniform(1, 2),
                        center=[tied, tied, 10])
        model = AlphaModel(sphere)
        model.add_tie(['r', 'n'])
        expected = [prior.Uniform(1, 2), prior.Uniform(-5, 5)]
        self.assertEqual(model._parameters, expected)

    @attr('fast')
    def test_add_tie_updates_parameter_names(self):
        tied = prior.Uniform(-5, 5)
        sphere = Sphere(n=prior.Uniform(1, 2), r=prior.Uniform(1, 2),
                        center=[tied, tied, 10])
        model = AlphaModel(sphere)
        model.add_tie(['r', 'n'])
        expected = ['n', 'center.0']
        self.assertEqual(model._parameter_names, expected)

    @attr('fast')
    def test_add_tie_updates_map(self):
        tied = prior.Uniform(-5, 5)
        sphere = Sphere(n=prior.Uniform(1, 2), r=prior.Uniform(1, 2),
                        center=[tied, tied, 10])
        model = AlphaModel(sphere)
        model.add_tie(['r', 'n'])
        expected = [dict, [[['n', '_parameter_0'], ['r', '_parameter_0'],
                            ['center', ['_parameter_1', '_parameter_1', 10]]]]]
        self.assertEqual(model._maps['scatterer'], expected)

    @attr('fast')
    def test_add_tie_specify_name(self):
        tied = prior.Uniform(-5, 5)
        sphere = Sphere(n=prior.Uniform(1, 2), r=prior.Uniform(1, 2),
                        center=[tied, tied, 10])
        model = AlphaModel(sphere)
        model.add_tie(['r', 'n'], new_name='dummy')
        expected = ['dummy', 'center.0']
        self.assertEqual(model._parameter_names, expected)

    @attr('fast')
    def test_add_3_way_tie(self):
        tied = prior.Uniform(-5, 5)
        n = prior.ComplexPrior(prior.Uniform(1, 2), prior.Uniform(0, 1))
        sphere = Sphere(n=n, r=prior.Uniform(0.5, 1),
                        center=[prior.Uniform(0, 1), prior.Uniform(0, 1),
                                prior.Uniform(0, 10)])
        model = AlphaModel(sphere)
        model.add_tie(['center.0', 'n.imag', 'center.1'])
        expected_map = [
            dict,
            [[['n', [make_complex, ['_parameter_0', '_parameter_1']]],
              ['r', '_parameter_2'],
              ['center', ['_parameter_1', '_parameter_1', '_parameter_3']]]]]
        expected_parameters = [prior.Uniform(1, 2), prior.Uniform(0, 1),
                               prior.Uniform(0.5, 1), prior.Uniform(0, 10)]
        expected_names = ['n.real', 'n.imag', 'r', 'center.2']
        self.assertEqual(model._maps['scatterer'], expected_map)
        self.assertEqual(model._parameters, expected_parameters)
        self.assertEqual(model._parameter_names, expected_names)


class TestFindOptics(unittest.TestCase):
    @attr('fast')
    def test_reads_noise_map(self):
        noise = {'red': 0.5, 'green': prior.Uniform(0, 1)}
        model = AlphaModel(Sphere(), noise_sd=noise)
        found_noise = model._find_noise([0.7], None)
        self.assertEqual(found_noise, {'red': 0.5, 'green': 0.7})

    @attr('fast')
    def test_noise_from_schema(self):
        model = AlphaModel(Sphere(), noise_sd=None)
        schema = detector_grid(2, 2)
        schema.attrs['noise_sd'] = 0.5
        found_noise = model._find_noise([], schema)
        self.assertEqual(found_noise, 0.5)

    @attr('fast')
    def test_model_noise_takes_precedence(self):
        model = AlphaModel(Sphere(), noise_sd=0.8)
        schema = detector_grid(2, 2)
        schema.attrs['noise_sd'] = 0.5
        found_noise = model._find_noise([], schema)
        self.assertEqual(found_noise, 0.8)

    @attr('fast')
    def test_no_noise_if_all_uniform(self):
        sphere = Sphere(r=prior.Uniform(0, 1), n=prior.Uniform(1, 2))
        model = AlphaModel(sphere)
        schema = detector_grid(2, 2)
        found_noise = model._find_noise([0.5, 0.5], schema)
        self.assertEqual(found_noise, 1)

    @attr('fast')
    def test_require_noise_if_nonuniform(self):
        sphere = Sphere(r=prior.Gaussian(0, 1), n=prior.Uniform(1, 2))
        model = AlphaModel(sphere)
        schema = detector_grid(2, 2)
        pars = [0.5, 0.5]
        self.assertRaises(MissingParameter, model._find_noise, pars, schema)

    @attr('fast')
    def test_reads_optics_from_map(self):
        med_n = prior.ComplexPrior(1.5, prior.Uniform(0, 0.1))
        wl = {'red': 0.5, 'green': prior.Uniform(0, 1)}
        pol = [1, prior.Uniform(0.5, 1.5)]
        model = AlphaModel(Sphere(), medium_index=med_n,
                           illum_wavelen=wl, illum_polarization=pol)
        pars = [0.01, 0.6, 1]
        found_optics = model._find_optics(pars, None)
        expected = {'medium_index': complex(1.5, 0.01),
                    'illum_wavelen': {'red': 0.5, 'green': 0.6},
                    'illum_polarization': [1, 1]}
        self.assertEqual(found_optics, expected)

    @attr('fast')
    def test_optics_from_schema(self):
        model = AlphaModel(Sphere(), medium_index=prior.Uniform(1, 2))
        schema = detector_grid(2, 2)
        schema.attrs['illum_wavelen'] = 0.6
        schema.attrs['illum_polarization'] = [1, 0]
        found_optics = model._find_optics([1.5], schema)
        expected = {'medium_index': 1.5, 'illum_wavelen': 0.6,
                    'illum_polarization': [1, 0]}
        self.assertEqual(found_optics, expected)

    @attr('fast')
    def test_model_optics_take_precedence(self):
        model = AlphaModel(Sphere(), medium_index=1.5, illum_wavelen=0.8)
        schema = detector_grid(2, 2)
        schema.attrs['illum_wavelen'] = 0.6
        schema.attrs['illum_polarization'] = [1, 0]
        found_optics = model._find_optics([], schema)
        expected = {'medium_index': 1.5, 'illum_wavelen': 0.8,
                    'illum_polarization': [1, 0]}
        self.assertEqual(found_optics, expected)

    @attr('fast')
    def test_missing_optics(self):
        model = AlphaModel(Sphere(), medium_index=1.5, illum_wavelen=0.8)
        schema = detector_grid(2, 2)
        schema.attrs['illum_wavelen'] = 0.6
        self.assertRaises(MissingParameter, model._find_optics, [], schema)


class TestAlphaModel(unittest.TestCase):
    @attr('fast')
    def test_initializable(self):
        scatterer = make_sphere()
        model = AlphaModel(scatterer, alpha=0.6)
        self.assertTrue(model is not None)

    @attr("fast")
    def test_yaml_round_trip_with_xarray(self):
        alpha_xarray = xr.DataArray(
            [1, 0.5],
            dims=['illumination'],
            coords={'illumination': ['red', 'green']})
        sphere = make_sphere()
        model = AlphaModel(sphere, alpha=alpha_xarray)
        reloaded = take_yaml_round_trip(model)
        self.assertEqual(reloaded, model)

    @attr("fast")
    def test_yaml_round_trip_with_dict(self):
        alpha_dict = {'red': 1, 'green': 0.5}
        sphere = make_sphere()
        model = AlphaModel(sphere, alpha=alpha_dict)
        reloaded = take_yaml_round_trip(model)
        self.assertEqual(reloaded, model)


class TestPerfectLensModel(unittest.TestCase):
    @attr('fast')
    def test_initializable(self):
        scatterer = make_sphere()
        model = PerfectLensModel(scatterer, lens_angle=0.6)
        self.assertTrue(model is not None)

    @attr('fast')
    def test_accepts_lens_angle_as_prior(self):
        scatterer = make_sphere()
        lens_angle = prior.Uniform(0, 1.0)
        model = PerfectLensModel(scatterer, lens_angle=lens_angle)
        self.assertIsInstance(model.lens_angle, prior.Prior)

    @attr('fast')
    def test_accepts_alpha_as_prior(self):
        scatterer = make_sphere()
        lens_angle = prior.Uniform(0, 1.0)
        alpha = prior.Uniform(0, 1.0)
        model = PerfectLensModel(scatterer, alpha=alpha, lens_angle=lens_angle)
        self.assertIsInstance(model.alpha, prior.Prior)

    @attr('fast')
    def test_forward_uses_alpha(self):
        model = PerfectLensModel(
            SPHERE_IN_METERS,
            alpha=prior.Uniform(0, 1.0),
            lens_angle=prior.Uniform(0, 1.0))
        pars_common = {
            'lens_angle': 0.3,
            'n': 1.5,
            'r': 0.5e-6,
            }
        pars_alpha0 = pars_common.copy()
        pars_alpha0.update({'alpha': 0})
        alpha0 = model.forward(pars_alpha0, xschema_lens)
        self.assertLess(alpha0.values.std(), 1e-6)


def make_sphere():
    index = prior.Uniform(1.4, 1.6, name='n')
    radius = prior.Uniform(0.2, 0.8, name='r')
    return Sphere(n=index, r=radius)


def make_model_kwargs():
    kwargs = {
        'noise_sd': 0.05,
        'medium_index': 1.33,
        'illum_wavelen': 0.66,
        'illum_polarization': (1, 0),
        'theory': 'auto',
        # constraints?
        # FIXME need to test alpha, lens_angle for other models
        }
    return kwargs


def take_yaml_round_trip(model):
    object_string = yaml.dump(model)
    loaded = yaml.load(object_string, Loader=holopy_object.FullLoader)
    return loaded


@attr('fast')
def test_io():
    model = ExactModel(Sphere(1), calc_holo)
    assert_read_matches_write(model)

    model = ExactModel(Sphere(1), calc_holo, theory=Mie(False))
    assert_read_matches_write(model)


if __name__ == '__main__':
    unittest.main()

