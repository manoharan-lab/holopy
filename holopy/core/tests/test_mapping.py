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

import numpy as np
import xarray as xr

from nose.plugins.attrib import attr

from holopy.scattering import Sphere, Spheres
from holopy.core import prior
from holopy.core.mapping import (Mapper, transformed_prior,
                                 make_xarray, read_map)


class TestParameterMapping(unittest.TestCase):
    @attr("fast")
    def test_map_value(self):
        mapper = Mapper()
        parameter = 14
        parameter_map = mapper.convert_to_map(parameter)
        expected = parameter
        self.assertEqual(parameter_map, expected)

    @attr("fast")
    def test_map_prior(self):
        mapper = Mapper()
        parameter = prior.Uniform(0, 1)
        position = len(mapper.parameters)
        parameter_map = mapper.convert_to_map(parameter, 'new name')
        expected = '_parameter_{}'.format(position)
        self.assertEqual(parameter_map, expected)

    @attr("fast")
    def test_mapping_adds_to_mapper(self):
        mapper = Mapper()
        parameter = prior.Uniform(0, 1)
        mapper.convert_to_map(parameter, 'new name')
        self.assertEqual(mapper.parameters[-1], parameter)
        self.assertEqual(mapper.parameter_names[-1], "new name")

    @attr("fast")
    def test_map_list(self):
        mapper = Mapper()
        parameter = [0, prior.Uniform(0, 1), prior.Uniform(2, 3)]
        position = len(mapper.parameters)
        parameter_map = mapper.convert_to_map(parameter)
        expected = [0, "_parameter_{}".format(position),
                    "_parameter_{}".format(position + 1)]
        self.assertEqual(parameter_map, expected)

    @attr("fast")
    def test_list_compound_name(self):
        mapper = Mapper()
        parameter = [0, prior.Uniform(0, 1), prior.Uniform(2, 3)]
        mapper.convert_to_map(parameter, 'prefix')
        self.assertEqual(mapper.parameter_names[-2], 'prefix.1')
        self.assertEqual(mapper.parameter_names[-1], 'prefix.2')

    @attr("fast")
    def test_map_dictionary(self):
        mapper = Mapper()
        parameter = {'a': 0, 'b': 1, 'c': prior.Uniform(0, 1)}
        position = len(mapper.parameters)
        parameter_map = mapper.convert_to_map(parameter)
        expected_placeholder = "_parameter_{}".format(position)
        expected = [dict, [[['a', 0], ['b', 1], ['c', expected_placeholder]]]]
        self.assertEqual(parameter_map, expected)

    @attr("fast")
    def test_map_dictionary_ignores_none(self):
        mapper = Mapper()
        parameter = {'a': 0, 'b': 1, 'c': None}
        parameter_map = mapper.convert_to_map(parameter)
        expected = [dict, [[['a', 0], ['b', 1]]]]
        self.assertEqual(parameter_map, expected)

    @attr("fast")
    def test_dict_compound_name(self):
        mapper = Mapper()
        parameter = {'a': 0, 'b': 1, 'c': prior.Uniform(0, 1)}
        mapper.convert_to_map(parameter, 'prefix')
        self.assertEqual(mapper.parameter_names[-1], 'prefix.c')

    @attr("fast")
    def test_map_xarray(self):
        mapper = Mapper()
        parameter = xr.DataArray(np.zeros((3, 3)),
                                 coords=[[10, 20, 30], ['a', 'b', 'c']],
                                 dims=('tens', 'letters'))
        parameter_map = mapper.convert_to_map(parameter)
        expected_1D = [make_xarray, ['letters', ['a', 'b', 'c'], [0, 0, 0]]]
        expected = [make_xarray, ['tens', [10, 20, 30],
                                  [expected_1D, expected_1D, expected_1D]]]
        self.assertEqual(parameter_map, expected)

    @attr("fast")
    def test_xarray_compound_name(self):
        mapper = Mapper()
        parameter = xr.DataArray(np.zeros((3, 3)),
                                 coords=[[10, 20, 30], ['a', 'b', 'c']],
                                 dims=('tens', 'letters')).astype('object')
        parameter[-1, -1] = prior.Uniform(0, 1)
        mapper.convert_to_map(parameter, 'prefix')
        self.assertEqual(mapper.parameter_names[-1], 'prefix.30.c')

    @attr("fast")
    def test_map_complex(self):
        mapper = Mapper()
        parameter = prior.ComplexPrior(1, prior.Uniform(2, 3))
        position = len(mapper.parameters)
        parameter_map = mapper.convert_to_map(parameter)
        placeholder = "_parameter_{}".format(position)
        expected = [transformed_prior, [complex, [1, placeholder]]]
        self.assertEqual(parameter_map, expected)

    @attr("fast")
    def test_complex_compound_name(self):
        mapper = Mapper()
        parameter = prior.ComplexPrior(prior.Uniform(0, 1),
                                       prior.Uniform(2, 3))
        mapper.convert_to_map(parameter, 'prefix')
        self.assertEqual(mapper.parameter_names[-2], 'prefix.real')
        self.assertEqual(mapper.parameter_names[-1], 'prefix.imag')

    @attr('fast')
    def test_map_transformed_prior(self):
        mapper = Mapper()
        transformed = prior.TransformedPrior(np.sqrt, prior.Uniform(0, 2),
                                             name='sqrt')
        position = len(mapper.parameters)
        parameter_map = mapper.convert_to_map(transformed)
        placeholder = "_parameter_{}".format(position)
        expected = [transformed_prior, [np.sqrt, [placeholder]]]
        self.assertEqual(parameter_map, expected)

    @attr('fast')
    def test_map_transformed_prior_names(self):
        mapper = Mapper()
        base_prior = [prior.Uniform(0, 2, name='first'), prior.Uniform(1, 2)]
        transformed = {'trans': prior.TransformedPrior(np.maximum, base_prior)}
        parameter_map = mapper.convert_to_map(transformed)
        self.assertEqual(mapper.parameter_names[-2:], ['first', 'trans.1'])

    @attr('fast')
    def test_named_transformed_prior(self):
        mapper = Mapper()
        base_prior = [prior.Uniform(0, 2, name='first'), prior.Uniform(1, 2)]
        transform = prior.TransformedPrior(np.maximum, base_prior, name='real')
        transform = {'fake': transform}
        parameter_map = mapper.convert_to_map(transform)
        self.assertEqual(mapper.parameter_names[-2:], ['first', 'real.1'])

    @attr('fast')
    def test_map_hierarchical_transformed_prior(self):
        mapper = Mapper()
        inner = prior.TransformedPrior(np.sqrt, prior.Uniform(0, 2))
        full = prior.TransformedPrior(np.maximum, [inner, prior.Uniform(0, 1)])
        position = len(mapper.parameters)
        parameter_map = mapper.convert_to_map(full)
        placeholder = ['_parameter_{}'.format(position + i) for i in range(2)]
        submap = [transformed_prior, [np.sqrt, [placeholder[0]]]]
        expected = [transformed_prior, [np.maximum, [submap, placeholder[1]]]]
        self.assertEqual(parameter_map, expected)

    @attr("fast")
    def test_map_composite_object(self):
        mapper = Mapper()
        parameter = [prior.ComplexPrior(0, 1), {'a': 2, 'b': [4, 5]}, 6]
        parameter_map = mapper.convert_to_map(parameter)
        expected = [[transformed_prior, [complex, [0, 1]]],
                    [dict, [[['a', 2], ['b', [4, 5]]]]], 6]
        self.assertEqual(parameter_map, expected)


class TestUnmapping(unittest.TestCase):
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
        parameter_map = [transformed_prior, [complex, ['_parameter_0',
                                                       '_parameter_1']]]
        values = [0, 1]
        self.assertEqual(read_map(parameter_map, values), complex(0, 1))

    @attr("fast")
    def test_read_complex_map_priors(self):
        parameter_map = [transformed_prior, [complex, ['_parameter_0',
                                                       '_parameter_1']]]
        priors = [prior.Uniform(0, 1), prior.Uniform(1, 2)]
        expected = prior.TransformedPrior(complex, [priors[0], priors[1]])
        self.assertEqual(read_map(parameter_map, priors), expected)

    @attr('fast')
    def test_read_transformed_prior_map_values(self):
        parameter_map = [transformed_prior, [np.sqrt, ['_parameter_0']]]
        values = [4]
        self.assertEqual(read_map(parameter_map, values), 2)

    @attr('fast')
    def test_read_transformed_prior_map_priors(self):
        parameter_map = [transformed_prior, [np.sqrt, ['_parameter_0']]]
        priors = [prior.Uniform(0, 1)]
        expected = prior.TransformedPrior(np.sqrt, priors)
        self.assertEqual(read_map(parameter_map, priors), expected)

    @attr('fast')
    def test_read_hierarchical_transformed(self):
        inner_map = [transformed_prior, [np.sqrt, ['_parameter_0']]]
        parameter_map = [transformed_prior, [np.maximum, ['_parameter_1',
                                                          inner_map]]]
        priors = [prior.Uniform(0, 1), prior.Uniform(1, 2)]
        expected_base = [prior.Uniform(1, 2),
                         prior.TransformedPrior(np.sqrt, prior.Uniform(0, 1))]
        expected_full = prior.TransformedPrior(np.maximum, expected_base)
        self.assertEqual(read_map(parameter_map, priors), expected_full)
        values = [25, 7]
        self.assertEqual(read_map(inner_map, values), 5)
        self.assertEqual(read_map(parameter_map, values), 7)

    @attr("fast")
    def test_read_composite_map(self):
        n_map = [
            dict,
            [[['red', [[transformed_prior, [complex, [1.5, "_parameter_2"]]],
                       [transformed_prior, [complex, [1.6, "_parameter_3"]]]]],
              ['green', [[transformed_prior, [complex, [1.7, "_parameter_4"]]],
                         [transformed_prior, [complex, [1.8, "_parameter_5"]]]
                         ]]]]]
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
        mapper = Mapper()
        parameter_map = mapper.convert_to_map(scatterer.parameters)
        expected = [prior.Uniform(0, 1),
                    prior.Uniform(0.5, 1.5),
                    prior.Uniform(0, 10)]
        self.assertEqual(mapper.parameters, expected)

    @attr('fast')
    def test_parameters_names(self):
        tied = prior.Uniform(0, 1)
        scatterer = Sphere(n=tied, r=prior.Uniform(0.5, 1.5),
                           center=[tied, 10, prior.Uniform(0, 10)])
        mapper = Mapper()
        parameter_map = mapper.convert_to_map(scatterer.parameters)
        expected = ['n', 'r', 'center.2']
        self.assertEqual(mapper.parameter_names, expected)

    @attr('fast')
    def test_parameters_map(self):
        tied = prior.Uniform(0, 1)
        scatterer = Sphere(n=tied, r=prior.Uniform(0.5, 1.5),
                           center=[tied, 10, prior.Uniform(0, 10)])
        mapper = Mapper()
        parameter_map = mapper.convert_to_map(scatterer.parameters)
        expected = [dict, [[['n', '_parameter_0'], ['r', '_parameter_1'],
                            ['center', ['_parameter_0', 10, '_parameter_2']]]]]
        self.assertEqual(parameter_map, expected)

    @attr('fast')
    def test_equal_not_identical_do_not_tie(self):
        scatterer = Sphere(n=prior.Uniform(1, 2), r=prior.Uniform(1, 2),
                           center=[10, 10, prior.Uniform(1, 2)])
        mapper = Mapper()
        parameter_amp = mapper.convert_to_map(scatterer.parameters)
        expected_priors = [prior.Uniform(1, 2),
                           prior.Uniform(1, 2),
                           prior.Uniform(1, 2)]
        expected_names = ['n', 'r', 'center.2']
        self.assertEqual(mapper.parameters, expected_priors)
        self.assertEqual(mapper.parameter_names, expected_names)

    @attr('fast')
    def test_transformed_priors_are_tied(self):
        base_prior = prior.Uniform(0, 2, name='x')
        transformed = prior.TransformedPrior(np.sqrt, base_prior, name='y')
        scatterer = Sphere(n=1.5, r=0.5, center=[base_prior, transformed,
                                                 prior.Uniform(5, 10)])
        mapper = Mapper()
        parameter_map = mapper.convert_to_map(scatterer.parameters)
        expected_priors = [base_prior, prior.Uniform(5, 10)]
        expected_names = ['x', 'center.2']
        self.assertEqual(mapper.parameters, expected_priors)
        self.assertEqual(mapper.parameter_names, expected_names)

    @attr('fast')
    def test_tied_name(self):
        tied = prior.Uniform(0, 1)
        s0 = Sphere(n=prior.Uniform(1, 2), r=tied, center=[1, 1, 1])
        s1 = Sphere(n=prior.Uniform(1, 2), r=tied, center=[1, 1, 1])
        mapper = Mapper()
        parameter_map = mapper.convert_to_map(Spheres([s0, s1]).parameters)
        expected_names = ['0:n', 'r', '1:n']
        self.assertEqual(mapper.parameter_names, expected_names)

    @attr('fast')
    def test_no_tied_name_if_not_shared_between_scatterers(self):
        s0_r = prior.Gaussian(0.5, 0.1)
        s1_r = prior.Gaussian(0.5, 0.1)
        s0 = Sphere(r=s0_r, n=1.5, center=[0, 3, 4])
        s1 = Sphere(r=s1_r, n=1.5, center=[s0_r + s1_r, 3, 4])
        mapper = Mapper()
        parameter_map = mapper.convert_to_map(Spheres([s0, s1]).parameters)
        expected_names = ['0:r', '1:r']
        self.assertEqual(mapper.parameter_names, expected_names)

    @attr('fast')
    def test_prior_name(self):
        tied = prior.Uniform(-5, 5, name='xy')
        sphere = Sphere(n=prior.Uniform(1, 2, name='index'), r=0.5,
                        center=[tied, tied, prior.Uniform(0, 10, name='z')])
        mapper = Mapper()
        parameter_map = mapper.convert_to_map(sphere.parameters)
        expected_names = ['index', tied.name, 'z']
        self.assertEqual(mapper.parameter_names, expected_names)

    @attr('fast')
    def test_duplicate_name(self):
        tied = prior.Uniform(-5, 5, name='dummy')
        sphere = Sphere(n=prior.Uniform(1, 2, name='dummy'), r=0.5,
                        center=[tied, tied, prior.Uniform(0, 10, name='z')])
        mapper = Mapper()
        parameter_map = mapper.convert_to_map(sphere.parameters)
        expected = ['dummy', 'dummy_0', 'z']
        self.assertEqual(mapper.parameter_names, expected)

    @attr('fast')
    def test_triplicate_name(self):
        tied = prior.Uniform(-5, 5, name='dummy')
        sphere = Sphere(n=prior.Uniform(1, 2, name='dummy'),
                        r=prior.Uniform(1, 2, name='dummy'),
                        center=[tied, tied, prior.Uniform(0, 10, name='z')])
        mapper = Mapper()
        parameter_map = mapper.convert_to_map(sphere.parameters)
        expected = ['dummy', 'dummy_0', 'dummy_1', 'z']
        self.assertEqual(mapper.parameter_names, expected)

    @attr('fast')
    def test_ties_on_separate_convert_to_map_calls(self):
        tied = prior.Uniform(-5, 5, name='to_tie')
        parameters1 = {'tie': tied, 'dummy1': 3, 'dummy2': prior.Uniform(0, 1)}
        parameters2 = {'tie': tied, 'dummy1': 3, 'dummy2': prior.Uniform(0, 1)}
        mapper = Mapper()
        map1 = mapper.convert_to_map(parameters1)
        map2 = mapper.convert_to_map(parameters2)
        s0 = Sphere(n=prior.Uniform(1, 2, name='dummy'),
                    r=prior.Uniform(1, 2, name='dummy'),
                    center=[tied, tied, prior.Uniform(0, 10, name='z')])
        expected_priors = [tied, prior.Uniform(0, 1), prior.Uniform(0, 1)]
        expected_names = ['to_tie', 'dummy2', 'dummy2_0']
        self.assertEqual(mapper.parameters, expected_priors)
        self.assertEqual(mapper.parameter_names, expected_names)


if __name__ == '__main__':
    unittest.main()
