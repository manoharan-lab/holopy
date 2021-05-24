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
import itertools

import yaml
import numpy as np
import xarray as xr

from nose.plugins.attrib import attr

from holopy.core import detector_grid, holopy_object
from holopy.scattering import (
    Sphere, Spheres, Mie, MieLens, AberratedMieLens, calc_holo, Multisphere)
from holopy.scattering.theory.scatteringtheory import ScatteringTheory
from holopy.scattering.errors import MissingParameter
from holopy.core.tests.common import assert_read_matches_write
from holopy.inference import prior, AlphaModel, ExactModel
from holopy.inference.model import Model
from holopy.core.mapping import transformed_prior
from holopy.scattering.tests.common import (
    xschema_lens, sphere as SPHERE_IN_METERS)


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
    def test_yaml_round_trip_with_xarray_in_scatterer(self):
        n = xr.DataArray([prior.Gaussian(1.5, 0.1), prior.Gaussian(1.7, 0.1)],
                         dims=['illumination'],
                         coords={'illumination': ['red', 'green']})
        sphere = Sphere(r=0.5, n=n, center=(5, 5, 5))
        model = AlphaModel(sphere)
        reloaded = take_yaml_round_trip(model)
        self.assertEqual(reloaded, model)

    @attr('fast')
    def test_scatterer_is_parameterized(self):
        sphere = Sphere(n=prior.Uniform(1, 2), r=prior.Uniform(0, 1))
        model = AlphaModel(sphere)
        self.assertEqual(model.scatterer, sphere)

    @attr('fast')
    def test_scatterers_maintain_attrs(self):
        spheres = Spheres([
            Sphere(center=(1, 2, 3), r=0.4, n=1.59),
            Sphere(center=(4, 5, 6), r=0.4, n=1.59)],
            warn=False)
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
    def test_theory_from_parameters(self):
        sphere = Sphere(n=prior.Uniform(1, 2), r=prior.Uniform(0, 1))
        theory_in = MieLens(lens_angle=prior.Uniform(0, 1.0))
        model = Model(sphere, theory=theory_in)

        np.random.seed(1021)
        lens_angle = np.random.rand()
        pars = {'lens_angle': lens_angle, 'n': 1.59, 'r': 0.5}
        theory_out = model.theory_from_parameters(pars)

        self.assertIsInstance(theory_out, theory_in.__class__)
        self.assertEqual(theory_out.lens_angle, lens_angle)

    @attr('fast')
    def test_theory_from_parameters_respects_nonfittable_options_mie(self):
        # tests for other theories are done in the scattering tests suite
        sphere = Sphere(n=prior.Uniform(1, 2), r=prior.Uniform(0, 1))
        pars = {'n': 1.59, 'r': 0.5}

        for c, f in itertools.product([True, False], [True, False]):
            theory_in = Mie(
                compute_escat_radial=c,
                full_radial_dependence=f)
            model = Model(sphere, theory=theory_in)
            theory_out = model.theory_from_parameters(pars)
            self.assertEqual(theory_out.compute_escat_radial, c)
            self.assertEqual(theory_out.full_radial_dependence, f)

    @attr('fast')
    def test_auto_theory_from_parameters_when_spheres_gives_multisphere(self):
        """testing a bug where the theory would always be Mie for spheres"""
        index = prior.Uniform(1, 2)
        radius = prior.Uniform(0, 1)
        sphere1 = Sphere(n=index, r=radius, center=[1, 2, 3])
        sphere2 = Sphere(n=index, r=radius, center=[1, 1, 1])
        spheres = Spheres([sphere1, sphere2])
        model = Model(spheres, theory='auto')

        self.assertIsInstance(model.theory, Multisphere)

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
    def test_initial_guess_scatterer(self):
        sphere = Sphere(n=prior.Uniform(1, 2),
                        r=prior.Uniform(0, 1, guess=0.8),
                        center=[2, 2, 2])
        model = AlphaModel(sphere)
        expected = Sphere(n=1.5, r=0.8, center=[2, 2, 2])
        self.assertEqual(model.initial_guess_scatterer, expected)

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

    @attr('fast')
    def test_theory_casts_from_auto(self):
        sphere = Sphere()
        model = AlphaModel(sphere, theory='auto')
        self.assertIsInstance(model.theory, ScatteringTheory)

    @attr('fast')
    def test_theory_casts_correctly_when_not_auto(self):
        sphere = Sphere()
        model = AlphaModel(sphere, theory=Mie())
        self.assertIsInstance(model.theory, ScatteringTheory)

    @attr('fast')
    def test_init_maps_theory_parameters(self):
        sphere = Sphere()
        theory = MieLens()
        model = AlphaModel(sphere, theory=theory)

        maps = model._maps
        self.assertIn('theory', maps)
        # raise ValueError(maps['scatterer'], sphere.parameters)
        theory_map = [dict, [[['lens_angle', 1.0]]]]
        self.assertEqual(maps['theory'], theory_map)


class TestAdd_Tie(unittest.TestCase):
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
        n = prior.ComplexPrior(prior.Uniform(1, 2), prior.Uniform(0, 1))
        sphere = Sphere(n=n, r=prior.Uniform(0.5, 1),
                        center=[prior.Uniform(0, 1), prior.Uniform(0, 1),
                                prior.Uniform(0, 10)])
        model = AlphaModel(sphere)
        model.add_tie(['center.0', 'n.imag', 'center.1'])
        expected_map = [
            dict,
            [[['n', [transformed_prior, [complex, ['_parameter_0',
                                                   '_parameter_1']]]],
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


class TestExactModel(unittest.TestCase):
    @attr('fast')
    def test_forward_correctly_creates_mielens_theory(self):
        model = ExactModel(
            SPHERE_IN_METERS,
            theory=MieLens(prior.Uniform(0., 1.0)))

        np.random.seed(1032)
        lens_angle = np.random.rand()
        pars = {
            'n': 1.5,
            'r': 0.5e-6,
            'lens_angle': lens_angle,
            }

        from_model = model.forward(pars, xschema_lens)

        theory = MieLens(lens_angle=lens_angle)
        scatterer = model.scatterer_from_parameters(pars)
        correct = calc_holo(
            xschema_lens, scatterer, theory=theory, scaling=1.0)

        self.assertTrue(np.all(from_model.values == correct.values))


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

    @attr('fast')
    def test_forward_uses_alpha(self):
        # we check that, if alpha = 0, the hologram is constant
        model = AlphaModel(
            SPHERE_IN_METERS,
            alpha=prior.Uniform(0, 1.0))
        pars = {
            'n': 1.5,
            'r': 0.5e-6,
            'alpha': 0,
            }
        holo = model.forward(pars, xschema_lens)
        self.assertLess(holo.values.std(), 1e-6)

    @attr('fast')
    def test_forward_correctly_creates_mielens_theory(self):
        model = AlphaModel(
            SPHERE_IN_METERS,
            theory=MieLens(prior.Uniform(0., 1.0)),
            alpha=prior.Uniform(0, 1.0))

        np.random.seed(1032)
        lens_angle = np.random.rand()
        alpha = np.random.rand()
        pars = {
            'n': 1.5,
            'r': 0.5e-6,
            'alpha': alpha,
            'lens_angle': lens_angle,
            }

        from_model = model.forward(pars, xschema_lens)

        theory = MieLens(lens_angle=lens_angle)
        scatterer = model.scatterer_from_parameters(pars)
        correct = calc_holo(
            xschema_lens, scatterer, theory=theory, scaling=alpha)

        self.assertTrue(np.all(from_model.values == correct.values))

    @attr('fast')
    def test_forward_correctly_creates_aberratedmielens_theory(self):
        n_ab_params = 4
        theory_parameterized = AberratedMieLens(
            lens_angle=prior.Uniform(0., 1.0),
            spherical_aberration=[
                prior.Uniform(-10, 10) for _ in range(n_ab_params)])
        model = AlphaModel(
            SPHERE_IN_METERS,
            theory=theory_parameterized,
            alpha=prior.Uniform(0, 1.0))

        np.random.seed(1032)
        lens_angle = np.random.rand()
        spherical_aberration = np.random.rand(n_ab_params) * 20 - 10
        alpha = np.random.rand()
        pars = {
            'n': 1.5,
            'r': 0.5e-6,
            'alpha': alpha,
            'lens_angle': lens_angle,
            }
        for i, v in enumerate(spherical_aberration):
            pars.update({f'spherical_aberration.{i}': v})

        from_model = model.forward(pars, xschema_lens)

        theory = AberratedMieLens(
            lens_angle=lens_angle, spherical_aberration=spherical_aberration)
        scatterer = model.scatterer_from_parameters(pars)
        correct = calc_holo(
            xschema_lens, scatterer, theory=theory, scaling=alpha)

        self.assertTrue(np.all(from_model.values == correct.values))


def make_sphere():
    index = prior.Uniform(1.4, 1.6, name='n')
    radius = prior.Uniform(0.2, 0.8, name='r')
    return Sphere(n=index, r=radius)


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
