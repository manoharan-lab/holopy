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

import yaml
import numpy as np
import xarray as xr
from collections import OrderedDict

from nose.plugins.attrib import attr
from numpy.testing import assert_raises

from holopy.core import detector_grid, update_metadata, holopy_object
from holopy.core.tests.common import assert_equal, assert_obj_close
from holopy.scattering import Sphere, Spheres, Mie, calc_holo
from holopy.scattering.scatterer.scatterer import _interpret_parameters
from holopy.scattering.errors import MissingParameter
from holopy.core.tests.common import assert_read_matches_write
from holopy.inference import (prior, AlphaModel, ExactModel,
                              NmpfitStrategy, EmceeStrategy,
                              available_fit_strategies,
                              available_sampling_strategies)
from holopy.inference.model import Model, PerfectLensModel
from holopy.inference.tests.common import SimpleModel
from holopy.scattering.tests.common import (
    xschema_lens, sphere as SPHERE_IN_METERS)


class TestModel(unittest.TestCase):
    model_keywords = [
        'noise_sd',
        'medium_index',
        'illum_wavelen',
        'illum_polarization',
        'theory',
        ]

    @attr('fast')
    def test_initializable(self):
        scatterer = make_sphere()
        model = Model(scatterer)
        self.assertTrue(model is not None)

    @attr('fast')
    def test_initializing_with_xarray_raises_error(self):
        sphere = make_sphere()
        for key in self.model_keywords:
            value = xr.DataArray(
                [1, 0.],
                dims=['illumination'],
                coords={'illumination': ['red', 'green']})
            kwargs = {key: value}
            error_regex = '{} cannot be an xarray'.format(key)
            with self.subTest(key=key):
                self.assertRaisesRegex(
                    ValueError, error_regex, Model, sphere, **kwargs)

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
    @unittest.skip("There is a problem with saving yaml xarrays")
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


class TestAlphaModel(unittest.TestCase):
    @attr('fast')
    def test_initializable(self):
        scatterer = make_sphere()
        model = AlphaModel(scatterer, alpha=0.6)
        self.assertTrue(model is not None)

    @attr('fast')
    def test_initializing_with_xarray_alpha_raises_error(self):
        sphere = make_sphere()
        alpha_xarray = xr.DataArray(
            [1, 0.5],
            dims=['illumination'],
            coords={'illumination': ['red', 'green']})
        error_regex = 'alpha cannot be an xarray'
        self.assertRaisesRegex(
            ValueError, error_regex, AlphaModel, sphere, alpha=alpha_xarray)

    @attr("fast")
    @unittest.skip("There is a problem with saving yaml xarrays")
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
    def test_initializing_with_xarray_lens_angle_raises_error(self):
        sphere = make_sphere()
        lens_angle_xarray = xr.DataArray(
            [0.6, 0.6],
            dims=['illumination'],
            coords={'illumination': ['red', 'green']})
        error_regex = 'lens_angle cannot be an xarray'
        self.assertRaisesRegex(
            ValueError, error_regex, PerfectLensModel, sphere,
            lens_angle=lens_angle_xarray)

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
    index = prior.Uniform(1.4, 1.6)
    radius = prior.Uniform(0.2, 0.8)
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
def test_ComplexPrior():
    parm = Sphere(n=prior.ComplexPrior(real=prior.Uniform(1.58,1.59), imag=.001))
    model = AlphaModel(parm, alpha=prior.Uniform(.6, 1, .7))
    assert_equal(model.parameters['n.real'].name, 'n.real')
    interpreted_pars = {'alpha':.7, 'n':{'real':1.585}}
    assert_equal(_interpret_parameters(model.parameters), interpreted_pars)


@attr('fast')
def test_multidim():
    par_s = Sphere(
        n={'r': prior.Uniform(-1,1), 'g': 0, 'b': prior.Gaussian(0,1),'a':0},
        r=xr.DataArray(
            [prior.Gaussian(0,1), prior.Uniform(-1,1), 0, 0],
            dims='alph', coords={'alph': ['a', 'b', 'c', 'd']}),
            center=[prior.Uniform(-1, 1), 0, 0])
    params = {'n:r': 3, 'n:g': 4, 'n:b': 5, 'n:a': 6, 'r:a': 7, 'r:b': 8,
              'r:c': 9, 'r:d': 10, 'center.0': 7, 'center.1': 8,
              'center.2': 9}
    out_s = Sphere(
        n={'r':3, 'g':0, 'b':5, 'a':0},
        r={'a':7, 'b':8, 'c':0, 'd':0}, center=[7, 0, 0])
    assert_obj_close(par_s.from_parameters(params), out_s)

    m = ExactModel(out_s, np.sum)
    parletters = {'r':prior.Uniform(-1,1),'g':0,'b':prior.Gaussian(0,1),'a':0}
    parcount = xr.DataArray([prior.Gaussian(0,1),prior.Uniform(-1,1),0,0],dims='numbers',coords={'numbers':['one', 'two', 'three', 'four']})

    m._use_parameters({'letters':parletters, 'count':parcount})
    expected_params = {'letters:r':prior.Uniform(-1,1, 0, 'letters:r'),'letters:b':prior.Gaussian(0,1,'letters:b'),'count:one':prior.Gaussian(0,1, 'count:one'),'count:two':prior.Uniform(-1,1, 0,'count:two')}
    assert_equal(m.parameters, expected_params)


@attr('fast')
def test_pullingoutguess():
    g = Sphere(center = (prior.Uniform(0, 1e-5, guess=.567e-5),
                   prior.Uniform(0, 1e-5, .567e-5), prior.Uniform(1e-5, 2e-5, 15e-6)),
         r = prior.Uniform(1e-8, 1e-5, 8.5e-7), n = prior.ComplexPrior(prior.Uniform(1, 2, 1.59),1e-4))

    model = ExactModel(g, calc_holo)

    s = Sphere(center = [.567e-5, .567e-5, 15e-6], n = 1.59 + 1e-4j, r = 8.5e-7)

    assert_equal(s.n, model.scatterer.guess.n)
    assert_equal(s.r, model.scatterer.guess.r)
    assert_equal(s.center, model.scatterer.guess.center)

    g = Sphere(center = (prior.Uniform(0, 1e-5, guess=.567e-5),
                   prior.Uniform(0, 1e-5, .567e-5), prior.Uniform(1e-5, 2e-5, 15e-6)),
         r = prior.Uniform(1e-8, 1e-5, 8.5e-7), n = 1.59 + 1e-4j)

    model = ExactModel(g, calc_holo)

    s = Sphere(center = [.567e-5, .567e-5, 15e-6], n = 1.59 + 1e-4j, r = 8.5e-7)

    assert_equal(s.n, model.scatterer.guess.n)
    assert_equal(s.r, model.scatterer.guess.r)
    assert_equal(s.center, model.scatterer.guess.center)


@attr('fast')
def test_find_noise():
    noise=0.5
    s = Sphere(n=prior.Uniform(1.5, 1.7), r=2, center=[1,2,3])
    data_base = detector_grid(10, spacing=0.5)
    data_noise = update_metadata(data_base, noise_sd=noise)
    model_u = AlphaModel(s, alpha=prior.Uniform(0.7,0.9))
    model_g = AlphaModel(s, alpha=prior.Gaussian(0.8, 0.1))
    pars = {'n':1.6, 'alpha':0.8}
    assert_equal(model_u._find_noise(pars, data_noise), noise)
    assert_equal(model_g._find_noise(pars, data_noise), noise)
    assert_equal(model_u._find_noise(pars, data_base), 1)
    assert_raises(MissingParameter, model_g._find_noise, pars, data_base)
    pars.update({'noise_sd':noise})
    assert_equal(model_g._find_noise(pars, data_base), noise)


@attr('fast')
def test_io():
    model = ExactModel(Sphere(1), calc_holo)
    assert_read_matches_write(model)

    model = ExactModel(Sphere(1), calc_holo, theory=Mie(False))
    assert_read_matches_write(model)


if __name__ == '__main__':
    unittest.main()

