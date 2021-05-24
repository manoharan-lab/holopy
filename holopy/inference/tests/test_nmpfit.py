# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, Ryan McGorty, Anna Wang, Solomon Barkley
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

import tempfile
import warnings
import unittest

import numpy as np
from nose.plugins.attrib import attr
from numpy.testing import (
    assert_equal, assert_approx_equal, assert_allclose, assert_raises)

from holopy.scattering import Sphere, Spheres, LayeredSphere, Mie, calc_holo
from holopy.core import detector_grid, load, save, update_metadata
from holopy.core.process import normalize
from holopy.core.tests.common import (
    assert_obj_close, get_example_data, assert_read_matches_write)
from holopy.scattering.errors import OverlapWarning
from holopy.inference import (
    LimitOverlaps, ExactModel, AlphaModel, NmpfitStrategy,
    LeastSquaresScipyStrategy)
from holopy.core.prior import ComplexPrior, Uniform

gold_alpha = .6497

gold_sphere = Sphere(1.582+1e-4j, 6.484e-7, (5.534e-6, 5.792e-6, 1.415e-5))


@attr('slow')
def test_fit_mie_single():
    holo = normalize(get_example_data('image0001'))

    parameters = [Uniform(0, 1e-5, name='x', guess=.567e-5),
                  Uniform(0, 1e-5, name='y', guess=.576e-5),
                  Uniform(1e-5, 2e-5, name='z', guess=15e-6),
                  Uniform(1, 2, name='n', guess=1.59),
                  Uniform(1e-8, 1e-5, name='r', guess=8.5e-7)]

    def make_scatterer(parlist):
        return Sphere(n=parlist[3], r = parlist[4], center = parlist[0:3])

    thry = Mie(False)
    model = AlphaModel(make_scatterer(parameters), theory=thry,
                  alpha=Uniform(.1, 1, name='alpha', guess=.6))

    result = NmpfitStrategy().fit(model, holo)

    assert_obj_close(result.scatterer, gold_sphere, rtol = 1e-3)
    assert_approx_equal(result.parameters['alpha'], gold_alpha,
                        significant=3)
    assert_equal(model, result.model)


@attr('slow')
def test_fit_mie_par_scatterer():
    holo = normalize(get_example_data('image0001'))

    s = Sphere(center = (Uniform(0, 1e-5, guess=.567e-5),
                         Uniform(0, 1e-5, .567e-5), Uniform(1e-5, 2e-5)),
               r = Uniform(1e-8, 1e-5, 8.5e-7),
               n = ComplexPrior(Uniform(1, 2, 1.59), 1e-4))

    thry = Mie(False)
    model = AlphaModel(s, theory=thry, alpha = Uniform(.1, 1, .6))

    result = NmpfitStrategy().fit(model, holo)
    assert_obj_close(result.scatterer, gold_sphere, rtol=1e-3)
    # TODO: see if we can get this back to 3 sig figs correct alpha
    assert_approx_equal(result.parameters['alpha'], gold_alpha,
                        significant=3)
    assert_equal(model, result.model)
    assert_read_matches_write(result)


class TestRandomSubsetFitting(unittest.TestCase):
    def _make_model(self):
        sphere = Sphere(
            center=(Uniform(0, 1e-5, guess=.567e-5),
                    Uniform(0, 1e-5, .567e-5),
                    Uniform(1e-5, 2e-5)),
            r=Uniform(1e-8, 1e-5, 8.5e-7),
            n=ComplexPrior(Uniform(1, 2, 1.59), 1e-4))

        model = AlphaModel(
            sphere, theory=Mie(False), alpha=Uniform(0.1, 1, 0.6))
        return model

    @attr('medium')
    def test_returns_close_values(self):
        model = self._make_model()
        holo = normalize(get_example_data('image0001'))

        np.random.seed(40)
        result = NmpfitStrategy(npixels=1000).fit(model, holo)

        # TODO: figure out if it is a problem that alpha is frequently
        # coming out wrong in the 3rd decimal place.
        self.assertAlmostEqual(
            result.parameters['alpha'],
            gold_alpha,
            places=3)
        # TODO: this tolerance has to be rather large to pass, we should
        # probably track down if this is a sign of a problem
        assert_obj_close(result.scatterer, gold_sphere, rtol=1e-2)


@attr('medium')
def test_serialization():
    par_s = Sphere(center = (Uniform(0, 1e-5, guess=.567e-5),
                         Uniform(0, 1e-5, .567e-5), Uniform(1e-5, 2e-5)),
               r = Uniform(1e-8, 1e-5, 8.5e-7),
               n = Uniform(1, 2, 1.59))

    alpha = Uniform(.1, 1, .6, 'alpha')

    schema = update_metadata(detector_grid(shape = 100, spacing = .1151e-6), illum_wavelen=.66e-6, medium_index=1.33, illum_polarization=(1,0))

    model = AlphaModel(par_s, medium_index=schema.medium_index, illum_wavelen=schema.illum_wavelen, alpha=alpha)

    initial_guess = model.scatterer_from_parameters(model.initial_guess)
    holo = calc_holo(schema, initial_guess, scaling=model.alpha.guess)

    result = NmpfitStrategy().fit(model, holo)
    temp = tempfile.NamedTemporaryFile(suffix = '.h5', delete=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        save(temp.name, result)
        loaded = load(temp.name)
        assert_obj_close(result, loaded, context = 'serialized_result')


def test_integer_correctness():
    # we keep having bugs where the fitter doesn't
    schema = detector_grid(shape=10, spacing=1)
    s = Sphere(center = (10.2, 9.8, 10.3), r = .5, n = 1.58)
    holo = calc_holo(schema, s, illum_wavelen = .660, medium_index = 1.33,
                     illum_polarization = (1, 0))
    par_s = Sphere(r = .5, n = 1.58,
                   center = (Uniform(5, 15), Uniform(5, 15), Uniform(5, 15)))
    model = AlphaModel(par_s, alpha = Uniform(.1, 1, .6))
    result = NmpfitStrategy().fit(model, holo)
    assert_allclose(result.scatterer.center, [10.2, 9.8, 10.3])


def test_model_guess():
    ps = Sphere(n=Uniform(1.5, 1.7, 1.59), r = .5, center=(5,5,5))
    m = ExactModel(ps, calc_holo)
    initial_guess = m.scatterer_from_parameters(m.initial_guess)
    assert_obj_close(initial_guess, Sphere(n=1.59, r=0.5, center=[5, 5, 5]))


def test_constraint():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OverlapWarning)
        spheres = Spheres([Sphere(r=.5, center=(0,0,0)),
                           Sphere(r=.5, center=(0,0,.2))])
        model = ExactModel(spheres, calc_holo, constraints=LimitOverlaps())
        cost = model.lnprior([])
        assert_equal(cost, -np.inf)


def test_layered():
    s = Sphere(n = (1,2), r = (1, 2), center = (2, 2, 2))
    sch = detector_grid((10, 10), .2)
    hs = calc_holo(sch, s, 1, .66, (1, 0))

    guess = LayeredSphere((1,2), (Uniform(1, 1.01), Uniform(.99,1)), (2, 2, 2))
    model = ExactModel(guess, calc_holo)
    res = NmpfitStrategy().fit(model, hs)
    assert_allclose(res.scatterer.t, (1, 1), rtol = 1e-12)


if __name__ == '__main__':
    unittest.main()
