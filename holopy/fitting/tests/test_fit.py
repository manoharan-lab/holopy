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
import numpy as np
from nose.plugins.attrib import attr
from numpy.testing import assert_equal, assert_approx_equal, assert_allclose, assert_raises

from ...scattering import Sphere, Spheres, LayeredSphere, Mie, calc_holo
from ...core import detector_grid, load, save, update_metadata
from ...core.process import normalize
from .. import fit, Parameter, ComplexParameter, Model, limit_overlaps
from ...core.tests.common import (assert_obj_close, get_example_data, assert_read_matches_write)
from holopy.scattering.errors import OverlapWarning

gold_alpha = .6497

gold_sphere = Sphere(1.582+1e-4j, 6.484e-7,
                     (5.534e-6, 5.792e-6, 1.415e-5))

# TODO: mie_single and mie_par_scatterer do not behave well under
# random subset fitting (par_scatterer will error with
# random_subset=.99). This could be a sign of some deeper problem, so
# it might be worth investigating - tgd 2014-09-18

@attr('slow')
def test_fit_mie_single():
    holo = normalize(get_example_data('image0001'))

    parameters = [Parameter(name='x', guess=.567e-5, limit = [0.0, 1e-5]),
                  Parameter(name='y', guess=.576e-5, limit = [0, 1e-5]),
                  Parameter(name='z', guess=15e-6, limit = [1e-5, 2e-5]),
                  Parameter(name='n', guess=1.59, limit = [1, 2]),
                  Parameter(name='r', guess=8.5e-7, limit = [1e-8, 1e-5])]

    def make_scatterer(parlist):
        return Sphere(n=parlist[3], r = parlist[4], center = parlist[0:3])

    thry = Mie(False)
    model = Model(make_scatterer(parameters), calc_holo, theory=thry,
                  alpha=Parameter(name='alpha', guess=.6, limit = [.1, 1]))

    result = fit(model, holo)

    assert_obj_close(result.scatterer, gold_sphere, rtol = 1e-3)
    assert_approx_equal(result.parameters['alpha'], gold_alpha, significant=3)
    assert_equal(model, result.model)

@attr('slow')
def test_fit_mie_par_scatterer():
    holo = normalize(get_example_data('image0001'))

    s = Sphere(center = (Parameter(guess=.567e-5, limit=[0,1e-5]),
                         Parameter(.567e-5, (0, 1e-5)), Parameter(15e-6, (1e-5, 2e-5))),
               r = Parameter(8.5e-7, (1e-8, 1e-5)),
               n = ComplexParameter(Parameter(1.59, (1,2)), 1e-4))

    thry = Mie(False)
    model = Model(s, calc_holo, theory=thry, alpha = Parameter(.6, [.1,1]))

    result = fit(model, holo)

    assert_obj_close(result.scatterer, gold_sphere, rtol=1e-3)
    # TODO: see if we can get this back to 3 sig figs correct alpha
    assert_approx_equal(result.parameters['alpha'], gold_alpha, significant=3)
    assert_equal(model, result.model)
    assert_read_matches_write(result)

@attr('fast')
def test_fit_random_subset():
    holo = normalize(get_example_data('image0001'))

    s = Sphere(center = (Parameter(guess=.567e-5, limit=[0,1e-5]),
                         Parameter(.567e-5, (0, 1e-5)), Parameter(15e-6, (1e-5, 2e-5))),
               r = Parameter(8.5e-7, (1e-8, 1e-5)), n = ComplexParameter(Parameter(1.59, (1,2)),1e-4))

    model = Model(s, calc_holo, theory=Mie(False), alpha = Parameter(.6, [.1,1]))
    np.random.seed(40)
    result = fit(model, holo, random_subset=.1)

    # TODO: this tolerance has to be rather large to pass, we should
    # probably track down if this is a sign of a problem
    assert_obj_close(result.scatterer, gold_sphere, rtol=1e-2)
    # TODO: figure out if it is a problem that alpha is frequently coming out
    # wrong in the 3rd decimal place.
    assert_approx_equal(result.parameters['alpha'], gold_alpha, significant=3)
    assert_equal(model, result.model)

    assert_read_matches_write(result)

def test_n():
    sph = Sphere(Parameter(.5), 1.6, (5,5,5))
    sch = detector_grid(shape=[100, 100], spacing=[0.1, 0.1])

    model = Model(sph, calc_holo, 1.33, .66, illum_polarization=(1, 0), alpha=1)
    holo = calc_holo(sch, model.scatterer.guess, 1.33, .66, (1, 0))
    assert_allclose(model.residual({'n' : .5}, holo), 0)


@attr('fast')
def test_serialization():
    par_s = Sphere(center = (Parameter(.567e-5, [0, 1e-5]), Parameter(.576e-6, [0, 1e-5]),
                                                           Parameter(15e-6, [1e-5,
                                                                       2e-5])),
                   r = Parameter(8.5e-7, [1e-8, 1e-5]), n = Parameter(1.59, [1,2]))

    alpha = Parameter(.6, [.1, 1], 'alpha')

    schema = update_metadata(detector_grid(shape = 100, spacing = .1151e-6), illum_wavelen=.66e-6, medium_index=1.33, illum_polarization=(1,0))

    model = Model(par_s, calc_func=calc_holo, medium_index=schema.medium_index, illum_wavelen=schema.illum_wavelen, alpha=alpha)

    holo = calc_holo(schema, model.scatterer.guess, scaling=model.alpha.guess)

    result = fit(model, holo)

    temp = tempfile.NamedTemporaryFile()
    save(temp, result)

    temp.flush()
    temp.seek(0)
    loaded = load(temp)
    temp.close()
    assert_obj_close(result, loaded, context = 'serialized_result')


def test_integer_correctness():
    # we keep having bugs where the fitter doesn't
    schema = detector_grid(shape = 100, spacing = .1)
    s = Sphere(center = (10.2, 9.8, 10.3), r = .5, n = 1.58)
    holo = calc_holo(schema, s, illum_wavelen = .660, medium_index = 1.33, illum_polarization = (1, 0))

    par_s = Sphere(center = (Parameter(guess = 10, limit = [5,15]), Parameter(10, [5, 15]), Parameter(10, [5, 15])),
                   r = .5, n = 1.58)

    model = Model(par_s, calc_holo, alpha = Parameter(.6, [.1, 1]))
    result = fit(model, holo)
    assert_allclose(result.scatterer.center, [10.2, 9.8, 10.3])

def test_model_guess():
    ps = Sphere(n=Parameter(1.59, [1.5,1.7]), r = .5, center=(5,5,5))
    m = Model(ps, calc_holo)
    assert_obj_close(m.scatterer.guess, Sphere(n=1.59, r=0.5, center=[5, 5, 5]))

def test_constraint():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OverlapWarning)
        spheres = Spheres([Sphere(r=.5, center=(0,0,0)),
                           Sphere(r=.5, center=(0,0,Parameter(.2)))])
        model = Model(spheres, calc_holo, constraints=limit_overlaps())
        cost = model.lnprior({'1:Sphere.center[2]' : .2})
        assert_equal(cost, -np.inf)

def test_layered():
    s = Sphere(n = (1,2), r = (1, 2), center = (2, 2, 2))
    sch = detector_grid((10, 10), .2)
    hs = calc_holo(sch, s, 1, .66, (1, 0))

    guess = LayeredSphere((1,2), (Parameter(1.01), Parameter(.99)), (2, 2, 2))
    model = Model(guess, calc_holo)
    res = fit(model, hs)
    assert_allclose(res.scatterer.t, (1, 1), rtol = 1e-12)
