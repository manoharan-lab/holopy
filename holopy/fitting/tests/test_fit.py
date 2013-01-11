# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca
# W. Perry, Jerome Fung, and Ryan McGorty
#
# This file is part of Holopy.
#
# Holopy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Holopy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Holopy.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import division

import tempfile

import numpy as np

from nose.tools import nottest
from nose.plugins.attrib import attr
from numpy.testing import assert_equal, assert_approx_equal, assert_allclose
from ...scattering.scatterer import Sphere, Spheres, Scatterer
from ...scattering.theory import Mie, Multisphere, DDA
from ...core import Optics, ImageSchema, load, save
from ...core.process import normalize
from ...core.helpers import OrderedDict
from .. import fit, Parameter, ComplexParameter, par, Parametrization, Model
from ...core.tests.common import assert_obj_close, get_example_data

gold_alpha = .6497

gold_sphere = Sphere(n = 1.582+1e-4j, r = 6.484e-7,
                     center = (5.534e-6, 5.792e-6, 1.415e-5))

@attr('medium')
def test_fit_mie_single():
    holo = normalize(get_example_data('image0001.yaml'))

    parameters = [Parameter(name='x', guess=.567e-5, limit = [0.0, 1e-5]),
                  Parameter(name='y', guess=.576e-5, limit = [0, 1e-5]),
                  Parameter(name='z', guess=15e-6, limit = [1e-5, 2e-5]),
                  Parameter(name='n', guess=1.59, limit = [1, 2]),
                  Parameter(name='r', guess=8.5e-7, limit = [1e-8, 1e-5])]


    def make_scatterer(x, y, z, r, n):
        return Sphere(n=n+1e-4j, r = r, center = (x, y, z))


    model = Model(Parametrization(make_scatterer, parameters), Mie.calc_holo,
                  alpha=Parameter(name='alpha', guess=.6, limit = [.1, 1]))

    result = fit(model, holo)

    assert_obj_close(result.scatterer, gold_sphere, rtol = 1e-3)
    assert_approx_equal(result.parameters['alpha'], gold_alpha, significant=3)
    assert_equal(model, result.model)


@attr('medium')
def test_fit_mie_par_scatterer():
    holo = normalize(get_example_data('image0001.yaml'))

    s = Sphere(center = (par(guess=.567e-5, limit=[0,1e-5]),
                         par(.567e-5, (0, 1e-5)), par(15e-6, (1e-5, 2e-5))),
               r = par(8.5e-7, (1e-8, 1e-5)),
               n = ComplexParameter(par(1.59, (1,2)), 1e-4j))


    model = Model(s, Mie.calc_holo, alpha = par(.6, [.1,1]))

    result = fit(model, holo)

    assert_obj_close(result.scatterer, gold_sphere, rtol=1e-3)
    # TODO: see if we can get this back to 3 sig figs correct alpha
    assert_approx_equal(result.parameters['alpha'], gold_alpha, significant=3)
    assert_equal(model, result.model)

@attr('fast')
def test_fit_random_subset():
    holo = normalize(get_example_data('image0001.yaml'))


    s = Sphere(center = (par(guess=.567e-5, limit=[0,1e-5]),
                         par(.567e-5, (0, 1e-5)), par(15e-6, (1e-5, 2e-5))),
               r = par(8.5e-7, (1e-8, 1e-5)), n = ComplexParameter(par(1.59, (1,2)),1e-4j))


    model = Model(s, Mie.calc_holo, schema_overlay=ImageSchema(use_random_fraction = .1), alpha = par(.6, [.1,1]))

    result = fit(model, holo)

    # we have to use a relatively loose tolerance here because the random
    # selection occasionally causes the fit to be a bit worse
    assert_obj_close(result.scatterer, gold_sphere, rtol=2e-2)
    # TODO: figure out if it is a problem that alpha is frequently coming out
    # wrong in the 3rd decimal place.
    assert_approx_equal(result.parameters['alpha'], gold_alpha, significant=2)
    assert_equal(model, result.model)



@nottest
@attr('slow')
def test_fit_superposition():
    """
    Fit Mie superposition to a calculated hologram from two spheres
    """
    # Make a test hologram
    optics = Optics(wavelen=6.58e-07, index=1.33, polarization=[0.0, 1.0],
                    divergence=0, spacing=None, train=None, mag=None,
                    pixel_scale=[2*2.302e-07, 2*2.302e-07])

    s1 = Sphere(n=1.5891+1e-4j, r = .65e-6,
                center=(1.56e-05, 1.44e-05, 15e-6))
    s2 = Sphere(n=1.5891+1e-4j, r = .65e-6,
                center=(3.42e-05, 3.17e-05, 10e-6))
    sc = Spheres([s1, s2])
    alpha = .629

    theory = Mie(optics, 100)
    holo = theory.calc_holo(sc, alpha)

    # Now construct the model, and fit
    parameters = [Parameter(name = 'x0', guess = 1.6e-5, limit = [0, 1e-4]),
                  Parameter('y0', 1.4e-5, [0, 1e-4]),
                  Parameter('z0', 15.5e-6, [0, 1e-4]),
                  Parameter('r0', .65e-6, [0.6e-6, 0.7e-6]),
                  Parameter('nr', 1.5891, [1, 2]),
                  Parameter('x1', 3.5e-5, [0, 1e-4]),
                  Parameter('y1', 3.2e-5, [0, 1e-4]),
                  Parameter('z1', 10.5e-6, [0, 1e-4]),
                  Parameter('r1', .65e-6, [0.6e-6, 0.7e-6])]

    def make_scatterer(x0, x1, y0, y1, z0, z1, r0, r1, nr):
        s = Spheres([
                Sphere(center = (x0, y0, z0), r=r0, n = nr+1e-4j),
                Sphere(center = (x1, y1, z1), r=r1, n = nr+1e-4j)])
        return s

    model = Model(parameters, Mie, make_scatterer=make_scatterer, alpha =
                  Parameter('alpha', .63, [.5, 0.8]))
    result = fit(model, holo)

    assert_obj_close(result.scatterer, sc)
    assert_approx_equal(result.alpha, alpha, significant=4)
    assert_equal(result.model, model)

@nottest
# TODO: disabled because it is old, slow, not functioning. Consider updating and
# reenabling as an integration test
@attr('slow')
def test_fit_multisphere_noisydimer_slow():
    optics = Optics(wavelen=658e-9, polarization = [0., 1.0],
                       divergence = 0., pixel_scale = [0.345e-6, 0.345e-6],
                       index = 1.334)

    holo = normalize(get_example_data('image0002.yaml'))

    # Now construct the model, and fit
    parameters = [Parameter(name = 'x0', guess = 1.64155e-5,
                            limit = [0, 1e-4]),
                  Parameter(1.7247e-5, [0, 1e-4], 'y0'),
                  Parameter(20.582e-6, [0, 1e-4], 'z0'),
                  Parameter(.6856e-6, [1e-8, 1e-4], 'r0'),
                  Parameter(1.6026, [1, 2], 'nr0'),
                  Parameter(1.758e-5, [0, 1e-4], 'x1'),
                  Parameter(1.753e-5, [0, 1e-4], 'y1'),
                  Parameter(21.2698e-6, [1e-8, 1e-4], 'z1'),
                  Parameter(.695e-6, [1e-8, 1e-4], 'r1'),
                  Parameter(1.6026, [1, 2], 'nr1')]

    def make_scatterer(x0, x1, y0, y1, z0, z1, r0, r1, nr0, nr1):
        s = Spheres([
                Sphere(center = (x0, y0, z0), r=r0, n = nr0+1e-5j),
                Sphere(center = (x1, y1, z1), r=r1, n = nr1+1e-5j)])
        return s

    # initial guess
    #s1 = Sphere(n=1.6026+1e-5j, r = .6856e-6,
    #            center=(1.64155e-05, 1.7247e-05, 20.582e-6))
    #s2 = Sphere(n=1.6026+1e-5j, r = .695e-6,
    #            center=(1.758e-05, 1.753e-05, 21.2698e-6))
    #sc = Spheres([s1, s2])
    #alpha = 0.99

    #lb1 = Sphere(1+1e-5j, 1e-8, 0, 0, 0)
    #ub1 = Sphere(2+1e-5j, 1e-5, 1e-4, 1e-4, 1e-4)
    #step1 = Sphere(1e-4+1e-4j, 1e-8, 0, 0, 0)
    #lb = Spheres([lb1, lb1]), .1
    #ub = Spheres([ub1, ub1]), 1
    #step = Spheres([step1, step1]), 0

    model = Model(parameters, Multisphere, make_scatterer=make_scatterer, alpha
    = Parameter(.99, [.1, 1.0], 'alpha'))
    result = fit(model, holo)
    print result.scatterer

    gold = np.array([1.642e-5, 1.725e-5, 2.058e-5, 1e-5, 1.603, 6.857e-7,
                     1.758e-5, 1.753e-5, 2.127e-5, 1e-5, 1.603,
                     6.964e-7])
    gold_alpha = 1.0

    assert_parameters_allclose(result.scatterer, gold, rtol=1e-2)
    # TODO: This test fails, alpha comes back as .9899..., where did
    # the gold come from?
    assert_approx_equal(result.alpha, gold_alpha, significant=2)


@nottest
@attr('fast')
def test_serialization():
    par_s = Sphere(center = (par(.567e-5, [0, 1e-5]), par(.576e-6, [0, 1e-5]),
                                                           par(15e-6, [1e-5,
                                                                       2e-5])),
                   r = par(8.5e-7, [1e-8, 1e-5]), n = par(1.59, [1,2]))

    alpha = par(.6, [.1, 1], 'alpha')

    schema = ImageSchema(shape = 100, spacing = .1151e-6, optics = Optics(.66e-6, 1.33))

    model = Model(par_s, Mie.calc_holo, alpha=alpha)

    holo = Mie.calc_holo(model.scatterer.guess, schema, model.alpha.guess)

    result = fit(model, holo)

    temp = tempfile.NamedTemporaryFile()
    save(temp, result)

    temp.flush()
    temp.seek(0)
    loaded = load(temp)

    assert_obj_close(result, loaded, context = 'serialized_result')

# TODO: disabled because it has gotten out of date and is slow, figure out
# something that will finish faster.
@nottest
@attr('slow')
def test_dda_fit():
    s = Sphere(n = 1.59, r = .2, center = (5, 5, 5))
    o = Optics(wavelen = .66, index=1.33, pixel_scale=.1)

    schema = ImageSchema(optics = o, shape = 100)

    h = Mie.calc_holo(s, schema)

    def make_scatterer(r, x, y, z):
        local_s = Sphere(r = r, center = (x, y, z))
        return Scatterer(local_s.indicators, n = s.n)

    parameters = [par(.18, [.1, .3], name='r', step=.1), par(5, [4, 6], 'x'),
                  par(5, [4,6], 'y'), par(5.2, [4, 6], 'z')]

    p = Parametrization(make_scatterer, parameters)

    model = Model(p, DDA.calc_holo)

    res = fit(model, h)

    assert_parameters_allclose(res.parameters, OrderedDict([('r',
    0.2003609439787491), ('x', 5.0128083665603995), ('y', 5.0125252883133617),
    ('z', 4.9775097284878775)]), rtol=1e-3)


def test_integer_correctness():
    # we keep having bugs where the fitter doesn't
    schema = ImageSchema(shape = 100, spacing = .1, optics = Optics(wavelen = .660, index = 1.33))
    s = Sphere(center = (10.2, 9.8, 10.3), r = .5, n = 1.58)
    holo = Mie.calc_holo(s, schema)

    par_s = Sphere(center = (par(guess = 10, limit = [5,15]), par(10, [5, 15]), par(10, [5, 15])),
                   r = .5, n = 1.58)

    model = Model(par_s, Mie.calc_holo, alpha = par(.6, [.1, 1]))
    result = fit(model, holo)
    assert_allclose(result.scatterer.center, [10.2, 9.8, 10.3])

def test_model_guess():
    ps = Sphere(n=par(1.59, [1.5,1.7]), r = .5, center=(5,5,5))
    m = Model(ps, Mie)
    assert_obj_close(m.scatterer.guess, Sphere(n=1.59, r=0.5, center=[5, 5, 5]))
