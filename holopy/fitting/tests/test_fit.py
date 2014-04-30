# Copyright 2011-2013, Vinothan N. Manoharan, Thomas G. Dimiduk,
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
from __future__ import division

import tempfile
import warnings
import numpy as np
import holopy as hp

from nose.tools import nottest, assert_raises
from nose.plugins.skip import SkipTest
from nose.plugins.attrib import attr
from numpy.testing import assert_equal, assert_approx_equal, assert_allclose, assert_array_equal
from holopy.fitting.minimizer import OpenOpt
from holopy.scattering.scatterer import Sphere, Spheres, Scatterer
from holopy.scattering.theory import Mie, Multisphere, DDA
from holopy.core import Optics, ImageSchema, load, save, Schema, Angles, Marray
from holopy.core.process import normalize
from holopy.core.helpers import OrderedDict
from holopy.fitting import fit, Parameter, ComplexParameter, par, Parametrization, Model
from holopy.core.tests.common import (assert_obj_close, get_example_data,
                                  assert_read_matches_write)
from holopy.fitting.fit import CostComputer
from holopy.fitting import Model, FitResult
from ..errors import InvalidMinimizer
from holopy.fitting.model import limit_overlaps, ParameterizedObject

gold_alpha = .6497

gold_sphere = Sphere(1.582+1e-4j, 6.484e-7,
                     (5.534e-6, 5.792e-6, 1.415e-5))

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

    thry = Mie(False)
    model = Model(Parametrization(make_scatterer, parameters), thry.calc_holo,
                  alpha=Parameter(name='alpha', guess=.6, limit = [.1, 1]))

    assert_raises(InvalidMinimizer, fit, model, holo, minimizer=Sphere)

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

    thry = Mie(False)
    model = Model(s, thry.calc_holo, alpha = par(.6, [.1,1]))

    result = fit(model, holo)

    assert_obj_close(result.scatterer, gold_sphere, rtol=1e-3)
    # TODO: see if we can get this back to 3 sig figs correct alpha
    assert_approx_equal(result.parameters['alpha'], gold_alpha, significant=3)
    assert_equal(model, result.model)
    assert_read_matches_write(result)

@attr('slow')
def test_fit_single_openopt():
    holo = normalize(get_example_data('image0001.yaml'))
    s = Sphere(center = (par(guess=.567e-5, limit=[.4e-5,.6e-5]),
                         par(.567e-5, (.4e-5, .6e-5)), par(15e-6, (1.3e-5, 1.8e-5))),
               r = par(8.5e-7, (5e-7, 1e-6)),
               n = ComplexParameter(par(1.59, (1.5,1.8)), 1e-4j))

    model = Model(s, Mie(False).calc_holo, alpha = par(.6, [.1,1]))
    try:
        minimizer = OpenOpt('scipy_slsqp')
    except ImportError:
        raise SkipTest
    result = fit(model, holo, minimizer = minimizer)
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

    model = Model(s, Mie.calc_holo, alpha = par(.6, [.1,1]))
    np.random.seed(40)
    result = fit(model, holo, use_random_fraction=.1)

    # we have to use a relatively loose tolerance here because the random
    # selection occasionally causes the fit to be a bit worse
    assert_obj_close(result.scatterer, gold_sphere, rtol=1e-2)
    # TODO: figure out if it is a problem that alpha is frequently coming out
    # wrong in the 3rd decimal place.
    assert_approx_equal(result.parameters['alpha'], gold_alpha, significant=2)
    assert_equal(model, result.model)

    assert_read_matches_write(result)

@attr('fast')
def test_next_model():
    exampleresult = FitResult(parameters={
        'center[1]': 31.367170884695756, 'r': 0.6465280831465722, 
        'center[0]': 32.24150087110443, 
        'center[2]': 35.1651561654966, 
        'alpha': 0.7176299231169572, 
        'n': 1.580122175314896}, 
        scatterer=Sphere(n=1.580122175314896, r=0.6465280831465722, 
        center=[32.24150087110443, 31.367170884695756, 35.1651561654966]), 
        chisq=0.0001810513851216454, rsq=0.9727020197282801, 
        converged=True, time=5.179728031158447, 
        model=Model(scatterer=ParameterizedObject(obj=
        Sphere(n=Parameter(guess=1.59, limit=[1.4, 1.7], name='n'), 
        r=Parameter(guess=0.65, limit=[0.6, 0.7], name='r'), 
        center=[Parameter(guess=32.110424836601304, limit=[2, 40], name='center[0]'), 
        Parameter(guess=31.56683986928105, limit=[4, 40], name='center[1]'), 
        Parameter(guess=33, limit=[5, 45], name='center[2]')])), 
        theory=Mie.calc_holo, alpha=Parameter(guess=0.6, limit=[0.1, 1], name='alpha'), 
        constraints=[]), minimizer = None, minimization_details = None)

    gold = Model(scatterer=ParameterizedObject(obj=Sphere(
        n=Parameter(guess=1.580122175314896, limit=[1.4, 1.7], name='n'), 
        r=Parameter(guess=0.6465280831465722, limit=[0.6, 0.7], name='r'), 
        center=[Parameter(guess=32.24150087110443, limit=[2, 40], name='center[0]'), 
        Parameter(guess=31.367170884695756, limit=[4, 40], name='center[1]'), 
        Parameter(guess=35.1651561654966, limit=[5, 45], name='center[2]')])), 
        theory=Mie.calc_holo, alpha=Parameter(guess=0.7176299231169572, limit=[0.1, 1], name='alpha'), 
        constraints=[])

    assert_obj_close(gold, exampleresult.next_model())

def test_n():
    sph = Sphere(par(.5), 1.6, (5,5,5))
    sch = ImageSchema(shape=[100, 100], spacing=[0.1, 0.1],
                      optics=Optics(wavelen=0.66,
                                    index=1.33,
                                    polarization=[1, 0],
                                    divergence=0.0),
                      origin=[0.0, 0.0, 0.0])

    model = Model(sph, Mie.calc_holo, alpha=1)
    holo = Mie.calc_holo(model.scatterer.guess, sch)
    coster = CostComputer(holo, model, use_random_fraction=.1)
    assert_allclose(coster.flattened_difference({'n' : .5}), 0)

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
    assert_read_matches_write(result)

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
    schema = ImageSchema(shape = 100, spacing = .1,
                         optics = Optics(wavelen = .660, index = 1.33, polarization = (1, 0)))
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


@attr('fast')
def test_fit_complex_parameter():
    '''
    Test that complex parameters are handled correctly when fit.
    '''

    # use a Sphere with complex n
    # a fake scattering model
    def scat_func(sph, schema, scaling = None):
        # TODO: scaling kwarg required, seems like a silly kluge
        def silly_function(theta):
            return theta * sph.r + sph.n.real * theta **2  + 2. * sph.n.imag
        #import pdb
        #pdb.set_trace()
        return Marray(np.array([silly_function(theta) for theta, phi in
                                schema.positions_theta_phi()]),
                      **schema._dict)

    # generate data
    schema = Schema(positions = Angles(np.linspace(0., np.pi/2., 6)))
    ref_sph = Sphere(r = 1.5, n = 0.4 + 0.8j)
    data = scat_func(ref_sph, schema)

    # varying both real and imaginary parts
    par_s = Sphere(r = par(1.49),
                   n = ComplexParameter(real = par(0.405), imag = par(0.81)))
    model = Model(par_s, scat_func)
    result = fit(model, data)
    assert_allclose(result.scatterer.r, ref_sph.r)
    assert_allclose(result.scatterer.n.real, ref_sph.n.real)
    assert_allclose(result.scatterer.n.imag, ref_sph.n.imag)

    # varying just the real part
    par_s2 = Sphere(r = par(1.49), n = ComplexParameter(real = par(0.405),
                                                       imag = 0.8))
    model2 = Model(par_s2, scat_func)
    result2 = fit(model2, data)
    assert_allclose(result2.scatterer.r, ref_sph.r)
    assert_allclose(result2.scatterer.n.real, ref_sph.n.real)
    assert_allclose(result2.scatterer.n.imag, ref_sph.n.imag)

def test_constraint():
    sch = ImageSchema(100)
    with warnings.catch_warnings():
        # TODO: we should really only supress overlap warnings here,
        # but I am too lazy to figure it out right now, and I don't
        # think we are likely to hit warnings here that won't get
        # caught elsewhere -tgd 2013-12-01
        warnings.simplefilter("ignore")
        spheres = Spheres([Sphere(r=.5, center=(0,0,0)),
                           Sphere(r=.5, center=(0,0,par(.2)))])
        model = Model(spheres, Multisphere.calc_holo, constraints=limit_overlaps())
        coster = CostComputer(sch, model)
        cost = coster._calc({'1:Sphere.center[2]' : .2})
        assert_equal(cost, np.ones_like(sch)*np.inf)

def test_layered():
    s = Sphere(n = (1,2), r = (1, 2), center = (2, 2, 2))
    sch = ImageSchema((10, 10), .2, Optics(.66, 1, (1, 0)))
    hs = Mie.calc_holo(s, sch)

    guess = hp.scattering.scatterer.sphere.LayeredSphere((1,2), (par(1.01), par(.99)), (2, 2, 2))
    model = Model(guess, Mie.calc_holo)
    res = fit(model, hs)
    assert_allclose(res.scatterer.t, (1, 1), rtol = 1e-12)
