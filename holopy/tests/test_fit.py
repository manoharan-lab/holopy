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

import os
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
import tempfile

import numpy as np

import holopy as hp

from nose.tools import with_setup, nottest, set_trace
from nose.plugins.attrib import attr
from numpy.testing import assert_equal, assert_approx_equal, assert_raises, dec
from scatterpy.theory import Mie, Multisphere
from scatterpy.scatterer import Sphere, SphereCluster

from holopy.analyze.fit import (par, Parameter, Model, fit, Nmpfit,
                                ParameterSpecficationError, GuessOutOfBoundsError,
                                MinimizerConvergenceFailed, Parameterization,
                                ComplexParameter) 

from scatterpy.tests.common import assert_parameters_allclose, assert_obj_close, assert_allclose


def setup_optics():
    # set up optics class for use in several test functions
    global optics
    wavelen = 658e-9
    polarization = [0., 1.0]
    divergence = 0
    pixel_scale = [.1151e-6, .1151e-6]
    index = 1.33
    
    optics = hp.optics.Optics(wavelen=wavelen, index=index,
                              pixel_scale=pixel_scale,
                              polarization=polarization,
                              divergence=divergence)
    
def teardown_optics():
    global optics
    del optics

gold_single = OrderedDict((('center[0]', 5.534e-6),
               ('center[1]', 5.792e-6),
               ('center[2]', 1.415e-5),
               ('n.imag', 1e-4),
               ('n.real', 1.582),
               ('r', 6.484e-7)))
gold_alpha = .6497


gold_single = OrderedDict((('center[0]', 5.534e-6),
                           ('center[1]', 5.792e-6),
                           ('center[2]', 1.415e-5),
                           ('n.real', 1.582),
                           ('r', 6.484e-7),
                           ('alpha', .6497)))
@attr('medium')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_fit_mie_single():
    """
    Fit Mie theory to a hologram of a single sphere
    """
    path = os.path.abspath(hp.__file__)
    path = os.path.join(os.path.split(path)[0],'tests', 'exampledata')
    holo = hp.process.normalize(hp.load(os.path.join(path, 'image0001.npy'),
                                        optics=optics))

    parameters = [Parameter(name='x', guess=.567e-5, limit = [0.0, 1e-5]),
                  Parameter(name='y', guess=.576e-5, limit = [0, 1e-5]),
                  Parameter(name='z', guess=15e-6, limit = [1e-5, 2e-5]),
                  Parameter(name='n', guess=1.59, limit = [1, 2]),
                  Parameter(name='r', guess=8.5e-7, limit = [1e-8, 1e-5])]

    
    def make_scatterer(x, y, z, r, n):
        return Sphere(n=n+1e-4j, r = r, center = (x, y, z))

    
    model = Model(Parameterization(make_scatterer, parameters), Mie,
                  alpha=Parameter(name='alpha', guess=.6, limit = [.1, 1]))

    result = fit(model, holo)

    assert_parameters_allclose(result.parameters, gold_single, rtol = 1e-3)
    assert_equal(model, result.model)

    
@attr('medium')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_fit_mie_par_scatterer():
    path = os.path.abspath(hp.__file__)
    path = os.path.join(os.path.split(path)[0],'tests', 'exampledata')
    holo = hp.process.normalize(hp.load(os.path.join(path, 'image0001.npy'),
                                        optics=optics))

    
    s = Sphere(center = (par(guess=.567e-5, limit=[0,1e-5]),
                         par(.567e-5, (0, 1e-5)), par(15e-6, (1e-5, 2e-5))),
               r = par(8.5e-7, (1e-8, 1e-5)), 
               n = ComplexParameter(par(1.59, (1,2)), 1e-4j))

    
    model = Model(s, Mie, alpha = par(.6, [.1,1]))

    result = fit(model, holo)

    # TODO: make new structure work with complex n
    gold_single = OrderedDict((('center[0]', 5.534e-6),
                               ('center[1]', 5.792e-6),
                               ('center[2]', 1.415e-5),
                               ('n.real', 1.582),
                               ('r', 6.484e-7))) 
    
    assert_parameters_allclose(result.scatterer, gold_single, rtol=1e-3)
    # TODO: see if we can get this back to 3 sig figs correct alpha
    assert_approx_equal(result.parameters['alpha'], gold_alpha, significant=3)
    assert_equal(model, result.model)

@attr('fast')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_fit_selection():
    path = os.path.abspath(hp.__file__)
    path = os.path.join(os.path.split(path)[0],'tests', 'exampledata')
    holo = hp.process.normalize(hp.load(os.path.join(path, 'image0001.npy'),
                                        optics=optics))

    
    s = Sphere(center = (par(guess=.567e-5, limit=[0,1e-5]),
                         par(.567e-5, (0, 1e-5)), par(15e-6, (1e-5, 2e-5))),
               r = par(8.5e-7, (1e-8, 1e-5)), n = ComplexParameter(par(1.59, (1,2)),1e-4j))

    
    model = Model(s, Mie, selection = .1, alpha =  par(.6, [.1,1], 'alpha'))

    result = fit(model, holo)
 
    
    assert_parameters_allclose(result.parameters, gold_single, rtol=1e-2)
    assert_equal(model, result.model)



@nottest
@attr('slow')
def test_fit_superposition():
    """
    Fit Mie superposition to a calculated hologram from two spheres
    """
    # Make a test hologram
    optics = hp.Optics(wavelen=6.58e-07, index=1.33, polarization=[0.0, 1.0],
                    divergence=0, pixel_size=None, train=None, mag=None,
                    pixel_scale=[2*2.302e-07, 2*2.302e-07])

    s1 = Sphere(n=1.5891+1e-4j, r = .65e-6, 
                center=(1.56e-05, 1.44e-05, 15e-6))
    s2 = Sphere(n=1.5891+1e-4j, r = .65e-6, 
                center=(3.42e-05, 3.17e-05, 10e-6))
    sc = SphereCluster([s1, s2])
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
        s = SphereCluster([
                Sphere(center = (x0, y0, z0), r=r0, n = nr+1e-4j),
                Sphere(center = (x1, y1, z1), r=r1, n = nr+1e-4j)])
        return s

    model = Model(parameters, Mie, make_scatterer=make_scatterer, alpha =
                  Parameter('alpha', .63, [.5, 0.8]))
    result = fit(model, holo)

    assert_parameters_allclose(result.scatterer, sc)
    assert_approx_equal(result.alpha, alpha, significant=4)
    assert_equal(result.model, model)

@attr('slow')
def test_fit_multisphere_noisydimer_slow():
    """
    Fit multisphere superposition model to noisified dimer hologram
    """
    optics = hp.Optics(wavelen=658e-9, polarization = [0., 1.0], 
                       divergence = 0., pixel_scale = [0.345e-6, 0.345e-6], 
                       index = 1.334)

    path = os.path.abspath(hp.__file__)
    path = os.path.join(os.path.split(path)[0],'tests', 'exampledata')
    holo = hp.process.normalize(hp.load(os.path.join(path, 'image0002.npy'),
                                        optics=optics))
    
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
        s = SphereCluster([
                Sphere(center = (x0, y0, z0), r=r0, n = nr0+1e-5j),
                Sphere(center = (x1, y1, z1), r=r1, n = nr1+1e-5j)])
        return s

    # initial guess
    #s1 = Sphere(n=1.6026+1e-5j, r = .6856e-6, 
    #            center=(1.64155e-05, 1.7247e-05, 20.582e-6)) 
    #s2 = Sphere(n=1.6026+1e-5j, r = .695e-6, 
    #            center=(1.758e-05, 1.753e-05, 21.2698e-6)) 
    #sc = SphereCluster([s1, s2])
    #alpha = 0.99

    #lb1 = Sphere(1+1e-5j, 1e-8, 0, 0, 0)
    #ub1 = Sphere(2+1e-5j, 1e-5, 1e-4, 1e-4, 1e-4)
    #step1 = Sphere(1e-4+1e-4j, 1e-8, 0, 0, 0)
    #lb = SphereCluster([lb1, lb1]), .1
    #ub = SphereCluster([ub1, ub1]), 1    
    #step = SphereCluster([step1, step1]), 0

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


@attr('fast')
def test_parameter():
    p = Parameter(name='x', guess=.567e-5, limit = [0.0, 1e-5])
    assert_equal(1e-6, p.unscale(p.scale(1e-6)))

    with assert_raises(GuessOutOfBoundsError):
        Parameter(guess=1, limit=(2, 3))

    with assert_raises(GuessOutOfBoundsError):
        Parameter(guess=1, limit=3)

    # include a fixed complex index
    pj = ComplexParameter(par(1.59),  1e-4j)


    # include a fitted complex index with a guess of 1e-4
    pj2 = ComplexParameter(par(1.59), par(1e-4j))


@attr('fast')
def test_model():
    parameters = [Parameter(name='x', guess=.567e-5, limit = [0.0, 1e-5]),
                  Parameter(name='y', guess=.576e-5, limit = [0, 1e-5]),
                  Parameter(name='z', guess=15e-6, limit = [1e-5, 2e-5]),
                  Parameter(name='r', guess=8.5e-7, limit = [1e-8, 1e-5])]

    def make_scatterer(x, y, z, r):
        return Sphere(n=1.59+1e-4j, r = r, center = (x, y, z))

    param = Parameterization(make_scatterer, parameters)
    model = Model(param, Mie)

    x, y, z, r = 1, 2, 3, 1
    values = {'x': x, 'y': y, 'z': z, 'r': r}
    s = model.scatterer.make_from(values)

    assert_parameters_allclose(s, Sphere(center=(x, y, z), n=1.59+1e-4j,
                                         r=r))

    # check that Model correctly returns None when asked for alpha on a
    # parameter set that does not contain alpha
    
    parameters = [Parameter(name='x', guess=.567e-5, limit = [0.0, 1e-5]),
                  Parameter(name='y', guess=.576e-5, limit = [0, 1e-5]),
                  Parameter(name='z', guess=15e-6, limit = [1e-5, 2e-5]),
                  Parameter(name='r', guess=8.5e-7, limit = [1e-8, 1e-5])]
    model = Model(param, Mie, alpha = Parameter(name='alpha', guess=.6, limit = [.1, 1]))

    assert_equal(model.alpha.guess, .6)

@attr('fast')
def test_scatterer_based_model():
    s = Sphere(center = (par(guess=.567e-5),par(limit=.567e-5), par(15e-6, (1e-5, 2e-5))),
               r = 8.5e-7, n = ComplexParameter(par(1.59, (1,2)),1e-4j))

    
    model = Model(s, Mie)

    assert_obj_close(model.parameters, [Parameter(name='center[0]', guess=5.67e-06),
                                    Parameter(name='center[2]', guess=1.5e-05,
                                              limit=(1e-05, 2e-05)),
                                    Parameter(name='n.real', guess=1.59, limit=(1,
                                    2))], context = 'model.parameters')

    s2 = Sphere(center=[Parameter(name='center[0]', guess=5.67e-06),
                        par(limit=5.67e-06, name='center[1]'),
                        Parameter(name='center[2]', guess=1.5e-05, limit=(1e-05, 2e-05))],
                n=ComplexParameter(Parameter(name='n.real', guess=1.59,
                                             limit=(1, 2)),1e-4j), r=8.5e-07)
    s2.n.imag.name='n.imag'
    
    assert_obj_close(model.scatterer.target, s2, context = 'model.scatterer')

    s3 = Sphere(center = (6e-6, 5.67e-6, 10e-6), n = 1.6+1e-4j, r = 8.5e-7)

#    assert_obj_close(model.make_scatterer((6e-6, 10e-6, 1.6)), s3, context = 'make_scatterer()')
    
    # model.make_scatterer

@with_setup(setup=setup_optics, teardown=teardown_optics)
@attr('fast')
def test_cost_func():
    
    parameters = [Parameter(name='x', guess=.567e-5, limit = [0.0, 1e-5]),
                  Parameter(name='y', guess=.576e-5, limit = [0, 1e-5]),
                  Parameter(name='z', guess=15e-6, limit = [1e-5, 2e-5]),
                  Parameter(name='r', guess=8.5e-7, limit = [1e-8, 1e-5])]

    def make_scatterer(x, y, z, r):
        return Sphere(n=1.59+1e-4j, r = r, center = (x, y, z))

    param = Parameterization(make_scatterer, parameters)
    model = Model(param, Mie, alpha = Parameter(name='alpha', guess=.6, limit = [.1, 1]))

    theory = Mie(optics, 100)
    holo = theory.calc_holo(Sphere(center = (.567e-5, .576e-5, 15e-6),
                                   r = 8.5e-7, n = 1.59+1e-4j), .6)

    cost_func = model.cost_func(holo)

    cost = cost_func(dict([(p.name, p.guess) for p in model.parameters]))

    assert_allclose(cost, np.zeros_like(cost), atol=1e-10)
    

@attr('fast')
def test_minimizer():
    x = np.arange(-10, 10, .1)
    a = 5.3
    b = -1.8
    c = 3.4
    gold_dict = OrderedDict((('a', a), ('b', b), ('c', c)))
    y = a*x**2 + b*x + c

    # This test does NOT handle scaling correctly -- we would need a Model
    # which knows the parameters to properly handle the scaling/unscaling
    def cost_func(pars):
        a = pars['a']
        b = pars['b']
        c = pars['c']
        return a*x**2 + b*x + c - y

    # test basic usage
    parameters = [Parameter(name='a', guess = 5),
                 Parameter(name='b', guess = -2),
                 Parameter(name='c', guess = 3)]
    minimizer = Nmpfit()
    result, minimization_details = minimizer.minimize(parameters, cost_func)
    assert_obj_close(gold_dict, result)

    # test inadequate specification
    with assert_raises(ParameterSpecficationError):
        minimizer.minimize([Parameter(name = 'a')], cost_func)

    # now test limiting minimizer iterations
    minimizer = Nmpfit(maxiter=1)
    try:
        result, minimization_details = minimizer.minimize(parameters, cost_func)
    except MinimizerConvergenceFailed as cf: # the fit shouldn't converge
        result, minimization_details = cf.result, cf.details
    assert_equal(minimization_details.niter, 2) # there's always an offset of 1

    # now test parinfo argument passing
    parameters2 = [Parameter(name='a', guess = 5, mpside = 2),
                   Parameter(name='b', guess = -2, limit = [-4, 4.]),
                   Parameter(name='c', guess = 3, step = 1e-4, mpmaxstep = 2., 
                             limit = [0., 12.])]
    minimizer = Nmpfit()
    result2, details2, parinfo = minimizer.minimize(parameters2, cost_func, 
                                                    debug = True)
    assert_equal(parinfo[0]['mpside'], 2)
    assert_equal(parinfo[2]['limits'], np.array([0., 12.])/3.)
    assert_allclose(parinfo[2]['step'], 1e-4/3.)
    assert_equal(parinfo[2]['limited'], [True, True])
    assert_obj_close(gold_dict, result2)
    

@attr('fast')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_serialization():
    par_s = Sphere(center = (par(.567e-5, [0, 1e-5]), par(.576e-6, [0, 1e-5]),
                                                           par(15e-6, [1e-5,
                                                                       2e-5])),
                   r = par(8.5e-7, [1e-8, 1e-5]), n = par(1.59, [1,2]))

    alpha = par(.6, [.1, 1], 'alpha')

    mie = Mie(optics, 100)

    model = Model(par_s, Mie, alpha=alpha)

    holo = mie.calc_holo(model.scatterer.guess, model.alpha.guess)

    result = fit(model, holo)

    temp = tempfile.NamedTemporaryFile()
    hp.io.save(temp, result)

    temp.flush()
    temp.seek(0)
    
    loaded = hp.io.yaml_io.load(temp)

    assert_obj_close(result, loaded)


@attr('fast')
def test_complex_parameter():
    # target
    a = 3.3
    b = 2.2 + 3.2j
    c = -1.9j
    x = np.arange(-5, 5, 0.01)
    y = a * x + b * x**2 + c * x**3

    # Need to create a minimal Model to unpack complex parameters

    # case 1: vary real parts of a & b only, hold c fixed
    params = [par(3., [0., 10.], 'a'),
              ComplexParameter(real = par(2., [0., 10.]), imag = 3.2, 
                               name = 'b'), 
              ComplexParameter(real = 0., imag = par(-1.9, -1.9), 
                               name = 'c')]

    def cost_func1(values):
        # proto-model should handle unscaling
        a, b = values.values()
        c = params[2].imag.limit * 1j
        return np.abs(y - (a * x + b * x**2 + c * x**3))

    real_params = [par(3., [0., 10.], 'a'),
                   par(2., [0., 10.], 'b_real')]
    minimizer = Nmpfit()
    result, details = minimizer.minimize(real_params, cost_func1)

'''
    # case 1: vary real parts of a & b only, hold c fixed

    params = [par(3., [0., 10.], 'a'),
              par(2., [0., 10.], 'b') + 3.2j, 
              par(-1.9j, -1.9j, 'c')]

    def cost_func1(a, b):
        a = params[0].unscale(a)
        # hack should be unnecessary in real make_scatterer
        # Model should handle this
        b_real = params[1].real.unscale(b.real)
        b_imag = params[1].imag.unscale(b.imag)
        b = b_real + b_imag * 1j
        
        value = a * x + b * x**2 + c * x**3
        return np.abs(y - value)

    minimizer = Nmpfit()
    result, details = minimizer.minimize(params, cost_func1)

    assert_allclose(result, (a, b.real))

    # case 2: vary a, fix b.real, vary b.imag, vary c
    params = [par(3.5, [-2., 12.], 'a'),
              2.2 + par(3.j, [0.j, 10.j], 'b'),
              par(-2.j, [-10.j, 0j], 'c')]

    def cost_func2(a, b, c):
        a = params[0].unscale(a)
        b_real = params[1].real.unscale(b.real)
        b_imag = params[1].imag.unscale(b.imag)
        b = b_real + b_imag * 1j
        c = params[2].imag.unscale(c)

    result, details = minimizer.minimize(params, cost_func2)
    assert_allclose(result, (a, b.imag, c.imag))

    # case 3: vary everything
    params = [par(3.5, [-2., 12.], 'a'),
              par(2., [0., 10.], 'b') + par(3.j, [0.j, 10.j], 'b'),
              par(-2.j, [-10.j, 0.j], 'c')]
    
    result, details = minimizer.minimize(params, cost_func2) # same cost func
    assert_allclose(result, (a, b.real, b.imag, c.imag))

    # case 4: fix everything
    params = [par(3.3, 3.3, 'a'),
              par(2.2, 2.2, 'b') + par(3.2j, 3.2j, 'b'),
              par(-1.9j, -1.9j, 'c')]

    result, details = minimizer.minimize(params, cost_func2)

    # case 5: incorrect name specification
    with assert_raises(ParameterSpecificationError):
        params = [par(3.5, [-2., 12.], 'a'),
                  par(2., [0., 10.], 'b') + par(3.j, [0.j, 10.j], 'smorrebrod'),
                  par(-2.j, [-10.j, 0.j], 'c')]
'''

from scatterpy.tests.test_dda import missing_dependencies

@dec.skipif(missing_dependencies(), 'a-dda not installed')
@attr('slow')
def test_dda_fit():
    from scatterpy.theory import DDA
    s = Sphere(n = 1.59, r = .2, center = (5, 5, 5))
    o = hp.Optics(wavelen = .66, index=1.33, pixel_scale=.1)

    mie = Mie(o, 100)

    h = mie.calc_holo(s)


    def in_sphere(r):
        def test(point):
            point = np.array(point)
            return (point**2).sum() < r**2
        return test


    def make_scatterer(r, x, y, z):
        return ScattererByFunction(in_sphere(r), s.n, [[-.3, .3],[-.3,.3],[-.3,.3]], (x, y, z))


    parameters = [par(.18, [.1, .3], name='r', step=.1), par(5, [4, 6], 'x'),
                  par(5, [4,6], 'y'), par(5.2, [4, 6], 'z')]

    p = Parameterization(make_scatterer, parameters)

    model = Model(p, DDA)

    res = fit(model, h)

    assert_equal(res.parameters, OrderedDict([('r', 0.2003609439787491), ('x', 5.0128083665603995), ('y', 5.0125252883133617), ('z', 4.9775097284878775)]))

