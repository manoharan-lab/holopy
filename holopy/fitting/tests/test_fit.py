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

from nose.tools import with_setup, nottest
from nose.plugins.attrib import attr
from numpy.testing import assert_equal, assert_approx_equal, assert_raises, dec, assert_allclose
from ...scattering.scatterer import Sphere, SphereCluster, ScattererByFunction
from ...scattering.theory import Mie, Multisphere, DDA
from ...core import Optics, ImageTarget, load, save, DataTarget
from ...core.process import normalize
from .. import fit, Parameter, ComplexParameter, par, Parametrization, Model
from ..minimizer import Nmpfit
from ..errors import ParameterSpecificationError, MinimizerConvergenceFailed


from ...scattering.tests.test_dda import dda_external_not_available
from ...core.tests.common import assert_obj_close, get_example_data, assert_parameters_allclose


def setup_optics():
    # set up optics class for use in several test functions
    global optics
    wavelen = 658e-9
    polarization = [0., 1.0]
    divergence = 0
    pixel_scale = [.1151e-6, .1151e-6]
    index = 1.33
    
    optics = Optics(wavelen=wavelen, index=index,
                              pixel_scale=pixel_scale,
                              polarization=polarization,
                              divergence=divergence)
    
def teardown_optics():
    global optics
    del optics

gold_single = OrderedDict((('center[0]', 5.534e-6),
               ('center[1]', 5.792e-6),
               ('center[2]', 1.415e-5),               ('n.imag', 1e-4),
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
    holo = normalize(get_example_data('image0001.npy', optics))

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

    assert_parameters_allclose(result.parameters, gold_single, rtol = 1e-3)
    assert_equal(model, result.model)

    
@attr('medium')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_fit_mie_par_scatterer():
    holo = normalize(get_example_data('image0001.npy', optics))

    s = Sphere(center = (par(guess=.567e-5, limit=[0,1e-5]),
                         par(.567e-5, (0, 1e-5)), par(15e-6, (1e-5, 2e-5))),
               r = par(8.5e-7, (1e-8, 1e-5)), 
               n = ComplexParameter(par(1.59, (1,2)), 1e-4j))

    
    model = Model(s, Mie.calc_holo, alpha = par(.6, [.1,1]))

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
    holo = normalize(get_example_data('image0001.npy', optics=optics))

    
    s = Sphere(center = (par(guess=.567e-5, limit=[0,1e-5]),
                         par(.567e-5, (0, 1e-5)), par(15e-6, (1e-5, 2e-5))),
               r = par(8.5e-7, (1e-8, 1e-5)), n = ComplexParameter(par(1.59, (1,2)),1e-4j))

    
    model = Model(s, Mie.calc_holo, metadata=DataTarget(use_random_fraction = .1), alpha = par(.6, [.1,1]))

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
    with assert_raises(ParameterSpecificationError):
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

    target = ImageTarget(shape = 100, optics = optics) 

    model = Model(par_s, Mie.calc_holo, alpha=alpha)

    holo = Mie.calc_holo(model.scatterer.guess, target, model.alpha.guess)

    result = fit(model, holo)

    temp = tempfile.NamedTemporaryFile()
    save(temp, result)

    temp.flush()
    temp.seek(0)
    
    loaded = load(temp)

    assert_obj_close(result, loaded)


@dec.skipif(dda_external_not_available(), 'a-dda not installed')
@attr('slow')
def test_dda_fit():
    s = Sphere(n = 1.59, r = .2, center = (5, 5, 5))
    o = Optics(wavelen = .66, index=1.33, pixel_scale=.1)

    target = ImageTarget(optics = o, shape = 100)

    h = Mie.calc_holo(s, target)


    def in_sphere(r):
        def test(point):
            point = np.array(point)
            return (point**2).sum() < r**2
        return test


    def make_scatterer(r, x, y, z):
        return ScattererByFunction(in_sphere(r), s.n, [[-.3, .3],[-.3,.3],[-.3,.3]], (x, y, z))


    parameters = [par(.18, [.1, .3], name='r', step=.1), par(5, [4, 6], 'x'),
                  par(5, [4,6], 'y'), par(5.2, [4, 6], 'z')]

    p = Parametrization(make_scatterer, parameters)

    model = Model(p, DDA.calc_holo)

    res = fit(model, h)

    assert_parameters_allclose(res.parameters, OrderedDict([('r',
    0.2003609439787491), ('x', 5.0128083665603995), ('y', 5.0125252883133617),
    ('z', 4.9775097284878775)]), rtol=1e-3)

