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

import os
from collections import OrderedDict
import tempfile

import numpy as np

import holopy as hp

from nose.tools import with_setup
from nose.plugins.attrib import attr
from numpy.testing import assert_allclose, assert_equal, assert_approx_equal
from scatterpy.theory import Mie
from scatterpy.scatterer import Sphere

from holopy.analyze.fit_new import Parameter, Model, fit, Minimizer
from common import assert_parameters_allclose, assert_obj_close


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

@attr('medium')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_fit_mie_single():
    path = os.path.abspath(hp.__file__)
    path = os.path.join(os.path.split(path)[0],'tests', 'exampledata')
    holo = hp.process.normalize(hp.load(os.path.join(path, 'image0001.npy'),
                                        optics=optics))

    parameters = [Parameter(name='x', guess=.567e-5, limit = [0.0, 1e-5]),
                  Parameter(name='y', guess=.576e-5, limit = [0, 1e-5]),
                  Parameter(name='z', guess=15e-6, limit = [1e-5, 2e-5]),
                  Parameter(name='r', guess=8.5e-7, limit = [1e-8, 1e-5]),
                  Parameter(name='n', guess=1.59, limit = [1, 2]),
                  Parameter(name='alpha', guess=.6, limit = [.1, 1])]

    
    def make_scatterer(x, y, z, r, n):
        return Sphere(n=n+1e-4j, r = r, center = (x, y, z))

    model = Model(parameters, Mie, make_scatterer=make_scatterer)

    result = fit(model, holo)

    assert_parameters_allclose(result.scatterer, gold_single)
    assert_approx_equal(result.alpha, gold_alpha, significant=4)
    assert_equal(model, result.model)

@attr('fast')
def test_parameter():
    par = Parameter(name='x', guess=.567e-5, limit = [0.0, 1e-5])
    assert_equal(1e-6, par.unscale(par.scale(1e-6)))


@attr('fast')
def test_model():
    parameters = [Parameter(name='x', guess=.567e-5, limit = [0.0, 1e-5]),
                  Parameter(name='y', guess=.576e-5, limit = [0, 1e-5]),
                  Parameter(name='z', guess=15e-6, limit = [1e-5, 2e-5]),
                  Parameter(name='r', guess=8.5e-7, limit = [1e-8, 1e-5]),
                  Parameter(name='alpha', guess=.6, limit = [.1, 1])]

    def make_scatterer(x, y, z, r):
        return Sphere(n=1.59+1e-4j, r = r, center = (x, y, z))

    model = Model(parameters, Mie, make_scatterer=make_scatterer)

    x, y, z, r = 1, 2, 3, 1
    s = model.make_scatterer(x, y, z, r)

    assert_parameters_allclose(s, Sphere(center=(x, y, z), n=(1.59+0.0001j),
                                         r=r))

    

@with_setup(setup=setup_optics, teardown=teardown_optics)
@attr('fast')
def test_cost_func():
    
    parameters = [Parameter(name='x', guess=.567e-5, limit = [0.0, 1e-5]),
        Parameter(name='y', guess=.576e-5, limit = [0, 1e-5]),
        Parameter(name='z', guess=15e-6, limit = [1e-5, 2e-5]),
        Parameter(name='r', guess=8.5e-7, limit = [1e-8, 1e-5]),
        Parameter(name='alpha', guess=.6, limit = [.1, 1])]

    def make_scatterer(x, y, z, r):
        return Sphere(n=1.59+1e-4j, r = r, center = (x, y, z))

    model = Model(parameters, Mie, make_scatterer=make_scatterer)


    holo = Mie(optics, 100).calc_holo(Sphere(center = (.567e-5, .576e-5, 15e-6),
                                             r = 8.5e-7, n = 1.59+1e-4j), .6)

    cost_func = model.cost_func(holo)

    cost = cost_func([p.scale(p.guess) for p in parameters])

    assert_allclose(cost, np.zeros_like(cost), atol=1e-10)


    

@attr('fast')
def test_minimizer():
    x = np.arange(-10, 10, .1)
    a = 5.3
    b = -1.8
    c = 3.4
    y = a*x**2 + b*x + c

    def cost_func(par_values):
        a, b, c = par_values
        return a*x**2 + b*x + c - y

    parameters = [Parameter(name='a', guess = 5),
                 Parameter(name='b', guess = -2),
                 Parameter(name='c', guess = 3)]

    minimizer = Minimizer()

    result, converged, minimization_details = minimizer.minimize(parameters,
                                                                 cost_func)

    assert_allclose([a, b, c], result)

    assert_equal(converged, True)

@attr('fast')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_serialization():
    parameters = [Parameter(name='x', guess=.567e-5, limit = [0.0, 1e-5]),
                  Parameter(name='y', guess=.576e-5, limit = [0, 1e-5]),
                  Parameter(name='z', guess=15e-6, limit = [1e-5, 2e-5]),
                  Parameter(name='r', guess=8.5e-7, limit = [1e-8, 1e-5]),
                  Parameter(name='n', guess=1.59, limit = [1, 2]),
                  Parameter(name='alpha', guess=.6, limit = [.1, 1])]
    
    def make_scatterer(x, y, z, r, n):
        return Sphere(n=n+1e-4j, r = r, center = (x, y, z))

    mie = Mie(optics, 100)

    model = Model(parameters, Mie, make_scatterer=make_scatterer)

    holo = mie.calc_holo(model.make_scatterer_from_par_values([p.guess for p in
                                                               parameters]),
                                                               parameters[-1].guess)

    result = fit(model, holo)

    temp = tempfile.NamedTemporaryFile()
    hp.io.save(temp, result)

    temp.flush()
    temp.seek(0)
    
    loaded = hp.io.yaml_io.load(temp)

    # manually put the make_scatterer function back in because save/load
    # currently does not handle them correctly.  This is a BUG, but not an easy
    # one to fix
    loaded.model.make_scatterer = make_scatterer

    assert_obj_close(result, loaded)
