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

from nose.plugins.attrib import attr
from nose.tools import with_setup
from numpy.testing import assert_equal, assert_raises, assert_allclose
import numpy as np

from ..parameter import Parameter, ComplexParameter, par
from ..model import Parametrization, Model
from ..errors import GuessOutOfBoundsError
from ..minimizer import Nmpfit
from ...scattering.scatterer import Sphere
from ...scattering.theory import Mie
from ...core.tests.common import assert_obj_close
from ...core import ImageTarget, Optics


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

@attr('fast')
def test_model():
    parameters = [Parameter(name='x', guess=.567e-5, limit = [0.0, 1e-5]),
                  Parameter(name='y', guess=.576e-5, limit = [0, 1e-5]),
                  Parameter(name='z', guess=15e-6, limit = [1e-5, 2e-5]),
                  Parameter(name='r', guess=8.5e-7, limit = [1e-8, 1e-5])]

    def make_scatterer(x, y, z, r):
        return Sphere(n=1.59+1e-4j, r = r, center = (x, y, z))

    param = Parametrization(make_scatterer, parameters)
    model = Model(param, Mie.calc_holo)

    x, y, z, r = 1, 2, 3, 1
    values = {'x': x, 'y': y, 'z': z, 'r': r}
    s = model.scatterer.make_from(values)

    assert_obj_close(s, Sphere(center=(x, y, z), n=1.59+1e-4j, r=r))

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

    param = Parametrization(make_scatterer, parameters)
    model = Model(param, Mie.calc_holo, alpha = Parameter(name='alpha', guess=.6, limit = [.1, 1]))


    target = ImageTarget(100, optics = optics)
    holo = Mie.calc_holo(Sphere(center = (.567e-5, .576e-5, 15e-6),
                                   r = 8.5e-7, n = 1.59+1e-4j), target, scaling=.6)

    cost_func = model.cost_func(holo)

    cost = cost_func(dict([(p.name, p.guess) for p in model.parameters]))

    assert_allclose(cost, np.zeros_like(cost), atol=1e-10)


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
