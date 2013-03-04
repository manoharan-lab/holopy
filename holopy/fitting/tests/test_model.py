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

import numpy as np

from nose.plugins.attrib import attr
from numpy.testing import assert_equal
from ...scattering.theory import Mie
from ...scattering.scatterer import Sphere, Spheres, Scatterer
from .. import fit, par, Model, ComplexParameter, Parametrization
from ...core.tests.common import assert_obj_close, get_example_data

@attr('fast')
def test_naming():

    #parameterizing with fixed params
    def makeScatterer(n,m):
        n**2+m
        return fake_sph
    parm = Parametrization(makeScatterer, [par(limit=4),par(2, [1,5])])
    
    assert_equal(parm._fixed_params,{None: 4})
    
@attr('fast')
def test_Get_Alpha():

    #checking get_alpha function
    sc = Spheres([Sphere(n = 1.58, r = par(0.5e-6), center = np.array([10., 10., 20.])),
              Sphere(n = 1.58, r = par(0.5e-6), center = np.array([9., 11., 21.]))])
    model = Model(sc, Mie.calc_holo, alpha = par(.7,[.6,1]))

    sc = Spheres([Sphere(n = 1.58, r = par(0.5e-6), center = np.array([10., 10., 20.])),
              Sphere(n = 1.58, r = par(0.5e-6), center = np.array([9., 11., 21.]))])
    model2 = Model(sc, Mie.calc_holo)
              
    assert_equal(model.get_alpha(model.parameters).guess, 0.7)
    assert_equal(model.get_alpha(model.parameters).name, 'alpha')
    assert_equal(model2.get_alpha(model2.parameters), 1.0)
    
    
@attr('fast')
def test_Tying():

    #tied parameters
    n1 = par(1.59)
    sc = Spheres([Sphere(n = n1, r = par(0.5e-6), center = np.array([10., 10., 20.])),
              Sphere(n = n1, r = par(0.5e-6), center = np.array([9., 11., 21.]))])
    model = Model(sc, Mie.calc_holo, alpha = par(.7,[.6,1]))
              
    assert_equal(model.parameters[0].guess, 1.59)
    assert_equal(model.parameters[1].guess, 5e-7)
    assert_equal(len(model.parameters),4)
    
    
@attr('fast')
def test_ComplexPar():

    #complex parameter
    def makeScatterer(n):
        n**2
        return fake_sph
        
    parm = Parametrization(makeScatterer, [ComplexParameter(real = par(1.58),imag = par(.001), name='n')])
    model = Model(parm, Mie.calc_holo, alpha = par(.7,[.6,1]))
    
    assert_equal(model.parameters[0].name,'n.real')
    assert_equal(model.parameters[1].name,'n.imag')


def test_pullingoutguess():
    g = Sphere(center = (par(guess=.567e-5, limit=[0,1e-5]),
                   par(.567e-5, (0, 1e-5)), par(15e-6, (1e-5, 2e-5))),
         r = par(8.5e-7, (1e-8, 1e-5)), n = ComplexParameter(par(1.59, (1,2)),1e-4))

    model = Model(g, Mie.calc_holo)

    s = Sphere(center = [.567e-5, .567e-5, 15e-6], n = 1.59 + 1e-4j, r = 8.5e-7)

    assert_equal(s.n, model.scatterer.guess.n)
    assert_equal(s.r, model.scatterer.guess.r)
    assert_equal(s.center, model.scatterer.guess.center)

    g = Sphere(center = (par(guess=.567e-5, limit=[0,1e-5]),
                   par(.567e-5, (0, 1e-5)), par(15e-6, (1e-5, 2e-5))),
         r = par(8.5e-7, (1e-8, 1e-5)), n = 1.59 + 1e-4j)

    model = Model(g, Mie.calc_holo)

    s = Sphere(center = [.567e-5, .567e-5, 15e-6], n = 1.59 + 1e-4j, r = 8.5e-7)

    assert_equal(s.n, model.scatterer.guess.n)
    assert_equal(s.r, model.scatterer.guess.r)
    assert_equal(s.center, model.scatterer.guess.center)

