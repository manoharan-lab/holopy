# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
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


import tempfile

import numpy as np
import xarray as xr
from collections import OrderedDict

from nose.plugins.attrib import attr
from holopy.core.tests.common import assert_equal, assert_obj_close
from holopy.scattering.theory import Mie
from holopy.scattering.scatterer import Sphere, Spheres
from holopy.fitting.model import Model
from holopy.fitting import ComplexParameter, Parameter as par
from holopy.core.tests.common import assert_read_matches_write
from holopy.scattering.calculations import calc_holo
from holopy.inference import prior


@attr('fast')
def test_Tying():
    # tied parameters
    n1 = par(1.59)
    sc = Spheres(
        [Sphere(n=n1, r=par(0.5e-6), center=np.array([10., 10., 20.])),
         Sphere(n=n1, r=par(0.5e-6), center=np.array([9., 11., 21.]))])
    model = Model(sc, calc_holo, alpha=par(.7, [.6, 1]))
    assert_equal(model.parameters['n'].guess, 1.59)
    assert_equal(model.parameters['0:r'].guess, 5e-7)
    assert_equal(len(model.parameters), 4)


@attr('fast')
def test_ComplexPar():
    # complex parameter
    def makeScatterer(n):
        n**2
        return fake_sph

    parm = Sphere(n=ComplexParameter(real=par(1.58), imag=par(.001), name='n'))
    model = Model(parm, calc_holo, alpha=par(.7, [.6, 1]))
    assert_equal(model.parameters['n.real'].name, 'n.real')
    assert_equal(model.parameters['n.imag'].name, 'n.imag')

def test_multidim():
    par_s = Sphere(
        n={'r': par(0,[-1,1]), 'g': par(0,0), 'b': prior.Gaussian(0,1),'a':0},
        r=xr.DataArray(
            [prior.Gaussian(0,1), par(0,[-1,1]), par(0,0),0],
            dims='alph', coords={'alph': ['a', 'b', 'c', 'd']}),
            center=[par(0, [-1, 1]), 0, 0])
    params = {'n_r': 3, 'n_g': 4, 'n_b': 5, 'n_a': 6, 'r_a': 7, 'r_b': 8,
              'r_c': 9, 'r_d': 10, 'center.0': 7, 'center.1': 8,
              'center.2': 9}
    out_s = Sphere(
        n={'r':3, 'g':0, 'b':5, 'a':0},
        r={'a':7, 'b':8, 'c':0, 'd':0}, center=[7, 0, 0])
    assert_obj_close(par_s.from_parameters(params), out_s)

    m = Model(out_s, np.sum)
    parletters = {'r':par(0,[-1,1]),'g':par(0,0),'b':prior.Gaussian(0,1),'a':0}
    parcount = xr.DataArray([prior.Gaussian(0,1),par(0,[-1,1]),par(0,0),0],dims='numbers',coords={'numbers':['one', 'two', 'three', 'four']})

    m._use_parameters({'letters':parletters, 'count':parcount})
    expected_params = {'letters_r':par(0,[-1,1], 'letters_r'),'letters_b':prior.Gaussian(0,1, 'letters_b'),'count_one':prior.Gaussian(0,1, 'count_one'),'count_two':par(0,[-1,1], 'count_two')}
    assert_equal(m.parameters, expected_params)


def test_pullingoutguess():
    g = Sphere(center = (par(guess=.567e-5, limit=[0,1e-5]),
                   par(.567e-5, (0, 1e-5)), par(15e-6, (1e-5, 2e-5))),
         r = par(8.5e-7, (1e-8, 1e-5)), n = ComplexParameter(par(1.59, (1,2)),1e-4))

    model = Model(g, calc_holo)

    s = Sphere(center = [.567e-5, .567e-5, 15e-6], n = 1.59 + 1e-4j, r = 8.5e-7)

    assert_equal(s.n, model.scatterer.guess.n)
    assert_equal(s.r, model.scatterer.guess.r)
    assert_equal(s.center, model.scatterer.guess.center)

    g = Sphere(center = (par(guess=.567e-5, limit=[0,1e-5]),
                   par(.567e-5, (0, 1e-5)), par(15e-6, (1e-5, 2e-5))),
         r = par(8.5e-7, (1e-8, 1e-5)), n = 1.59 + 1e-4j)

    model = Model(g, calc_holo)

    s = Sphere(center = [.567e-5, .567e-5, 15e-6], n = 1.59 + 1e-4j, r = 8.5e-7)

    assert_equal(s.n, model.scatterer.guess.n)
    assert_equal(s.r, model.scatterer.guess.r)
    assert_equal(s.center, model.scatterer.guess.center)


def test_io():
    model = Model(Sphere(par(1)), calc_holo)
    assert_read_matches_write(model)

    model = Model(Sphere(par(1)), calc_holo, theory=Mie(False))
    assert_read_matches_write(model)
