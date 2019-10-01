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
from numpy.testing import assert_raises

from holopy.core import detector_grid, update_metadata
from holopy.core.tests.common import assert_equal, assert_obj_close
from holopy.scattering.theory import Mie
from holopy.scattering.scatterer import Sphere, Spheres
from holopy.scattering.scatterer.scatterer import _interpret_parameters
from holopy.scattering.errors import MissingParameter
from holopy.core.tests.common import assert_read_matches_write
from holopy.scattering.calculations import calc_holo
from holopy.inference import prior, AlphaModel, ExactModel


@attr('fast')
def test_ComplexPar():
    # complex parameter
    def makeScatterer(n):
        n**2
        return fake_sph

    parm = Sphere(n=prior.ComplexPrior(real=prior.Uniform(1.58,1.59), imag=.001))
    model = AlphaModel(parm, alpha=prior.Uniform(.6, 1, .7))
    assert_equal(model.parameters['n.real'].name, 'n.real')
    interpreted_pars = {'alpha':.7, 'n':{'real':1.585}}
    assert_equal(_interpret_parameters(model.parameters), interpreted_pars)


@attr('fast')
def test_multidim():
    par_s = Sphere(
        n={'r': prior.Uniform(-1,1), 'g': 0, 'b': prior.Gaussian(0,1),'a':0},
        r=xr.DataArray(
            [prior.Gaussian(0,1), prior.Uniform(-1,1), 0, 0],
            dims='alph', coords={'alph': ['a', 'b', 'c', 'd']}),
            center=[prior.Uniform(-1, 1), 0, 0])
    params = {'n:r': 3, 'n:g': 4, 'n:b': 5, 'n:a': 6, 'r:a': 7, 'r:b': 8,
              'r:c': 9, 'r:d': 10, 'center.0': 7, 'center.1': 8,
              'center.2': 9}
    out_s = Sphere(
        n={'r':3, 'g':0, 'b':5, 'a':0},
        r={'a':7, 'b':8, 'c':0, 'd':0}, center=[7, 0, 0])
    assert_obj_close(par_s.from_parameters(params), out_s)

    m = ExactModel(out_s, np.sum)
    parletters = {'r':prior.Uniform(-1,1),'g':0,'b':prior.Gaussian(0,1),'a':0}
    parcount = xr.DataArray([prior.Gaussian(0,1),prior.Uniform(-1,1),0,0],dims='numbers',coords={'numbers':['one', 'two', 'three', 'four']})

    m._use_parameters({'letters':parletters, 'count':parcount})
    expected_params = {'letters:r':prior.Uniform(-1,1, 0, 'letters:r'),'letters:b':prior.Gaussian(0,1,'letters:b'),'count:one':prior.Gaussian(0,1, 'count:one'),'count:two':prior.Uniform(-1,1, 0,'count:two')}
    assert_equal(m.parameters, expected_params)


@attr('fast')
def test_pullingoutguess():
    g = Sphere(center = (prior.Uniform(0, 1e-5, guess=.567e-5),
                   prior.Uniform(0, 1e-5, .567e-5), prior.Uniform(1e-5, 2e-5, 15e-6)),
         r = prior.Uniform(1e-8, 1e-5, 8.5e-7), n = prior.ComplexPrior(prior.Uniform(1, 2, 1.59),1e-4))

    model = ExactModel(g, calc_holo)

    s = Sphere(center = [.567e-5, .567e-5, 15e-6], n = 1.59 + 1e-4j, r = 8.5e-7)

    assert_equal(s.n, model.scatterer.guess.n)
    assert_equal(s.r, model.scatterer.guess.r)
    assert_equal(s.center, model.scatterer.guess.center)

    g = Sphere(center = (prior.Uniform(0, 1e-5, guess=.567e-5),
                   prior.Uniform(0, 1e-5, .567e-5), prior.Uniform(1e-5, 2e-5, 15e-6)),
         r = prior.Uniform(1e-8, 1e-5, 8.5e-7), n = 1.59 + 1e-4j)

    model = ExactModel(g, calc_holo)

    s = Sphere(center = [.567e-5, .567e-5, 15e-6], n = 1.59 + 1e-4j, r = 8.5e-7)

    assert_equal(s.n, model.scatterer.guess.n)
    assert_equal(s.r, model.scatterer.guess.r)
    assert_equal(s.center, model.scatterer.guess.center)


@attr('fast')
def test_find_noise():
    noise=0.5
    s = Sphere(n=prior.Uniform(1.5, 1.7), r=2, center=[1,2,3])
    data_base = detector_grid(10, spacing=0.5)
    data_noise = update_metadata(data_base, noise_sd=noise)
    model_u = AlphaModel(s, alpha=prior.Uniform(0.7,0.9))
    model_g = AlphaModel(s, alpha=prior.Gaussian(0.8, 0.1))
    pars = {'n':1.6, 'alpha':0.8}
    assert_equal(model_u._find_noise(pars, data_noise), noise)
    assert_equal(model_g._find_noise(pars, data_noise), noise)
    assert_equal(model_u._find_noise(pars, data_base), 1)
    assert_raises(MissingParameter, model_g._find_noise, pars, data_base)
    pars.update({'noise_sd':noise})
    assert_equal(model_g._find_noise(pars, data_base), noise)


@attr('fast')
def test_io():
    model = ExactModel(Sphere(1), calc_holo)
    assert_read_matches_write(model)

    model = ExactModel(Sphere(1), calc_holo, theory=Mie(False))
    assert_read_matches_write(model)

