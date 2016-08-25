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


from __future__ import division

from nose.tools import assert_raises, assert_equal
from numpy.testing import assert_allclose
import numpy as np

from holopy.inference import prior
from holopy.scattering.scatterer import Sphere
from holopy.core.tests.common import assert_obj_close
from holopy.inference.noise_model import AlphaModel

def test_uniform():
    u = prior.Uniform(0, 1)
    assert_equal(u.lnprob(.4), 0)
    assert_equal(u.lnprob(-.1), -np.inf)
    assert_equal(u.lnprob(4), -np.inf)
    assert_equal(u.guess, .5)

def test_bounded_gaussian():
    g = prior.BoundedGaussian(1, 1, 0, 2)
    assert_equal(g.lnprob(-1), -np.inf)
    assert_equal(g.lnprob(3), -np.inf)
    assert_equal(g.guess, 1)

def test_gaussian():
    g = prior.Gaussian(1, 1)
    assert_equal(g.guess, 1)

def test_bayesian_model():
    s = Sphere(r=prior.Gaussian(.5, .1), n=prior.Gaussian(1.6, .1), center=(prior.Gaussian(5, 1), prior.Gaussian(6, 1), prior.Gaussian(7, 1)))
    m = AlphaModel(s, None, alpha=prior.Gaussian(.7, .1), noise_sd=.01)
    assert_equal([p.guess for p in m.parameters], [5, 6, 7, 1.6, .5, .7])

    assert_allclose(m.lnprior({'r':.5, 'n':1.6,'center[0]':5, 'center[1]':6, 'center[2]':7, 'alpha':.7}), np.array([ 1.39412408]))
