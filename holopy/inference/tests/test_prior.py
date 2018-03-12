# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, Ryan McGorty, Anna Wang, Solomon Barkley
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




from numpy.testing import assert_raises, assert_equal
import numpy as np

from holopy.inference import prior
from holopy.inference.result import UncertainValue
from holopy.core.tests.common import assert_obj_close
#GOLD:log(sqrt(0.5/pi))-1/2
gold_sigma=-1.4189385332

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
    assert_obj_close(g.lnprob(0),gold_sigma)

def test_updated():
    p=prior.BoundedGaussian(1,2,-1,2)    
    d=UncertainValue(1,0.5,1)
    u=prior.updated(p,d)
    assert_equal(u.guess,1)
    assert_obj_close(u.lnprob(0),gold_sigma)

def test_prior_math():
    u = prior.Uniform(1,2)
    g = prior.Gaussian(1,2)
    b = prior.BoundedGaussian(1,2,0,3)

    assert_equal(u+1, prior.Uniform(2,3))
    assert_equal(1+u, prior.Uniform(2,3))
    assert_equal(-u, prior.Uniform(-2,-1))
    assert_equal(1-u, prior.Uniform(-1,0))
    assert_equal(u-1, prior.Uniform(0,1))

    assert_equal(g+1., prior.Gaussian(2,2.))
    assert_equal(-g, prior.Gaussian(-1,2))
    assert_equal(b+1., prior.BoundedGaussian(2.,2,1.,4.))
    assert_equal(-b, prior.BoundedGaussian(-1,2,-3,0))

    assert_equal(g+g, prior.Gaussian(2,np.sqrt(8)))
    with assert_raises(TypeError):
        u+u
    with assert_raises(TypeError):
        g+b
    with assert_raises(TypeError):
        g+[0,1]
    with assert_raises(TypeError):
        g+np.array([0,1])

