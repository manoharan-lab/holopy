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




from numpy.testing import assert_raises, assert_equal, assert_allclose
import numpy as np

from holopy.inference.prior import Gaussian, Uniform
from holopy.inference import prior
from holopy.inference.result import UncertainValue
from holopy.core.tests.common import assert_obj_close
from holopy.scattering.errors import ParameterSpecificationError
#GOLD:log(sqrt(0.5/pi))-1/2
gold_sigma=-1.4189385332

def test_uniform():
    u = Uniform(1, 2)
    assert_equal(u.lnprob(1.4), 0)
    assert_equal(u.lnprob(0.9), -np.inf)
    assert_equal(u.lnprob(4), -np.inf)
    assert_equal(u.guess, 1.5)
    assert_equal(u.interval, 1)
    assert_raises(ParameterSpecificationError, Uniform, 4, 6, 7)
    assert_raises(ParameterSpecificationError, Uniform, 4, 4)

def test_improper_guess():
    improper1 = Uniform(-2, np.inf)
    assert_equal(improper1.guess, -2)
    improper2 = Uniform(-2, np.inf, -1)
    assert_equal(improper2.guess, -1)
    improper3 = Uniform(-np.inf, -2)
    assert_equal(improper3.guess,-2)
    improper4 = Uniform(-np.inf, np.inf)
    assert_equal(improper4.guess, 0)

def test_bounded_gaussian():
    g = prior.BoundedGaussian(1, 1, 0, 2)
    assert_equal(g.lnprob(-1), -np.inf)
    assert_equal(g.lnprob(3), -np.inf)
    assert_equal(g.guess, 1)

def test_gaussian():
    g = Gaussian(1, 1)
    assert_equal(g.guess, 1)
    assert_obj_close(g.lnprob(0),gold_sigma)

def test_complex_prior():
    p1 = prior.ComplexPrior(Uniform(0,2), Gaussian(2,1))
    assert_obj_close(p1.real, Uniform(0,2))
    assert_obj_close(p1.imag, Gaussian(2,1))
    assert_equal(p1.guess, 1+2j)
    assert_obj_close(p1.lnprob(0.5+1j), gold_sigma - np.log(2))
    p2 = prior.ComplexPrior(1, Gaussian(2,1))
    assert_equal(p2.guess, 1+2j)
    p3 = prior.ComplexPrior(Uniform(0,2), 2)
    assert_equal(p3.guess, 1+2j)

def test_scale_factor():
    p1 = Gaussian(3, 1)
    assert_equal(p1.scale_factor, 3)
    p2 = Gaussian(0, 2)
    assert_equal(p2.scale_factor, 2)
    p4 = Uniform(-1, 1, 0)
    assert_equal(p4.scale_factor, 0.2)
    p5 = Uniform(1, 4)
    assert_equal(p5.scale_factor, 2.5)
    p6 = Uniform(0, np.inf)
    assert_equal(p6.scale_factor, 1)
    assert_equal(p2.scale(10), 5)
    assert_equal(p2.unscale(5), 10)

def test_updated():
    p=prior.BoundedGaussian(1,2,-1,2)    
    d=UncertainValue(1,0.5,1)
    u=prior.updated(p,d)
    assert_equal(u.guess,1)
    assert_obj_close(u.lnprob(0),gold_sigma)

def test_prior_math():
    u = Uniform(1,2)
    g = Gaussian(1,2)
    b = prior.BoundedGaussian(1,2,0,3)

    assert_equal(u+1, Uniform(2,3))
    assert_equal(1+u, Uniform(2,3))
    assert_equal(-u, Uniform(-2,-1))
    assert_equal(1-u, Uniform(-1,0))
    assert_equal(u-1, Uniform(0,1))
    assert_equal(2*u, Uniform(2,4))
    assert_equal(u*2, Uniform(2,4))
    assert_equal(u/2, Uniform(0.5,1))
    assert_equal(-1*u, Uniform(-2,-1))
    assert_equal(u*(-1), Uniform(-2,-1))

    assert_equal(g+1., Gaussian(2,2.))
    assert_equal(-g, Gaussian(-1,2))
    assert_equal(b+1., prior.BoundedGaussian(2.,2,1.,4.))
    assert_equal(-b, prior.BoundedGaussian(-1,2,-3,0))
    assert_equal(2*g, Gaussian(2,4))
    assert_equal(g*2, Gaussian(2,4))
    assert_equal(g/2, Gaussian(0.5,1))
    assert_equal(-1*g, Gaussian(-1,2))
    assert_equal(g*(-1), Gaussian(-1,2))

    assert_equal(g+g, Gaussian(2,np.sqrt(8)))
    assert_equal(g+np.array([0,1]),np.array([Gaussian(1,2), Gaussian(2,2)]))
    assert_equal(g*np.array([1,2]),np.array([Gaussian(1,2), Gaussian(2,4)]))

    with assert_raises(TypeError):
        u+u
    with assert_raises(TypeError):
        g+b
    with assert_raises(TypeError):
        g+[0,1]
    with assert_raises(TypeError):
        g*g

def test_generate_guess():
    gold1 = np.array([[-0.091949, 0.270532], [-1.463350, 0.691041],
        [1.081791, 0.220404], [-0.239325, 0.811950], [-0.491129, 0.010526]])
    gold2 = np.array([[-0.045974, 0.535266], [-0.731675, 0.745520],
        [0.540895, 0.510202], [-0.119662, 0.805975], [-0.245564, 0.405263]])
    pars = [Gaussian(0,1), Uniform(0,1,0.8)]
    guess1 = prior.generate_guess(pars, 5, seed=22)
    guess2 = prior.generate_guess(pars, 5, scaling=0.5, seed=22)
    assert_allclose(guess1, gold1, atol=1e-5)
    assert_allclose(guess2, gold2, atol=1e-5)

