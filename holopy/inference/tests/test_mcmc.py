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
