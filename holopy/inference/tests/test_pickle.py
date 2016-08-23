from holopy.core.tests.common import assert_pickle_roundtrip, get_example_data

from holopy.inference.mcmc import Emcee
from holopy.inference.noise_model import AlphaModel
from holopy.inference.timeseries import TimeIndependent, TimeSeriesAlphaModel
from holopy.inference import prior
from holopy.scattering.scatterer import Sphere
from holopy.scattering.theory import Mie

import numpy as np

def test_prior():
    g = prior.Gaussian(1, 1)
    assert_pickle_roundtrip(g)
    assert_pickle_roundtrip(g.lnprob)

def test_AlphaModelholo_likelihood():
    holo = get_example_data('image0001.yaml')
    s = Sphere(prior.Gaussian(.5, .1), prior.Gaussian(1.6, .1),
               (prior.Gaussian(5, 1), prior.Gaussian(5, 1), prior.Gaussian(5, 1)))
    model = AlphaModel(s, Mie, alpha = prior.Gaussian(.7, .1), noise_sd=.01)
    assert_pickle_roundtrip(model)


def test_emcee():
    holo = get_example_data('image0001.yaml')
    s = Sphere(prior.Gaussian(.5, .1), prior.Gaussian(1.6, .1),
               (prior.Gaussian(5, 1), prior.Gaussian(5, 1), prior.Gaussian(5, 1)))
    model = AlphaModel(s, Mie, alpha=prior.Gaussian(.7, .1), noise_sd=.01)
    emcee = Emcee(model, holo)
    assert_pickle_roundtrip(emcee)

def test_TimeSeriesAlphaModel():
    n = TimeIndependent(prior.Gaussian(5, .5))
    assert_pickle_roundtrip(n)
    st = Sphere(n=n, r=TimeIndependent(prior.BoundedGaussian(1.6,.1, 0, np.inf)), center=(prior.Gaussian(10, 1), prior.Gaussian(10, 1), prior.BoundedGaussian(1.6, .1, 0, np.inf)))
    assert_pickle_roundtrip(st)
    noise_sd = .1
    mt = TimeSeriesAlphaModel(st, Mie, noise_sd, alpha=prior.Uniform(0, 1), n_frames=2)
    assert_pickle_roundtrip(mt)
