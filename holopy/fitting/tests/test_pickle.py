from holopy.core.tests.common import assert_pickle_roundtrip, get_example_data

from holopy.fitting.mcmc import Emcee
from holopy.fitting.noise_model import AlphaModel
from holopy.fitting import prior
from holopy.fitting import Model
from holopy.scattering.scatterer import Sphere
from holopy.scattering.theory import Mie

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
