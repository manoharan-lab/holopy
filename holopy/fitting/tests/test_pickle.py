from holopy.core.tests.common import assert_pickle_roundtrip, get_example_data

from holopy.fitting.mcmc import GaussianPrior, HologramLikelihood, Emcee
from holopy.fitting import Model
from holopy.scattering.scatterer import Sphere
from holopy.scattering.theory import Mie

def test_prior():
    g = GaussianPrior(1, 1)
    assert_pickle_roundtrip(g)
    assert_pickle_roundtrip(g.lnprob)

def test_holo_likelihood():
    holo = get_example_data('image0001.yaml')
    s = Sphere(GaussianPrior(.5, .1), GaussianPrior(1.6, .1),
               (GaussianPrior(5, 1), GaussianPrior(5, 1), GaussianPrior(5, 1)))
    model = Model(s, Mie.calc_holo, alpha = GaussianPrior(.7, .1))
    hl = HologramLikelihood(holo, model, .01, .1)
    assert_pickle_roundtrip(hl)


def test_emcee():
    holo = get_example_data('image0001.yaml')
    s = Sphere(GaussianPrior(.5, .1), GaussianPrior(1.6, .1),
               (GaussianPrior(5, 1), GaussianPrior(5, 1), GaussianPrior(5, 1)))
    model = Model(s, Mie.calc_holo, alpha = GaussianPrior(.7, .1))
    emcee = Emcee(model, holo, .01)
    assert_pickle_roundtrip(emcee)
