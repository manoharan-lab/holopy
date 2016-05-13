from __future__ import division

from holopy.fitting.fit import CostComputer
from holopy.fitting.model import Model
from holopy.core.holopy_object import HoloPyObject
from holopy.fitting.parameter import Parameter
from holopy.fitting.minimizer import Minimizer
from emcee import PTSampler, EnsembleSampler
from holopy.core.helpers import _ensure_array

from numpy import random
import numpy as np
from scipy import stats

class ProbabilityComputer(HoloPyObject):
    def lnprob(self):
        raise NotImplementedError


class Prior(Parameter):
    pass

class UniformPrior(Prior):
    def __init__(self, lower_bound, upper_bound, name=None):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.name = name
        self._lnprob = np.log(1/(self.upper_bound - self.lower_bound))

    def lnprob(self, p):
        if p < self.lower_bound or p > self.upper_bound:
            return -np.inf
        # For a uniform prior, the value is the same every time, so we precompute it
        return self._lnprob
        # Turns out scipy.stats is noticably slower than doing it ourselves
        #return stats.uniform.logpdf(p, self.lower_bound, self.upper_bound)

    @property
    def interval(self):
        return self.upper_bound - self.lower_bound

    @property
    def guess(self):
        return (self.upper_bound - self.lower_bound)/2

    def sample(self, size=None):
        return random.uniform(self.lower_bound, self.upper_bound, size)

class GaussianPrior(Prior):
    def __init__(self, mu, sd, name=None):
        self.mu = mu
        self.sd = sd
        self.sdsq = sd**2
        self.name=name
        self._lnprob_normalization = -np.log(self.sd * np.sqrt(2*np.pi))

    def lnprob(self, p):
        return self._lnprob_normalization - (p-self.mu)**2/(2*self.sdsq)
        # Turns out scipy.stats is noticably slower than doing it ourselves
        #return stats.norm.logpdf(p, self.mu, self.sd)

    def prob(self, p):
        return stats.norm.pdf(p, self.mu, self.sd)

    @property
    def guess(self):
        return self.mu

    def sample(self, size=None):
        return random.normal(self.mu, self.sd, size=size)


class BoundedGaussianPrior(GaussianPrior):
    # Note: this is not normalized
    def __init__(self, mu, sd, lower_bound=-np.inf, upper_bound=np.inf, name=None):
        if mu < lower_bound or mu > upper_bound:
            raise OutOfBoundsError(mu, lower_bound, upper_bound)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        super(BoundedGaussianPrior, self).__init__(mu, sd, name)

    def lnprob(self, p):
        p = _ensure_array(p)
        if (p < self.lower_bound).any() or (p > self.upper_bound).any():
            return -np.inf
        else:
            return super(BoundedGaussianPrior, self).lnprob(p)


    def sample(self, size=None):
        val = super(BoundedGaussianPrior, self).sample(size)
        # TODO: do something smarter than just clipping
        return np.clip(val, self.lower_bound, self.upper_bound)

        return val


class ScattererPrior(Prior):
    def __init__(self, scatterer):
        self.scatterer = scatterer

    def lnprob(self, par_vals):
        val = 0
        for name, par in self.scatterer.parameters.items():
            val += par.lnprob(par_vals[name])
        return val

    def guess(self):
        return self.scatterer.from_parameters(dict([(p.name, p.guess) for p in self.scatterer.parameters.values()]))


    def sample(self, size=None):
        return self.scatterer.from_parameters(dict([(p.name, p.sample(size)) for p in self.scatterer.parameters.values()]))

    @property
    def parameters(self):
        return self.scatterer.parameters


class BayesianModel(Model):
    def lnprior(self, par_vals):
        return sum([p.lnprob(par_vals[p.name]) for p in self.parameters])

class HologramLikelihood(ProbabilityComputer):
    def __init__(self, data, model, noise_sd, random_subset=None):
        self.data = data
        self.N = data.shape[0] * data.shape[1]
        self.noise_sd = noise_sd
        self.random_subset = random_subset
        self.computer = CostComputer(data, model, random_subset)
        if random_subset is not None:
            self.N = self.computer.selection.size


    def forward(self, par_vals):
        return self.computer._calc(par_vals)

    def lnprob(self, par_vals):
#        I have tried resetting the random subset each time we
#        compute an lnprob, but that appears to lead to instability in
#        the solution, so we will just have to hope that the chance of
#        picking a subset which is not representatative is vanishingly
#        small. Which I think it is. -tgd 2016-05-04
        return (-self.N*np.log(self.noise_sd*np.sqrt(2 * np.pi)) -
                ((self.computer.flattened_difference(par_vals)**2).sum()/
                 (2*self.noise_sd**2)))


class Emcee(HoloPyObject):
    def __init__(self, model, data, noise_sd, nwalkers=50, random_subset=None, threads=None):
        self.parameters = model.parameters
        self.ndim = len(self.parameters)
        self.model = model
        self.likelihood = HologramLikelihood(data, model, noise_sd, random_subset)
        self.nwalkers = nwalkers
        self.threads = threads

    def _pack(self, vals):
        pars = {}
        for par, value in zip(self.parameters, vals):
            pars[par.name] = value
        return pars

    def lnprior(self, vals):
        return self.model.lnprior(self._pack(vals))

    def lnlike(self, vals):
        return self.likelihood.lnprob(self._pack(vals))

    def lnposterior(self, vals):
        lnprior = self.lnprior(vals)
        # prior is sometimes used to forbid thing like negative radius
        # which will fail if you attempt to compute a hologram of, so
        # don't try to compute likelihood where the prior already
        # forbids you to be
        if lnprior == -np.inf:
            return lnprior
        else:
            return lnprior + self.lnlike(vals)

    def make_guess(self):
        return np.vstack([p.sample(size=(self.nwalkers)) for p in self.parameters]).T

    def make_sampler(self):
        return EnsembleSampler(self.nwalkers, self.ndim, self.lnposterior, threads=self.threads)

    def sample(self, n_samples, p0=None):
        sampler = self.make_sampler()
        if p0 is None:
            p0 = self.make_guess()

        sampler.run_mcmc(p0, n_samples)

        return sampler


class PTemcee(Emcee):
    def __init__(self, model, data, noise_sd, nwalkers=20, ntemps=10, random_subset=None, threads=None):
        super(PTemcee, self).__init__(model=model, data=data, noise_sd=noise_sd, nwalkers=nwalkers, random_subset=random_subset, threads=threads)
        self.ntemps = ntemps

    def make_guess(self):
        return np.dstack([p.sample(size=(self.ntemps, self.nwalkers)) for p in self.parameters])

    def make_sampler(self):
        return PTSampler(self.ntemps, self.nwalkers, self.ndim, self.lnlike, self.lnprior, threads=self.threads)
