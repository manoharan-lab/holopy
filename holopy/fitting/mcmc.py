from __future__ import division

from holopy.fitting.fit import CostComputer
from holopy.fitting.model import Model
from holopy.core.holopy_object import HoloPyObject
from holopy.fitting.parameter import Parameter
from holopy.fitting.minimizer import Minimizer
from emcee import PTSampler, EnsembleSampler
from holopy.core.helpers import _ensure_array

from time import time
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

        return EmceeResult(sampler, self.model)


class PTemcee(Emcee):
    def __init__(self, model, data, noise_sd, nwalkers=20, ntemps=10, random_subset=None, threads=None):
        super(PTemcee, self).__init__(model=model, data=data, noise_sd=noise_sd, nwalkers=nwalkers, random_subset=random_subset, threads=threads)
        self.ntemps = ntemps

    def make_guess(self):
        return np.dstack([p.sample(size=(self.ntemps, self.nwalkers)) for p in self.parameters])

    def make_sampler(self):
        return PTSampler(self.ntemps, self.nwalkers, self.ndim, self.lnlike, self.lnprior, threads=self.threads)


def subset_tempering(model, data, noise_sd, final_len=600, nwalkers=500, min_pixels=20, max_pixels=4000, threads='all', stages=3, stage_len=30, cutoff=.5):
    """
    Parameters
    ----------
    final_len: int
        Number of samples to use in final run
    stages: int
        Number subset stages to use
    min_pixels: int
        Number of pixels to use in the first stage
    max_pixels: int
        Number of pixels to use in the final stage
    cutoff: float (0 < cutoff < 1)
        At the end of each preliminary stage, discard samples with
        lnprobability < np.percentile(lnprobability, 100*cutoff). IE, discard
        he lower fraction of samples, the default argument of .5 corresponds to
        discarding samples below the median.
    stage_len: int
        Number of samples to use in preliminary stages
    """
    if threads == 'all':
        import multiprocessing
        threads = multiprocessing.cpu_count()
    if threads != None:
        print("Using {} threads".format(threads))


    fractions = np.logspace(np.log10(min_pixels), np.log10(max_pixels), stages+1)/data.size

    stage_fractions = fractions[:-1]
    final_fraction = fractions[-1]

    def new_par(par, mu, std):
        if hasattr(par, 'lower_bound'):
            return BoundedGaussianPrior(mu, std, getattr(par, 'lower_bound', -np.inf), getattr(par, 'upper_bound', np.inf), name=par.name)
        else:
            return GaussianPrior(mu, std, name=par.name)

    def sample_string(p):
        lb, ub = "", ""
        if getattr(p, 'lower_bound', -np.inf) != -np.inf:
            lb = ", lb={}".format(p.lower_bound)
        if getattr(p, 'upper_bound', np.inf) != np.inf:
            ub = ", ub={}".format(p.upper_bound)
        return "{p.name}: mu={p.mu:.3g}, sd={p.sd:.3g}{lb}{ub}".format(p=p, lb=lb, ub=ub)


    p0 = None
    for fraction in stage_fractions:
        tstart = time()
        emcee = Emcee(model=model, data=data, noise_sd=noise_sd, nwalkers=nwalkers, random_subset=fraction, threads=threads)
        result = emcee.sample(stage_len, p0)

        most_probable = result.most_probable_values()
        # TODO: use some form of clustering instead?
        good_samples = result.sampler.flatchain[np.where(result.sampler.flatlnprobability > np.percentile(result.sampler.flatlnprobability, 100*cutoff))]
        std = good_samples.std(axis=0)
        new_pars = [new_par(*p) for p in zip(model.parameters, most_probable, std)]
        p0 = np.vstack([p.sample(size=nwalkers) for p in new_pars]).T
        tend = time()
        print("--------\nStage at f={} finished in {}s.\nDrawing samples for next stage from:\n{}".format(fraction, tend-tstart, '\n'.join([sample_string(p) for p in new_pars])))

    tstart = time()
    emcee = Emcee(model=model, data=data, noise_sd=noise_sd, nwalkers=nwalkers, random_subset=final_fraction, threads=threads)
    result = emcee.sample(final_len, p0)
    tend = time()
    print("--------\nFinal stage at f={}, took {}s".format(final_fraction, tend-tstart))
    return result


class EmceeResult(HoloPyObject):
    def __init__(self, sampler, model):
        self.sampler = sampler
        self.model = model

        acceptance_fraction = sampler.acceptance_fraction.mean()
        if acceptance_fraction > .5 or acceptance_fraction < .2:
            print("Acceptance fraction is: {} which is outside the desired range of 0.2 to 0.5")

    @property
    def _names(self):
        return [p.name for p in self.model.parameters]

    def plot_traces(self, traces=10, burn_in=0):
        import matplotlib.pyplot as plt
        names = self._names
        samples = self.sampler.chain
        plt.figure(figsize=(9, 8), linewidth=.1)
        for var in range(len(names)):
            plt.subplot(3, 2, var+1)
            plt.plot(samples[:traces, burn_in:, var].T, color='k', linewidth=.3)
            plt.title(names[var])

    def data_frame(self, burn_in=0, thin='acor', include_lnprob=True):
        """
        Format the results into a data frame

        Parameters
        ----------
        burn_in : int
            Discard this many samples of burn in
        thin: int or 'acor'
            Thin the data by this factor if an int, or by the parameter
            autocorrelation if thin == 'acor'

        Returns
        -------
        df : DataFrame
            A data frame of samples for each parameter
        """
        import pandas as pd
        if thin == 'acor':
            thin = int(max(self.sampler.acor))
        chain = self.sampler.chain[burn_in::thin, ...]
        names = self._names
        npar = len(names)
        df = pd.DataFrame({n: t for (n, t) in zip(names,
                                                  chain.reshape(-1, npar).T)})
        if include_lnprob:
            df['lnprob'] = self.sampler.lnprobability[burn_in::thin].reshape(-1)

        return df

    def pairplots(self, filename=None, include_lnprob=False, burn_in=0, thin='acor'):

        df = self.data_frame(burn_in=burn_in, thin=thin, include_lnprob=include_lnprob)
        df = df.rename(columns={'center[0]': 'x', 'center[1]': 'y', 'center[2]': 'z' })
        xyz = ['x', 'y', 'z']
        xyz_enum = [(list(df.columns).index(v), v) for v in xyz]
        import seaborn as sns
        import matplotlib.pyplot as plt

        max_xyz_extent = (df.max() - df.min()).loc[xyz].max()

        def limits(x, y):
            xm = df[x].mean()
            ym = df[y].mean()
            # dividing by two would fill the plot exactly, but it is
            # nicer to have a little space around the outermost point
            e = max_xyz_extent/1.8
            return {'xmin': xm-e, 'xmax': xm+e, 'ymin': ym-e, 'ymax': ym+e}

        def plot():
            g = sns.PairGrid(df)
            g.map_diag(sns.kdeplot)
            g.map_lower(sns.kdeplot, cmap="Blues_d")
            g.map_upper(sns.regplot)
            for i, v in xyz_enum:
                for j, u in xyz_enum:
                    g.axes[j][i].axis(**limits(v, u))
            return g

        if filename is not None:
            isinteractive = plt.isinteractive()
            plt.ioff()
            g = plot()
            g.savefig(filename)
            plt.close(g.fig)
            if isinteractive:
                plt.ion()
        else:
            plot()

    def most_probable_values(self):
        values = self.sampler.chain[np.where(self.sampler.lnprobability ==
                               self.sampler.lnprobability.max())]
        if values.ndim == 2:
            if np.any(values.min(axis=0) != values.max(axis=0)):
                print("warning: multiple values with identical probability, output will be two dimensional")
            else:
                values = values[0]

        return values

    def most_probable_values_dict(self):
        dict([(n, v) for (n, v) in zip(self._names, self.most_probable_values())])

class UncertainValue(HoloPyObject):
    """
    Represent an uncertain value

    Parameters
    ----------
    value: float
        The value
    plus: float
        The plus n_sigma uncertainty (or the uncertainty if it is symmetric)
    minus: float or None
        The minus n_sigma uncertainty, or None if the uncertainty is symmetric
    n_sigma: int (or float)
        The number of sigma the uncertainties represent
    """
    def __init__(self, value, plus, minus=None, n_sigma=1):
        self.value = value
        self.plus = plus
        self.minus = minus
        self.n_sigma = n_sigma
