from __future__ import division

from holopy.fitting.fit import CostComputer
from holopy.fitting.errors import ParameterSpecificationError
from holopy.fitting.model import Model
from holopy.core import Image
from holopy.core.holopy_object import HoloPyObject
import prior
from random_subset import make_subset_data

from emcee import PTSampler, EnsembleSampler
import h5py
import yaml
import pandas as pd

from time import time
import numpy as np
from matplotlib.ticker import MaxNLocator

class ProbabilityComputer(HoloPyObject):
    def lnprob(self):
        raise NotImplementedError


class Emcee(HoloPyObject):
    def __init__(self, model, data, nwalkers=50, random_subset=None, threads=None, preprocess=None):
        self.model = model
        if preprocess is None:
            preprocess = lambda x: x
        if not isinstance(data, Image):
            self.n_frames = len(data)
            self.data = [make_subset_data(preprocess(d), random_subset) for d in data]
        else:
            self.n_frames = 1
            self.data = make_subset_data(preprocess(data), random_subset)
        self.nwalkers = nwalkers
        self.threads = threads


    def make_guess(self):
        return np.vstack([p.sample(size=(self.nwalkers)) for p in self.model.parameters]).T

    def make_sampler(self):
        return EnsembleSampler(self.nwalkers, len(list(self.model.parameters)), self.model.lnposterior, threads=self.threads, args=[self.data])

    def sample(self, n_samples, p0=None):
        sampler = self.make_sampler()
        if p0 is None:
            p0 = self.make_guess()

        sampler.run_mcmc(p0, n_samples)

        sampler.pool.terminate()
        sampler.pool.join()

        return EmceeResult(sampler, self.model)


class PTemcee(Emcee):
    def __init__(self, model, data, noise_sd, nwalkers=20, ntemps=10, random_subset=None, threads=None):
        super(PTemcee, self).__init__(model=model, data=data, noise_sd=noise_sd, nwalkers=nwalkers, random_subset=random_subset, threads=threads)
        self.ntemps = ntemps

    def make_guess(self):
        return np.dstack([p.sample(size=(self.ntemps, self.nwalkers)) for p in self.parameters])

    def make_sampler(self):
        return PTSampler(self.ntemps, self.nwalkers, self.ndim, self.lnlike, self.lnprior, threads=self.threads)


def subset_tempering(model, data, final_len=600, nwalkers=500, min_pixels=10, max_pixels=1000, threads='all', stages=3, stage_len=30, preprocess=None, verbose=True):
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
    stage_len: int
        Number of samples to use in preliminary stages
    """
    def log(s):
        if verbose:
            print(s)
    if threads == 'all':
        import multiprocessing
        threads = multiprocessing.cpu_count()
    if threads != None:
        log("Using {} threads".format(threads))

    if preprocess is None:
        preprocess = lambda x: x

    # TODO: is there a better way of figuring out if data is a list, tuple, iterator, ...?
    if not isinstance(data, Image):
        n_pixels = preprocess(data[0]).size
    else:
        n_pixels = preprocess(data).size
    fractions = np.logspace(np.log10(min_pixels), np.log10(max_pixels), stages+1)/n_pixels

    stage_fractions = fractions[:-1]
    final_fraction = fractions[-1]


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
        emcee = Emcee(model=model, data=data, nwalkers=nwalkers, random_subset=fraction, threads=threads, preprocess=preprocess)
        result = emcee.sample(stage_len, p0)
        new_pars = result.updated_priors()

        # TODO need to do something if we come back sd == 0
        p0 = np.vstack([p.sample(size=nwalkers) for p in new_pars]).T
        tend = time()
        log("--------\nStage at f={} finished in {}s.\nDrawing samples for next stage from:\n{}".format(fraction, tend-tstart, '\n'.join([sample_string(p) for p in new_pars])))

    tstart = time()
    emcee = Emcee(model=model, data=data, nwalkers=nwalkers, random_subset=final_fraction, threads=threads)
    result = emcee.sample(final_len, p0)
    tend = time()
    log("--------\nFinal stage at f={}, took {}s".format(final_fraction, tend-tstart))
    return result

class SamplingResult(HoloPyObject):
    def __init__(self, samples, model, autocorrelation=None):
        self.samples = samples
        self.model = model
        # None means we don't know the autocorrelation (ie it wasn't stored)
        self._autocorrelation=autocorrelation

    @property
    def autocorrelation(self):
        return self._autocorrelation

    @property
    def _names_internal(self):
        return [p.name for p in self.model.parameters]

    @property
    def _names(self):
        lookup = {'center[0]': 'x', 'center[1]': 'y', 'center[2]': 'z'}
        return [lookup.get(p.name, p.name) for p in self.model.parameters]

    def pairplots(self, filename=None, include_lnprob=False, burn_in=0, thin='acor', include_vars='all', figsize=None, max_x_ticks=None):
        df = self.data_frame(burn_in=burn_in, thin=thin, include_lnprob=include_lnprob)
        if include_vars == 'all':
            include_vars = self._names
        df = df.rename(columns={'center[0]': 'x', 'center[1]': 'y', 'center[2]': 'z' })
        df = df.iloc[:,[list(df.columns).index(v) for v in include_vars]]
        xyz = [x for x in 'x', 'y', 'z' if x in df.columns]
        #xyz = [x for x in 'center[0]', 'center[1]', 'center[2]' if x in df.columns]
        xyz_enum = [(list(df.columns).index(v), v) for v in xyz]
        rest_enum = [(list(df.columns).index(v), v) for v in include_vars if v not in xyz]
        import seaborn as sns
        import matplotlib.pyplot as plt

        max_xyz_extent = (df.max() - df.min()).loc[xyz].max()


        def limits(x, y, extent):
            xm = df[x].mean()
            ym = df[y].mean()
            # dividing by two would fill the plot exactly, but it is
            # nicer to have a little space around the outermost point
            e = extent/1.6
            return {'xmin': xm-e, 'xmax': xm+e, 'ymin': ym-e, 'ymax': ym+e}

        def plot():
            g = sns.PairGrid(df)
            g.map_diag(sns.kdeplot)
            #g.map_lower(sns.kdeplot, cmap="Blues_d")
            g.map_offdiag(sns.kdeplot, cmap="Blues_d")
            #g.map_upper(plt.scatter, s=1)

            for i, v in xyz_enum:
                for j, u in xyz_enum:
                    g.axes[j][i].axis(**limits(v, u, max_xyz_extent))
            for i, v in rest_enum:
                extent = df[v].max() - df[v].min()
                g.axes[i][i].axis(**limits(v, v, extent))
            if figsize is not None:
                g.fig.set_size_inches(*figsize)
            if max_x_ticks is not None:
                for i in range(len(include_vars)):
                    g.axes[i][i].get_xaxis().set_major_locator(MaxNLocator(max_x_ticks))
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
            g = plot()

        return g


    def values(self):
        d = self.data_frame(thin=None)
        d = d.sort_values('lnprob', ascending=False)
        mp = d.iloc[0,:-1]
        def find_bound(f, i):
            b = d.iloc[0, :-1]
            while (b == mp).any() and i < d.shape[0]:
                b = f(b, d.iloc[i,:-1])
                i+=1
            return b

        i = 0
      
        while d.lnprob.iloc[i] > d.lnprob.max()-.5 and i < d.shape[0]:
            i+=1

        upper = find_bound(np.maximum, i+1)
        lower = find_bound(np.minimum, i+1)
        return [UncertainValue(mp[p], upper[p]-mp[p], mp[p]-lower[p]) for p in self._names]

    def confidence_interval(self, interval=.68, burn_in=0, thin=None):
        q = self.data_frame(thin=thin, burn_in=burn_in).quantile([.5-interval/2, .5, .5+interval/2])
        mp = q.iloc[1]
        plus = q.iloc[2] - q.iloc[1]
        minus = q.iloc[1] - q.iloc[0]
        return [UncertainValue(mp[p], plus[p], minus[p]) for p in self._names]

    def plot_traces(self, traces=10, thin=None, burn_in=0):
        import matplotlib.pyplot as plt
        samples = self.data_frame(burn_in=burn_in, thin=thin, flat=False)
        names = self._names
        pars = len(names)
        rows = (pars+1)//2
        plt.figure(figsize=(9, rows*2.8), linewidth=.1)
        for var in range(pars):
            plt.subplot(rows, 2, var+1)
            plt.plot(samples.iloc[var, burn_in:, :traces], color='k', linewidth=.3)
            plt.title(names[var])

    def _repr_html_(self):
        results = "{}".format(", ".join(["{}:{}".format(n, v._repr_latex_()) for n, v in zip(self._names, self.confidence_interval())]))
        n_samples = self.samples.shape[1] * self.samples.shape[2]
        block = """<h4>{s.__class__.__name__}</h4> {results}
{n_samples} Samples
        """.format(s=self, results=results, n_samples=n_samples)
        return "<br>".join(block.split('\n'))

    def save(self, filename, burn_in=0, thin='acor', include_lnprob=True):
        thin = self._interpret_thin(thin)
        df = self.data_frame(burn_in=burn_in, thin=thin, include_lnprob=include_lnprob, flat=False)
        groupname = 'samples'
        df.to_hdf(filename, groupname)
        f = h5py.File(filename)
        g = f[groupname]
        g.attrs['model'] = yaml.dump(self.model)
        g.attrs['autocorrelation'] = self.autocorrelation/thin
        f.close()

    @classmethod
    def load(cls, filename):
        samples = pd.read_hdf(filename, 'samples')

    def updated_priors(self, extra_uncertainty=None):
        if extra_uncertainty is None:
            extra_uncertainty = np.zeros(len(list(self._names)))
        return [prior.updated(*p) for p in
                zip(self.model.parameters, self.values(), extra_uncertainty)]


    def _interpret_thin(self, thin):
        if thin == 'acor':
            thin = self.autocorrelation
        elif thin is None:
            thin = 1
        if thin < 1:
            thin = 1

        return thin

    def data_frame(self, burn_in=0, thin='acor', include_lnprob=True, flat=True, xyz=True):
        thin = self._interpret_thin(thin)
        df = self.samples.iloc[:, :, burn_in::thin]
        if flat:
            df = df.to_frame()
        else:
            df = df
        if xyz:
            df = df.rename(columns={'center[0]': 'x', 'center[1]': 'y', 'center[2]': 'z' })
        return df

def load_sampling(filename):
    samples = pd.read_hdf(filename)
    f = h5py.File(filename)
    g = f['samples']
    model = yaml.load(g.attrs['model'])
    f.close()
    return SamplingResult(samples, model)

class EmceeResult(SamplingResult):
    def __init__(self, sampler, model):
        self.sampler = sampler
        self.model = model

    def plot_lnprob(self, traces='all', burn_in=0):
        import matplotlib.pyplot as plt
        if traces == 'all':
            traces = slice(None)
        plt.plot(self.sampler.lnprobability[traces, burn_in:].T, color='k', linewidth=.1)

    @property
    def n_steps(self):
        return self.sampler.lnprobability.shape[1]

    @property
    def approx_independent_steps(self):
        return int(self.n_steps/max(self.sampler.acor))

    @property
    def acceptance_fraction(self):
        return self.sampler.acceptance_fraction.mean()

    def data_frame(self, burn_in=0, thin='acor', include_lnprob=True, flat=True, xyz=True):
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
        thin = self._interpret_thin(thin)
        chain = self.sampler.chain[:, burn_in::thin, :]
        names = self._names
        npar = len(names)
        df = pd.Panel({n: t for n, t in zip(names, chain.T)})
        df['lnprob'] = self.sampler.lnprobability[:, burn_in::thin].T
        if flat:
            df = df.to_frame()
        #if xyz:
        #    df = df.rename(columns={'center[0]': 'x', 'center[1]': 'y', 'center[2]': 'z' })
        return df

    @property
    def autocorrelation(self):
        return max(self.sampler.acor)

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
        return {n: v for (n, v) in zip(self._names, self.most_probable_values())}

    def _repr_html_(self):
        results = "{}".format(", ".join(["{}:{}".format(n, v._repr_latex_()) for n, v in zip(self._names, self.values())]))
        block = """<h4>{s.__class__.__name__}</h4> {results}
{s.sampler.chain.shape[0]} walkers
{s.n_steps} Steps
~ {s.approx_independent_steps} of which are independent
Acceptance Fraction: {s.acceptance_fraction}
        """.format(s=self, results=results)
        return "<br>".join(block.split('\n'))


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

    def _repr_latex_(self):
        from IPython.display import Math
        confidence=""
        if self.n_sigma != 1:
            confidence=" (\mathrm{{{}\ sigma}})".format(self.n_sigma)
        display_precision = int(round(np.log10(self.value/(min(self.plus, self.minus))) + .6))
        value_fmt = "{{:.{}g}}".format(max(display_precision, 2))
        value = value_fmt.format(self.value)
        return "${value}^{{+{s.plus:.2g}}}_{{-{s.minus:.2g}}}{confidence}$".format(s=self, confidence=confidence, value=value)

def timeseries(model, data, centered_subimage=False):
    results = []
    for frame in data:
        res = subset_tempering(model, frame)
        results.append[res.values()]
        model = res.updated_priors()
