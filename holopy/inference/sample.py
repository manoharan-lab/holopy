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
"""
Sample posterior probabilities given model and data

.. moduleauthor:: Thomas G. Dimiduk <tom@dimiduk.net>
"""
import multiprocessing

import xarray as xr
import numpy as np
from emcee import EnsembleSampler
import emcee

from holopy.core.holopy_object import HoloPyObject
from holopy.core.metadata import make_subset_data
from holopy.inference.result import SamplingResult, TemperedSamplingResult

from . import prior

def autothreads(threads='auto', quiet=False):
    if threads == 'auto':
        threads = multiprocessing.cpu_count()
    if threads is None:
        threads = 1
    return threads

def sample_one_sigma_gaussian(result):
    v = result.values()
    new_pars = [prior.updated(p, v[p.name]) for p in result.model._parameters]
    return np.vstack([p.sample(size=result.strategy.nwalkers)] for p in new_pars).T

def tempered_sample(model, data, nwalkers=100, min_pixels=50, max_pixels=2000,
                    samples=600, next_initial_dist=sample_one_sigma_gaussian,
                    stages=3, stage_len=30, seed=None, threads='auto'):
    if seed is not None:
        np.random.seed(seed)
    s = TemperedStrategy(next_initial_dist, nwalkers, min_pixels, max_pixels, stages=stages, stage_len=stage_len, seed=seed, threads=threads)
    return s.sample(model, data, samples)

class EmceeStrategy(HoloPyObject):
    def __init__(self, nwalkers=100, pixels=2000, threads='auto', cleanup_threads=True, seed=None, resample_pixels=False):
        self.nwalkers = nwalkers
        self.pixels = pixels
        self.threads = threads
        self.cleanup_threads = cleanup_threads
        self.seed = seed
        if resample_pixels:
            self.new_pixels = self.pixels
        else:
            self.new_pixels = None

    def make_guess(self, parameters, scaling=1, seed=None):
        def sample(prior):
            raw_sample = prior.sample(size=self.nwalkers)
            scaled_guess = prior.guess + scaling * (raw_sample - prior.guess)
            return scaled_guess

        if seed is not None:
            np.random.seed(seed)
        return np.vstack([sample(p) for p in parameters]).T

    def sample(self, model, data, nsamples=1000, walker_initial_pos=None):
        if self.pixels is not None and self.new_pixels is None:
            data = make_subset_data(data, pixels=self.pixels, seed=self.seed)
        if walker_initial_pos is None:
            walker_initial_pos = self.make_guess(model._parameters, seed=self.seed)
        sampler = sample_emcee(model=model, data=data, nwalkers=self.nwalkers,
                               walker_initial_pos=walker_initial_pos, nsamples=nsamples,
                               threads=self.threads, cleanup_threads=self.cleanup_threads, seed=self.seed, new_pixels=self.new_pixels)

        try:
            acor = sampler.acor
        except emcee.autocorr.AutocorrError:
            acor = None

        samples = emcee_samples_DataArray(sampler, model._parameters)
        lnprobs = emcee_lnprobs_DataArray(sampler)
        return SamplingResult(xr.Dataset({'samples': samples, 'lnprobs': lnprobs, 'data': data}),
                              model=model, strategy=self)


class TemperedStrategy(EmceeStrategy):
    def __init__(self, next_initial_dist=sample_one_sigma_gaussian, nwalkers=100, min_pixels=50, max_pixels=1000, threads='auto', stages=3, stage_len=30, seed=None, resample_pixels=False):

        self.seed = seed
        self.stages = stages
        self.stage_strategies = []
        for p in np.logspace(np.log10(min_pixels), np.log10(max_pixels), stages+1):
            self.stage_strategies.append(EmceeStrategy(nwalkers=nwalkers, pixels=int(round(p)), threads=threads, seed=seed, resample_pixels=resample_pixels))
            if seed is not None:
                seed += 1

        self.threads=threads
        self.stage_len=stage_len
        self.nwalkers=nwalkers
        self.next_initial_dist = next_initial_dist

    def sample(self, model, data, nsamples=1000):
        stage_results = []
        guess = self.make_guess(model._parameters)
        for stage in self.stage_strategies[:-1]:
            result = stage.sample(model, data, nsamples=self.stage_len, walker_initial_pos=guess)
            guess = self.next_initial_dist(result)
            stage_results.append(result)

        result = self.stage_strategies[-1].sample(model=model, data=data, nsamples=nsamples, walker_initial_pos=guess)

        return TemperedSamplingResult(end_result=result, stage_results=stage_results, strategy=self)


def get_acor(sampler):
    try:
        return sampler.acor
    except emcee.autocorr.AutocorrError:
        return None

def emcee_samples_DataArray(sampler, parameters):
    return xr.DataArray(sampler.chain, dims=['walker', 'chain', 'parameter'],
                        coords={'parameter': [p.name for p in parameters]},
                        attrs={"acceptance_fraction": sampler.acceptance_fraction.mean(),
                               "autocorr": get_acor(sampler)})

def emcee_lnprobs_DataArray(sampler):
    return xr.DataArray(sampler.lnprobability, dims=['walker', 'chain'],
                        attrs={"acceptance_fraction": sampler.acceptance_fraction.mean(),
                               "autocorr": get_acor(sampler)})

def sample_emcee_autocorr(model, data, nwalkers, independent_samples, walker_initial_pos,
                          estimated_autocorr, threads='auto'):
    def target_chain_len(autocorr):
        return estimated_burn_in + max(desired_samples/self.nwalkers, 10) * estimated_autocorr

    samples = target_chain_len(estimated_autocorr)

    sampler = sample_emcee(model, data, nwalkers, walker_initial_pos, )

    acor = sampler.acor
    # if the actual autocorr is larger than our estimate, run some more samples
    if target_chain_len(sampler.acor) > samples:
        more = target_chain_len(sampler.acor) - samples
        sampler.run_mcmc(None, more)

def sample_emcee(model, data, nwalkers, nsamples, walker_initial_pos,
                 threads='auto', cleanup_threads=True, seed=None, new_pixels = None):
    sampler = EnsembleSampler(nwalkers, len(model._parameters),
                              model.lnposterior,
                              threads=autothreads(threads), args=[data, new_pixels])
    if seed is not None:
        np.random.seed(seed)
        seed_state = np.random.mtrand.RandomState(seed).get_state()
        sampler.random_state=seed_state

    sampler.run_mcmc(walker_initial_pos, nsamples)

    if sampler.pool is not None and cleanup_threads:
        sampler.pool.terminate()
        sampler.pool.join()

    return sampler

