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
import time

import xarray as xr
import numpy as np
import emcee

from holopy.core.holopy_object import HoloPyObject
from holopy.core.metadata import make_subset_data
from holopy.core.utils import choose_pool
from holopy.inference.model import LnpostWrapper
from holopy.inference.result import SamplingResult, TemperedSamplingResult
from . import prior

def sample_one_sigma_gaussian(result):
    par_ranges = result.intervals
    new_pars = [prior.updated(result.model.parameters[p.name], p) for p in par_ranges]
    return np.vstack([p.sample(size=result.strategy.nwalkers)] for p in new_pars).T


class EmceeStrategy(HoloPyObject):
    def __init__(self, nwalkers=100, npixels=None, parallel='auto', cleanup_threads=True, seed=None, resample_pixels=False):
        self.nwalkers = nwalkers
        self.npixels = npixels
        self.parallel = parallel
        self.cleanup_threads = cleanup_threads
        self.seed = seed
        if resample_pixels:
            self.new_pixels = self.npixels
        else:
            self.new_pixels = None

    def optimize(self, model, data, nsamples=1000, walker_initial_pos=None):
        time_start = time.time()
        if self.npixels is not None and self.new_pixels is None:
            data = make_subset_data(data, pixels=self.npixels, seed=self.seed)
        if walker_initial_pos is None:
            walker_initial_pos = model.generate_guess(self.nwalkers, seed=self.seed)
        sampler = sample_emcee(model=model, data=data, nwalkers=self.nwalkers,
                               walker_initial_pos=walker_initial_pos, nsamples=nsamples,
                               parallel=self.parallel, cleanup_threads=self.cleanup_threads, seed=self.seed, new_pixels=self.new_pixels)

        samples = emcee_samples_DataArray(sampler, model._parameters)
        lnprobs = emcee_lnprobs_DataArray(sampler)

        d_time = time.time() - time_start
        kwargs = {'lnprobs': lnprobs, 'samples':samples}
        return SamplingResult(data, model, self, d_time, kwargs)

    # deprecated as of 3.3
    def sample(self, model, data, nsamples=1000, walker_initial_pos=None):
        from holopy.fitting import fit_warning
        fit_warning('EmceeStrategy.optimize', 'EmceeStrategy.sample')
        return self.optimize(model, data, nsamples, walker_initial_pos)

class TemperedStrategy(EmceeStrategy):
    def __init__(self, next_initial_dist=sample_one_sigma_gaussian, nwalkers=100, min_pixels=None, npixels=1000, parallel='auto', stages=3, stage_len=30, seed=None, resample_pixels=False):
        self.stages = stages
        self.stage_strategies = []
        if min_pixels is None:
            min_pixels = npixels/20
        for p in np.logspace(np.log10(min_pixels), np.log10(npixels), stages+1):
            self.stage_strategies.append(EmceeStrategy(nwalkers=nwalkers, npixels=int(round(p)), parallel=parallel, seed=seed, resample_pixels=resample_pixels))
            if seed is not None:
                seed += 1
        self.parallel=parallel
        self.stage_len=stage_len
        self.nwalkers=nwalkers
        self.next_initial_dist = next_initial_dist

    def optimize(self, model, data, nsamples=1000, walker_initial_pos = None):
        start_time = time.time()
        stage_results = []
        guess = walker_initial_pos
        for stage in self.stage_strategies[:-1]:
            result = stage.optimize(model, data, nsamples=self.stage_len, walker_initial_pos=guess)
            guess = self.next_initial_dist(result)
            stage_results.append(result)

        end_result = self.stage_strategies[-1].optimize(model=model, data=data, nsamples=nsamples, walker_initial_pos=guess)
        d_time = time.time()-start_time
        return TemperedSamplingResult(end_result, stage_results, self, d_time)

def emcee_samples_DataArray(sampler, parameters):
    return xr.DataArray(sampler.chain, dims=['walker', 'chain', 'parameter'],
                        coords={'parameter': [p.name for p in parameters]},
                        attrs={"acceptance_fraction": sampler.acceptance_fraction.mean()})

def emcee_lnprobs_DataArray(sampler):
    return xr.DataArray(sampler.lnprobability, dims=['walker', 'chain'],
                        attrs={"acceptance_fraction": sampler.acceptance_fraction.mean()})

def sample_emcee(model, data, nwalkers, nsamples, walker_initial_pos,
                 parallel='auto', cleanup_threads=True, seed=None, new_pixels = None):
    obj_func = LnpostWrapper(model, data, new_pixels)
    pool = choose_pool(parallel)
    sampler = emcee.EnsembleSampler(nwalkers, len(model._parameters), 
                                        obj_func.evaluate, pool=pool)
    if seed is not None:
        np.random.seed(seed)
        seed_state = np.random.mtrand.RandomState(seed).get_state()
        sampler.random_state=seed_state

    sampler.run_mcmc(walker_initial_pos, nsamples)
    if pool is not parallel:
        pool.close()

    return sampler

