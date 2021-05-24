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
try:
    import emcee
    _EMCEE_MISSING = False
except ModuleNotFoundError:
    _EMCEE_MISSING = True

from holopy.core.holopy_object import HoloPyObject
from holopy.core.metadata import make_subset_data
from holopy.core.utils import choose_pool, LnpostWrapper
from holopy.core.errors import DependencyMissing
from holopy.inference.result import SamplingResult, TemperedSamplingResult
from holopy.inference import prior


def sample_one_sigma_gaussian(result):
    par_ranges = result.intervals
    new_pars = [prior.updated(result.model.parameters[p.name], p)
                for p in par_ranges]
    return np.vstack([[p.sample(size=result.strategy.nwalkers)]
                      for p in new_pars]).T


class EmceeStrategy(HoloPyObject):
    _default_nsamples = 1000

    def __init__(self, nwalkers=100, nsamples=None, npixels=None,
                 walker_initial_pos=None, parallel='auto', seed=None):
        self.nwalkers = nwalkers
        if nsamples is None:
            nsamples = self._default_nsamples
        self.nsamples = nsamples
        self.npixels = npixels
        self.walker_initial_pos = walker_initial_pos
        self.parallel = parallel
        self.seed = seed

    def sample(self, model, data):
        time_start = time.time()
        if self.npixels is not None:
            data = make_subset_data(data, pixels=self.npixels, seed=self.seed)
        if self.walker_initial_pos is None:
            self.walker_initial_pos = model.generate_guess(self.nwalkers,
                                                           seed=self.seed)
        sampler = sample_emcee(model=model, data=data, nwalkers=self.nwalkers,
                               walker_initial_pos=self.walker_initial_pos,
                               nsamples=self.nsamples, parallel=self.parallel,
                               seed=self.seed)

        samples = emcee_samples_DataArray(sampler, model._parameter_names)
        lnprobs = emcee_lnprobs_DataArray(sampler)

        d_time = time.time() - time_start
        kwargs = {'lnprobs': lnprobs, 'samples': samples}
        return SamplingResult(data, model, self, d_time, kwargs)


class TemperedStrategy(EmceeStrategy):
    def __init__(self, next_initial_dist=sample_one_sigma_gaussian,
                 nwalkers=100, nsamples=1000, min_pixels=None, npixels=1000,
                 walker_initial_pos=None, parallel='auto', stages=3,
                 stage_len=30, seed=None):
        self.nwalkers = nwalkers
        self.parallel = parallel
        self.seed = seed
        self.walker_initial_pos = walker_initial_pos
        self.next_initial_dist = next_initial_dist
        self.stage_strategies = []
        if min_pixels is None:
            min_pixels = npixels/20
        npixels_for_stages = np.logspace(np.log10(min_pixels),
                                         np.log10(npixels), stages + 1)
        for stage_pixels in npixels_for_stages[:-1]:
            self.add_stage_strategy(stage_len, stage_pixels)
        self.add_stage_strategy(nsamples, npixels)

    def add_stage_strategy(self, nsamples, npixels):
        self.stage_strategies.append(
            EmceeStrategy(nwalkers=self.nwalkers,
                          nsamples=nsamples,
                          npixels=int(round(npixels)),
                          parallel=self.parallel,
                          seed=self.seed))
        if self.seed is not None:
            self.seed += 1

    def sample(self, model, data):
        start_time = time.time()
        stage_results = []
        guess = self.walker_initial_pos
        for i, strategy in enumerate(self.stage_strategies):
            strategy.walker_initial_pos = guess
            result = strategy.sample(model, data)
            stage_results.append(result)
            guess = self.next_initial_dist(result)
        d_time = time.time()-start_time
        return TemperedSamplingResult(result, stage_results, self, d_time)


def emcee_samples_DataArray(sampler, parameter_names):
    acceptance_fraction = sampler.acceptance_fraction.mean()
    try:
        chain = sampler.get_chain()
    except AttributeError:
        # emcee version < 3.0.0
        chain = sampler.chain
    return xr.DataArray(chain,
                        dims=['walker', 'chain', 'parameter'],
                        coords={'parameter': [par for par in parameter_names]},
                        attrs={"acceptance_fraction": acceptance_fraction})


def emcee_lnprobs_DataArray(sampler):
    acceptance_fraction = sampler.acceptance_fraction.mean()
    try:
        lnprob = sampler.get_log_prob()
    except AttributeError:
        # emcee version < 3.0.0
        lnprob = sampler.lnprobability
    return xr.DataArray(lnprob, dims=['walker', 'chain'],
                        attrs={"acceptance_fraction": acceptance_fraction})


def sample_emcee(model, data, nwalkers, nsamples, walker_initial_pos,
                 parallel='auto', seed=None):
    if _EMCEE_MISSING:
        raise DependencyMissing(
            'emcee', "Install it with \'conda install -c conda-forge emcee\'.")

    obj_func = LnpostWrapper(model, data)
    pool = choose_pool(parallel)
    sampler = emcee.EnsembleSampler(nwalkers, len(model._parameters),
                                    obj_func.evaluate, pool=pool)
    if seed is not None:
        np.random.seed(seed)
        seed_state = np.random.mtrand.RandomState(seed).get_state()
        sampler.random_state = seed_state

    sampler.run_mcmc(walker_initial_pos, nsamples)
    if pool is not parallel:
        pool.close()

    return sampler
