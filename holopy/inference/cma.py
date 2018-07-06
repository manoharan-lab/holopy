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
import cma

from holopy.core.holopy_object import HoloPyObject
from . import prior

class CMAESStrategy(HoloPyObject):
    def __init__(self, pixels=2000, threads='auto', cleanup_threads=True, seed=None):
        self.pixels = pixels
        self.threads = threads
        self.cleanup_threads = cleanup_threads
        self.seed = seed

    def make_guess(self, parameters):
        return np.vstack([p.sample(size=(self.nwalkers)) for p in parameters]).T

    def optimize(self, model, data, initial_distribution=None):
        if self.pixels is not None:
            data = make_subset_data(data, pixels=self.pixels)
        if initial_distribution is None:
            initial_distribution = self.make_guess(model.parameters)
        #rescale parameter ranges?
        #determine population size
        #set seed
        #set -inf to NaN for automatic resample
        #determine when converged
        #process and return results object
        sampler = sample_emcee(model=model, data=data, nwalkers=self.nwalkers,
                               walker_initial_pos=walker_initial_pos, nsamples=nsamples,
                               threads=self.threads, cleanup_threads=self.cleanup_threads, seed=self.seed)

        samples = emcee_samples_DataArray(sampler, model.parameters)
        lnprobs = emcee_lnprobs_DataArray(sampler)
        return SamplingResult(xr.Dataset({'samples': samples, 'lnprobs': lnprobs, 'data': data}),
                              model=model, strategy=self)

'''
CODE TO REPLACE MANUALLY REPLACE NaN VALUES
from http://cma.gforge.inria.fr/html-pythoncma/cma.CMAEvolutionStrategy-class.html
>>> while not es.stop():
...     fit, X = [], []
...     while len(X) < es.popsize:
...         curr_fit = None
...         while curr_fit in (None, np.NaN):
...             x = es.ask(1)[0]
...             curr_fit = cma.fcts.somenan(x, cma.fcts.elli) # might return np.NaN
...         X.append(x)
...         fit.append(curr_fit)
...     es.tell(X, fit)
...     es.disp()
'''
def sample_emcee(model, data, nwalkers, nsamples, walker_initial_pos,
                 threads='auto', cleanup_threads=True, seed=None):
    sampler = EnsembleSampler(nwalkers, len(list(model.parameters)),
                              model.lnposterior,
                              threads=autothreads(threads), args=[data])
    if seed is not None:
        np.random.seed(seed)
        seed_state = np.random.mtrand.RandomState(seed).get_state()
        sampler.random_state=seed_state

    sampler.run_mcmc(walker_initial_pos, nsamples)

    if sampler.pool is not None and cleanup_threads:
        sampler.pool.terminate()
        sampler.pool.join()

    return sampler

