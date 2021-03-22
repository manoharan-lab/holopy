# Copyright 2011-2019, Vinothan N. Manoharan, Thomas G. Dimiduk,
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
Stochastic fitting of models to data

.. moduleauthor:: Solomon Barkley
"""
import time
import os
import sys
import tempfile
import shutil
import warnings

import numpy as np
import xarray as xr
try:
    import cma
    _CMA_MISSING = False
except ModuleNotFoundError:
    _CMA_MISSING = True

from holopy.core.holopy_object import HoloPyObject
from holopy.core.metadata import make_subset_data
from holopy.core.utils import choose_pool, LnpostWrapper
from holopy.core.errors import DependencyMissing
from holopy.inference import prior
from holopy.inference.result import FitResult, UncertainValue


class CmaStrategy(HoloPyObject):
    """
    Inference strategy defining a Covariance Matrix Adaptation Evolutionary
    Strategy using cma package

    Parameters
    ----------
    npixels : int, optional
        Number of pixels in the image to fit. default fits all.
    resample_pixels: Boolean, optional
        If true (default), new pixels are chosen for each call of posterior.
        Otherwise, a single pixel subset is used throughout calculation.
    parent_fraction: float, optional
        Fraction of each generation to use to construct the next generation.
        Takes symbol \mu in cma literature
    weight_function: function, optional
        takes arguments (i, popsize), i in range(popsize); returns weight of i
    tols: dict, optional
        tolerance values to overwrite the cma defaults
    seed: int, optional
        random seed to use
    parallel: optional
        number of threads to use or pool object or one of {None, 'all', 'mpi'}.
        Default tries 'mpi' then 'all'.
    """
    def __init__(self, npixels=None, popsize=None, resample_pixels=True,
                 parent_fraction=0.25, weight_function=None,
                 walker_initial_pos=None, tols={}, seed=None,
                 parallel='auto'):
        self.npixels = npixels
        self.popsize = popsize
        if resample_pixels:
            self.new_pixels = self.npixels
        else:
            self.new_pixels = None
        if weight_function is None:
            def weight_function(x, n):
                return (x + 1) <= (parent_fraction * n)
        self.weights = weight_function
        self.walker_initial_pos = walker_initial_pos
        self.tols = {'maxiter': 2000, 'tolx': 0.001, 'tolfun': 0.1,
                     'tolstagnation': 100}
        self.tols.update(tols)
        self.seed = seed
        self.parallel = parallel

    def fit(self, model, data):
        parameters = model._parameters
        par_names = model._parameter_names
        time_start = time.time()
        if self.npixels is not None and self.new_pixels is None:
            data = make_subset_data(data, pixels=self.npixels, seed=self.seed)
        if self.popsize is None:
            npars = len(parameters)
            self.popsize = int(2 + npars + np.sqrt(npars))
            # cma default popsize is 4+3*ln(n). Ours is larger for npars > 5.

        if self.walker_initial_pos is None:
            self.walker_initial_pos = model.generate_guess(self.popsize,
                                                           seed=self.seed)
        obj_func = LnpostWrapper(model, data, self.new_pixels, True)
        sampler = run_cma(obj_func.evaluate, parameters,
                          self.walker_initial_pos, self.weights, self.tols,
                          self.seed, self.parallel)
        xrecent = sampler.logger.data['xrecent']
        samples = xr.DataArray(
            [xrecent[:, 5:]], dims=['walker', 'chain', 'parameter'],
            coords={'parameter': par_names})
        lnprobs = xr.DataArray([-xrecent[:, 4]], dims=['walker', 'chain'])
        best_vals = sampler.best.get()[0]
        diffs = sampler.result.stds
        intervals = [UncertainValue(best_val, diff, name=par) for
                     best_val, diff, par in zip(best_vals, diffs, par_names)]
        stop = dict(sampler.stop())
        d_time = time.time() - time_start
        kwargs = {'lnprobs': lnprobs, 'samples': samples,
                  'intervals': intervals, 'stop_condition': stop,
                  'popsize': self.popsize}
        return FitResult(data, model, self, d_time, kwargs)


def run_cma(obj_func, parameters, initial_population, weight_function,
            tols={}, seed=None, parallel='auto'):
    """
    instantiate and run a CMAEvolutionStrategy object

    Parameters
    ----------
    obj_func : Function
        function to be minimized (not maximized like posterior)
    parameters : list of Prior objects
        parameters to fit
    initial_population: array
        starting population with shape = (popsize, len(parameters))
    weight_function: function
        takes arguments (i, popsize), i in range(popsize); returns weight of i
    tols: dict, optional
        tolerance values to overwrite the cma defaults
    seed: int, optional
        random seed to use
    parallel: optional
        number of threads to use or pool object or one of {None, 'all', 'mpi'}.
        Default tries 'mpi' then 'all'.
    """
    if _CMA_MISSING:
        install_msg = "Install it with \'conda install -c conda-forge cma\'."
        raise DependencyMissing('cma', install_msg)

    popsize = len(initial_population)
    stds = np.std(initial_population, axis=0, ddof=1)
    weights = [weight_function(i, popsize) for i in range(popsize)]
    if weights[-1] > 0:
        weights[-1] = 0
        warnings.warn('Setting weight of worst parent to 0')
    with tempfile.TemporaryDirectory() as tempdir:
        cmaoptions = {'CMA_stds': stds, 'CMA_recombination_weights': weights,
                      'verb_filenameprefix': tempdir, 'verbose': -3}
        cmaoptions.update(tols)
        if seed is not None:
            cmaoptions.update({'seed': seed})
        guess = [par.guess for par in parameters]
        cma_strategy = cma.CMAEvolutionStrategy(guess, 1, cmaoptions)
        cma_strategy.inject(initial_population, force=True)
        solutions = np.zeros((popsize, len(parameters)))
        func_vals = np.zeros(popsize)
        pool = choose_pool(parallel)
        while not cma_strategy.stop():
            invalid = np.ones(popsize, dtype=bool)
            inf_replace_counter = 0
            while invalid.any() and inf_replace_counter < 10:
                attempts = cma_strategy.ask(np.sum(invalid))
                solutions[invalid, :] = attempts
                func_vals[invalid] = list(pool.map(obj_func, attempts))
                invalid = ~np.isfinite(func_vals)
                inf_replace_counter += 1  # catches case where all are inf
            cma_strategy.tell(solutions, func_vals)
            cma_strategy.logger.add()
        cma_strategy.logger.load()

    if pool is not parallel:
        # I made pool, responsible for closing it.
        pool.close()
    return cma_strategy

