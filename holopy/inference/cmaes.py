import time
import os
import sys

import numpy as np
import xarray as xr
import cma

from holopy.core.holopy_object import HoloPyObject
from holopy.core.metadata import make_subset_data
from holopy.core.utils import choose_pool
from holopy.inference import prior
from holopy.inference.model import lnpost_wrapper
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
        If true (default), npixel new pixels are chosen for each call ofposterior.
        Otherwise, a single pixel subset is used throughout calculation.
    parent_fraction: float, optional
        Fraction of each generation to use to construct the next generation. 
        Takes symbol \mu in cma literature
    weight_function: function, optional
        takes arguments i, popsize with i in range(popsize); returns weight of i
    tols: dict, optional
        tolerance values to overwrite the cma defaults
    seed: int, optional
        random seed to use
    parallel: int or float, optional
        number of threads to use or one of {None, 'all' or 'mpi'}. 
        Default tries 'mpi' then 'all'.
    """
    def __init__(self, npixels=None, resample_pixels=True,
                    parent_fraction=0.25, weight_function=None,
                    tols={}, seed=None, parallel='auto'):
        self.npixels = npixels
        if resample_pixels:
            self.new_pixels = self.npixels
        else:
            self.new_pixels = None
        if weight_function is None:
            def weight_function(x, n):
                return (x + 1) <= (parent_fraction * n)
            self.weights = weight_function
        else:
            self.weights = weight_function
        self.tols = {'maxiter':2000, 'tolx':0.001, 'tolfun':0.1,
                     'tolstagnation':100}
        self.tols.update(tols)
        self.seed = seed
        self.parallel = parallel

    def optimize(self, model, data, popsize=None, walker_initial_pos=None):
        parameters = model._parameters
        time_start = time.time()
        if self.npixels is not None and self.new_pixels is None:
            data = make_subset_data(data, pixels=self.npixels, seed=self.seed)
        if popsize is None:
            numpars = len(parameters)
            popsize = int(2 + numpars + np.sqrt(numpars)) #cma uses 4+3*ln(n)
        if walker_initial_pos is None:
            walker_initial_pos = prior.make_guess(parameters, popsize, 
                                                            seed=self.seed)
        obj_func = lnpost_wrapper(model, data, self.new_pixels, True)
        sampler = run_cma(obj_func.evaluate, parameters, walker_initial_pos, 
                            self.weights, self.tols, self.seed, self.parallel)
        xrecent = sampler.logger.data['xrecent']
        samples = xr.DataArray([xrecent[:,5:]], 
                        dims = ['walker','chain','parameter'], 
                        coords = {'parameter':[p.name for p in parameters]})
        lnprobs = xr.DataArray([-xrecent[:,4]], dims=['walker', 'chain'])
        best_vals = sampler.best.get()[0]
        diffs = sampler.result.stds
        intervals = [UncertainValue(best_val, diff, name=par.name) 
                for best_val, diff, par in zip(best_vals, diffs, parameters)]
        stop = dict(sampler.stop())
        d_time = time.time() - time_start
        kwargs = {'lnprobs':lnprobs, 'samples':samples, 'stop_condition':stop}
        return FitResult(data, model, self, intervals, d_time, kwargs)


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
        takes arguments i, popsize with i in range(popsize); returns weight of i
    tols: dict, optional
        tolerance values to overwrite the cma defaults
    seed: int, optional
        random seed to use
    parallel: int or float, optional
        number of threads to use or one of {None, 'all' or 'mpi'}. 
        Default tries 'mpi' then 'all'.
    """

    popsize = len(initial_population)
    stds = [par.sd if isinstance(par, prior.Gaussian) 
                    else par.interval/4 for par in parameters]
    weights = [weight_function(i, popsize) for i in range(popsize)]
    cmaoptions = {'CMA_stds':stds, 'CMA_recombination_weights':weights,
                                                                'verbose':-3}
    cmaoptions.update(tols)
    if seed is not None:
        cmaoptions.update({'seed':seed})
    guess = [par.guess for par in parameters]
    cma_strategy = cma.CMAEvolutionStrategy(guess, 1, cmaoptions)
    cma_strategy.inject(initial_population, force=True)
    solutions = np.array([[None] * len(parameters)] * popsize)
    func_vals = np.array([0.] * popsize)
    while not cma_strategy.stop():
        invalid = [True] * popsize
        inf_replace_counter = 0
        while np.sum(invalid) > 0 and inf_replace_counter < 10:
            attempts = cma_strategy.ask(np.sum(invalid))
            solutions[invalid, :] = attempts
            pool = choose_pool(parallel)
            func_vals[invalid] = list(pool.map(obj_func, attempts))
            pool.close()
            invalid = ~np.isfinite(func_vals)
            inf_replace_counter += 1 # catches case where all are inf
        cma_strategy.tell(solutions, func_vals)
        cma_strategy.logger.add()
    cma_strategy.logger.load()
    return cma_strategy
