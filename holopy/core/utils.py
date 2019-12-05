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
Misc utility functions to make coding more convenient

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
"""


import os
import sys
import shutil
import errno
from copy import copy
import itertools

import numpy as np
import xarray as xr
try:
    import schwimmbad
    NO_SCHWIMMBAD = False
except ModuleNotFoundError:
    NO_SCHWIMMBAD = True

from holopy.core.errors import DependencyMissing
from holopy.core.holopy_object import HoloPyObject


# FIXME this is difficult to test, since it works on the file level
# perhaps make this capture the output rather than writing it to devnull?
class SuppressOutput(object):
    STD_OUT = 1
    def __init__(self, suppress_output=True):
        self.suppress_output = suppress_output

    def __enter__(self):
        if self.suppress_output:
            #store default (current) stdout
            self.default_stdout = os.dup(self.STD_OUT)
            self.devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self.devnull, self.STD_OUT)
        return self

    def __exit__(self, *args):
        if self.suppress_output:
            os.dup2(self.default_stdout, self.STD_OUT)
            os.close(self.devnull)
            os.close(self.default_stdout)


def ensure_array(x):
    '''
    if x is None, returns None. Otherwise, gives x in a form so that each of:
    `len(x)`, `x[0]`, `x+2` will not fail.
    '''
    if x is None:
        return None
    elif not isinstance(x, xr.DataArray):
        x = np.array(x)
    if x.shape == ():
        # len() and indexing will fail. Need to expand to 1-D
        if isinstance(x, xr.DataArray) and len(x.coords)>0:
            x = x.expand_dims(list(x.coords))
        else:
            x = np.array([x])
    return x

def ensure_listlike(x):
    if x is None:
        return []
    try:
        iter(x)
        return x
    except TypeError:
        return [x]

def ensure_scalar(x):
    return ensure_array(x).item()

def mkdir_p(path):
    '''
    Equivalent to mkdir -p at the shell, this function makes a
    directory and its parents as needed, silently doing nothing if it
    exists.
    '''
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass
        else: raise # pragma: no cover

def dict_without(d, keys):
    """
    Exclude a list of keys from a dictionary.

    Silently ignores any key in keys that is not in the dict (this is
    intended to be used to make sure a dict does not contain specific
    keys)
    Parameters
    ----------
    d : dict
        The dictionary to operate on
    keys : list(string)
        The keys to exclude
    returns : d2
        A copy of dict without any of the specified keys
    """
    d = copy(d)
    for key in ensure_array(keys):
        try:
            del d[key]
        except KeyError:
            pass
    return d

def updated(d, update={}, filter_none=True, **kwargs):
    """Return a dictionary updated with keys from update

    Analgous to sorted, this is an equivalent of d.update as a
    non-modifying free function

    Parameters
    ----------
    d : dict
        The dict to update
    update : dict
        The dict to take updates from

    """
    d = copy(d)
    for key, val in itertools.chain(update.items(), kwargs.items()):
        if val is not None or filter_none is False:
            d[key] = val
    return d

def repeat_sing_dims(indict, keys = 'all'):
    if keys == 'all':
        subdict = indict
    else:
       subdict = {key: indict[key] for key in keys}

    subdict = {key: ensure_array(val) for key, val in subdict.items()}
    maxlen = max([len(val) for val in subdict.values()])

    subdict={key:np.repeat(val, maxlen) for key, val in subdict.items() if len(val)==1}

    return updated(indict, subdict)

class LnpostWrapper(HoloPyObject):
    '''
    We want to be able to define a specific model.lnposterior calculation that
    only takes parameter values as an argument for passing into optimizers.
    However, individual functions can't be pickled to distribute hologram
    calculations with python multiprocessing. This class solves both issues.
    '''
    def __init__(self, model, data, new_pixels=None, minus=False):
        self.parameters = model._parameters
        self.data = data
        self.pixels = new_pixels
        self.func = model.lnposterior
        self.prefactor = -1 if minus else 1

    def evaluate(self, par_vals):
        pars_dict = {par.name:val for par, val in zip(self.parameters, par_vals)}
        return self.prefactor * self.func(pars_dict, self.data, self.pixels)


def choose_pool(parallel):
    """
    This is a remake of schwimmbad.choose_pool with a single argument that has more options.
    """
    # TODO: This function should be refactored as a factory class with methods
    #       to enable more thorough testing of imports, MPI behaviour, etc.
    if hasattr(parallel, 'map'):
        # user-defined pool
        pool = parallel
    elif parallel is None:
        # serial calculation - define dummy pool
        pool = NonePool()
    elif NO_SCHWIMMBAD:
        raise DependencyMissing('schwimmbad',
            "To perform inference calculations in parallel, install schwimmbad"
            " with \'conda install -c conda-forge schwimmbad\' or define your "
            "Strategy object with a 'parallel' keyword argument that is a "
            "multiprocessing.Pool object. To run serial calculations instead, "
            "pass in parallel=None.")
    elif isinstance(parallel, int):
        pool = schwimmbad.MultiPool(parallel)
    elif parallel is 'all':
        threads = os.cpu_count()
        pool = choose_pool(threads)
    elif parallel is 'mpi':
        pool = schwimmbad.MPIPool()
        # need to kill all non-master instances of currently running script
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    elif parallel is 'auto':
        # try mpi, otherwise go for multiprocessing
        if schwimmbad.MPIPool.enabled():
            pool = choose_pool('mpi')
        else:
            pool = choose_pool('all')
    else:
        raise TypeError("Could not interpret 'parallel' argument. Use an "
                        "integer, 'mpi', 'all', 'auto', None or pass a pool "
                        "object with 'map' method.")
    return pool

class NonePool():
    def map(self, function, arguments):
        return map(function, arguments)
    def close(self):
        pass
