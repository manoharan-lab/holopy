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
import shutil
import errno
import numpy as np
import xarray as xr
from copy import copy
import itertools

def _ensure_array(x):
    if np.isscalar(x):
        return np.array([x])
    else:
        return np.array(x)

def ensure_listlike(x):
    if x is None:
        return []
    try:
        iter(x)
        return x
    except TypeError:
        return [x]

def _ensure_pair(x):
    if x is None:
        return None
    try:
        x[1]
        return np.array(x)
    except (IndexError, TypeError):
        return np.array([x, x])

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
    Exclude a list of keys from a dictionary
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
    for key in _ensure_array(keys):
        try:
            del d[key]
        except KeyError:
            pass
    return d

def is_none(o):
    """
    Check if something is None.

    This can't be done with a simple is check anymore because numpy decided that
    array is None should do an element wise comparison.

    Parameters
    ----------
    o : object
        Anything you want to see if is None
    """

    return isinstance(o, type(None))

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
        if val is not None:
            d[key] = val

    return d

def copy_metadata(old, new, do_coords=True):
    def find_and_rename(oldkey, oldval):
        for newkey, newval in new.coords.items():
            if np.array_equal(oldval.values, newval.values):
                return new.rename({newkey: oldkey})
            raise ValueError("Coordinate {} does not appear to have a coresponding coordinate in {}".format(oldkey, new))

    if hasattr(old, 'attrs') and hasattr(old, 'name') and hasattr(old, 'coords'):
        if not hasattr(new,'coords'):
            #new is a numpy array, not xarray
            new=xr.DataArray(new, dims=['x', 'y'])
        new.attrs = old.attrs
        new.name = old.name
        if hasattr(old, 'z') and not hasattr(new, 'z'):
            new.coords['z'] = old.coords['z']
        if hasattr(old, 'flat') and hasattr(new, 'flat'):
            new['flat'] = old['flat']
        if do_coords:
            for key, val in old.coords.items():
                if key not in new.coords:
                    new = find_and_rename(key, val)
    return new

def get_values(a):
    return getattr(a, 'values', a)

def flat(a, keep_xy=True):
    if hasattr(a, 'flat'):
        # TODO handle case where we have flat but not xyz
        return a
    if hasattr(a, 'x') and hasattr(a, 'y') and keep_xy:
        a['x_orig'] = a.x
        a['y_orig'] = a.y
        # TODO: remove *_orig coords from a or avoid adding them
        f = a.stack(flat=a.dims)
        del a['x_orig']
        del a['y_orig']
        return f.rename({'x_orig': 'x', 'y_orig': 'y'})
    else:
        return a.stack(flat=a.dims)

def from_flat(a):
    if hasattr(a, 'flat'):
        return a.unstack('flat')
    return a

def make_subset_data(data, random_subset, return_selection=False):
    if random_subset is None:
        return data
    n_sel = int(np.ceil(data.size*random_subset))
    selection = np.random.choice(data.size, n_sel, replace=False)
    subset = flat(data)[selection]
    subset = copy_metadata(data, subset, do_coords=False)
    if return_selection:
        return subset, selection
    else:
        return subset
