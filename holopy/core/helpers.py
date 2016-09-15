# Copyright 2011-2013, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, and Ryan McGorty, Anna Wang
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
from copy import copy

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

def ensure_3d(x):
    """
    Make sure you have a 3d coordinate.

    If given a 2d coordinate, add a z=0 to make it 3d

    Parameters
    ----------
    x : list, array or tuple with 2 or 3 elements
        a the coordinate that should be 3d

    Returns
    -------
    x3d : np.ndarray
        A coordinate that has 3 elements
    """
    if len(x) not in [2, 3]:
        raise Exception("{0} cannot be interpreted as a coordinate")
    if len(x) == 2:
        return np.append(x, 0)
    else:
        return np.array(x)

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

def coord_grid(bounds, spacing=None):
    """
    Return a nd grid of coordinates

    Parameters
    ----------
    bounds : tuple of tuples or ndarray
        Upper and lower bounds of the region
    spacing : float or complex (optional)
        Spacing between points, or if complex, number of points along each
        dimension. If spacing is not provided, then bounds should be integers,
        and coord_grid will return integer indexs in that range
    """
    bounds = np.array(bounds)
    if bounds.ndim == 1:
        bounds = np.vstack((np.zeros(3), bounds)).T

    if spacing:
        if np.isscalar(spacing) or len(spacing) == 1:
            spacing = np.ones(3) * spacing
    else:
        spacing = [None, None, None]

    grid = np.mgrid[[slice(b[0], b[1], s) for b, s in
                     zip(bounds, spacing)]]
    return np.concatenate([g[...,np.newaxis] for g in grid], 3)

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

def updated(d, update):
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
    d.update(update)
    return d
