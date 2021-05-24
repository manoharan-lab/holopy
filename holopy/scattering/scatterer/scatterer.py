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

'''
The abstract base class for all scattering objects

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
'''

from collections import defaultdict
from itertools import chain
from copy import copy, deepcopy
from numbers import Number

import numpy as np
import xarray as xr

from holopy.core.holopy_object import HoloPyObject
from holopy.core.utils import ensure_array
from holopy.scattering.errors import (
    InvalidScatterer, ParameterSpecificationError)


class Scatterer(HoloPyObject):
    """
    Base class for scatterers

    """
    def __init__(self, indicators, n, center):
        """
        Parameters
        ----------
        indicators : function or list of functions
            Function or functions returning true for points inside the
            scatterer (or inside a specific domain) and false outside.
        n : complex
            Index of refraction of the scatterer or each domain.
        center : (float, float, float)
            The center of mass of the scatterer.
        """
        if not isinstance(indicators, Indicators):
            indicators = Indicators(indicators)
        self.indicators = indicators
        self.n = ensure_array(n)
        self.center = np.array(center)

    @property
    def x(self):
        return self.center[0]

    @property
    def y(self):
        return self.center[1]

    @property
    def z(self):
        return self.center[2]

    def translated(self, coord1, coord2=None, coord3=None):
        """
        Make a copy of this scatterer translated to a new location

        Parameters
        ----------
        x, y, z : float
            Value of the translation along each axis

        Returns
        -------
        translated : Scatterer
            A copy of this scatterer translated to a new location
        """
        if coord2 is None and len(ensure_array(coord1) == 3):
            # entered translation vector
            trans_coords = ensure_array(coord1)
        elif coord2 is not None and coord3 is not None:
            # entered 3 coords
            trans_coords = np.array([coord1, coord2, coord3])
        else:
            raise InvalidScatterer(
                self, "Cannot interpret translation coordinates")
        new = copy(self)
        new.center = self.center + trans_coords
        return new

    @property
    def _parameters(self):
        # classes that have anything complicated happening with their variables
        # should override this, but for simple classes the variable self._dict
        # is the correct answer
        return self._dict

    @property
    def parameters(self):
        """
        Get a dictionary of this scatterer's parameters

        Parameters
        ----------
        None

        Returns
        -------
        parameters: dict
            A dictionary of this scatterer's parameters. This dict can be
            passed to Scatterer.from_parameters to make a copy of this
            scatterer
        """
        return deepcopy(self._parameters)

    def from_parameters(self, parameters):
        """
        Create a Scatterer from a dictionary of parameters

        Parameters
        ----------
        parameters: dict
            Parameters for a scatterer.  This should be of the form returned by
            Scatterer.parameters.
        Returns
        -------
        scatterer: Scatterer class
            A scatterer with the given parameter values
        """
        # This will need to be overriden for subclasses that do anything
        # complicated with parameters
        parameters = {key: parameters[key] if key in parameters else
                      self.parameters[key] for key in self.parameters.keys()}
        return type(self)(**parameters)

    def contains(self, points):
        return self.in_domain(points) > 0

    def index_at(self, points, background=0):
        domains = self.in_domain(points)
        ns = ensure_array(self.n)
        if np.iscomplex(np.append(self.n, background)).any():
            dtype = complex
        else:
            dtype = float
        index = np.ones_like(domains, dtype=dtype) * background
        for i, n in enumerate(ns):
            index[domains == i + 1] = n
        return index

    def in_domain(self, points):
        """
        Tell which domain of a scatterer points are in

        Parameters
        ----------
        points : np.ndarray (Nx3)
           Point or list of points to evaluate

        Returns
        -------
        domain : np.ndarray (N)
           The domain of each point. Domain 0 means not in the particle
        """
        points = np.array(points)
        if points.ndim == 1:
            points = points.reshape((1, 3))
        domains = np.zeros(points.shape[:-1], dtype='int')
        indicators = self.indicators(points - self.center)
        # Indicators earlier in the list have priority
        for i, ind in reversed(list(enumerate(indicators))):
            domains[np.nonzero(ind)] = i + 1
        return domains

    @property
    def num_domains(self):
        return len(self.indicators)

    def _index_type(self, background=0.):
        if np.iscomplex([self.n]).any() or np.iscomplex(background):
            return complex
        else:
            return float

    @property
    def bounds(self):
        return [(c+b[0], c+b[1]) for c, b in zip(self.center,
                                                 self.indicators.bound)]

    def _voxel_coords(self, spacing):
        if np.isscalar(spacing) or len(spacing) == 1:
            spacing = np.ones(3) * spacing

        grid = np.mgrid[
            [slice(b[0], b[1], s) for b, s in zip(self.bounds, spacing)]]
        return np.concatenate([g[..., np.newaxis] for g in grid], 3)

    def voxelate(self, spacing, medium_index=0):
        """
        Represent a scatterer by discretizing into voxels

        Parameters
        ----------
        spacing : float
            The spacing between voxels in the returned voxelation
        medium_index : float
            The background index of refraction to fill in at regions where the
            scatterer is not present

        Returns
        -------
        voxelation : np.ndarray
            An array with refractive index at every pixel
        """
        return self.index_at(self._voxel_coords(spacing))

    def voxelate_domains(self, spacing):
        return self.in_domain(self._voxel_coords(spacing))


class CenteredScatterer(Scatterer):
    def __init__(self, center=None):
        if center is not None and (np.isscalar(center) or len(center) != 3):
            msg = ("center specified as {0}, "
                   "center should be specified as (x, y, z)".format(center))
            raise InvalidScatterer(self, msg)
        self.center = center


def find_bounds(indicator):
    """
    Finds the bounds needed to contain an indicator function

    Notes
    -----
    Will probably determine incorrect bounds for functions which are not convex

    """
    # we don't know what units the user might be using, so start by
    # assuming something really small and stepping up from there
    bounds = [[-1e-9, 1e-9], [-1e-9, 1e-9], [-1e-9, 1e-9]]
    for i in range(3):
        for j in range(2):
            point = np.zeros(3)
            point[i] = bounds[i][j]
            # find the extent along this axis by sequential logarithmic search
            while indicator(point):
                point[i] *= 10
            iter = 0
            while not indicator(point) and iter < 10:
                point[i] /= 2
                iter += 1
            while indicator(point):
                point[i] *= 1.1
            bounds[i][j] = point[i]

    #TODO: handle non convex functions
    #TODO: handle functions not containing the origin

    #TODO: add a check along the boundaries of the square to make sure
    #something like an oblique ellipsoid doesn't get missed'
    return bounds


def bound_union(d1, d2):
    new = [[0, 0], [0, 0], [0, 0]]
    for i in range(3):
        new[i][0] = min(d1[i][0], d2[i][0])
        new[i][1] = max(d1[i][1], d2[i][1])
    return new


class Indicators(HoloPyObject):
    """
    Class holding functions describing a scatterer

    One or more functions (one per domain) that take Nx3 arrays of points and
    return a boolean array of membership in each domain. More than one
    indicator is allowed to return true for a given point, in that case the
    point is considered a member of the first domain with a true value.
    """
    def __init__(self, functions, bound=None):
        try:
            len(functions)
        except TypeError:
            functions = [functions]
        self.functions = functions
        if bound is not None:
            self.bound = bound
        else:
            self.bound = [[0, 0], [0, 0], [0, 0]]
            for function in functions:
                self.bound = bound_union(self.bound, find_bounds(function))

    def __call__(self, points):
        return [test(points) for test in self.functions]
