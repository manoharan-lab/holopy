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

'''
The abstract base class for all scattering objects

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
'''
from __future__ import division
from collections import defaultdict

from ...core.helpers import OrderedDict

from itertools import chain
from copy import copy

import numpy as np

from ...core.holopy_object  import HoloPyObject
from ...core.helpers import _ensure_array
from ..errors import ScattererDefinitionError


class Scatterer(HoloPyObject):
    """
    Base class for scatterers

    """
    def __init__(self, indicators, n, location):
        """
        Parameters
        ----------
        indicators : function or list of functions
            Function or functions returning true for points inside the scatterer (or
            inside a specific domain) and false outside.
        n : complex
            Index of refraction of the scatterer or each domain.
        bounding_box : ((float, float), (float, float), (float, float))
            Optional. Box containing the scatterer. If a bounding box is not given, the
            constructor will attempt to determine one.
        """
        if not isinstance(indicators, Indicators):
            indicators = Indicators(indicators)
        self.indicators = indicators
        self.n = _ensure_array(n)
        self.location = np.array(location)

    def translated(self, x, y, z):
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
        new = copy(self)
        new.location = self.location + np.array((x, y, z))
        return new

    def contains(self, points):
        return self.in_domain(points) > 0

    def index_at(self, points, background = 0):
        domains = self.in_domain(points)
        ns = _ensure_array(self.n)
        if np.iscomplex(np.append(self.n, background)).any():
            dtype = np.complex
        else:
            dtype = np.float
        index = np.ones_like(domains, dtype=dtype) * background
        for i, n in enumerate(ns):
            index[domains==i+1] = n
        return index

    def in_domain(self, points):
        """
        Tell which domain of a scatterer points are in

        Paramaters
        ----------
        points : np.ndarray (Nx3)
           Point or list of points to evaluate

        Returns
        -------
        domain : np.ndarray (N)
           The domain of each point. Domain 0 means not in the particle
        """
        points = np.array(points)
        if points.ndim==1:
            points = points.reshape((1, 3))
        domains = np.zeros(points.shape[:-1], dtype='int')
        # Indicators earlier in the list have priority
        for i, ind in reversed(list(enumerate(self.indicators(points-self.location)))):
            domains[np.nonzero(ind)] = i+1
        return domains

    def _index_type(self, background=0.):
        if np.iscomplex([self.n]).any() or np.iscomplex(background):
            return np.complex
        else:
            return np.float

    @property
    def x(self):
        return self.location[0]
    @property
    def y(self):
        return self.location[1]
    @property
    def z(self):
        return self.location[2]

    def like_me(self, **overrides):
        pars = dict(self._dict)
        pars.update(overrides)

        return self.__class__(**pars)

    @property
    def bounds(self):
        return [(c+b[0], c+b[1]) for c, b in zip(self.location,
                                                 self.indicators.bound)]

    def _voxel_coords(self, spacing):
        if np.isscalar(spacing) or len(spacing) == 1:
            spacing = np.ones(3) * spacing

        grid = np.mgrid[[slice(b[0], b[1], s) for b, s in
                            zip(self.bounds, spacing)]]
        return np.concatenate([g[...,np.newaxis] for g in grid], 3)

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
    def __init__(self, center = None):
        if center is not None and (np.isscalar(center) or len(center) != 3):
            raise ScattererDefinitionError("center specified as {0}, center "
                "should be specified as (x, y, z)".format(center), self)
        self.location = center

    @property
    def center(self):
        return self.location

    @center.setter
    def center(self, val):
        self.location = val

    # eliminate parameters and from_parameters?  This is kind of fitting
    # specific information.  Or should it be in serializable?  In many ways this
    # is just a slight variation on what we do to put something in yaml format.
    # It is probably possible to have to_dict, to_string, to_yaml all with
    # mostly common code
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
            A dictionary of this scatterer's parameters.  This dict can be
            passed to Scatterer.from_parameters to make a copy of this scatterer
        """

        # classes that have anything complicated happening with their variables
        # should override this, but for simple classes the variable dict is the
        # correct answer

        # we return an OrderedDict to make it easer to keep parameters in the
        # same order in cases where a list of parameters is needed and will be
        # later passed to Scatterer.from_parameters

        def expand(key, par):
            if isinstance(par, (list, tuple, np.ndarray)):
                subs = (expand('{0}[{1}]'.format(key, p[0]), p[1]) for p in enumerate(par))
                return chain(*subs)
            else:
                return [(key, par)]

        return OrderedDict(sorted(chain(*[expand(*p) for p in
                                          self._dict.iteritems()])))

    @classmethod
    def from_parameters(cls, parameters):
        """
        Create a Scatterer from a dictionary of parameters

        Parameters
        ----------
        parameters: dict or list
            Parameters for a scatterer.  This should be of the form returned by
            Scatterer.parameters.

        Returns
        -------
        scatterer: Scatterer class
            A scatterer with the given parameter values
        """
        # This will need to be overriden for subclasses that do anything
        # complicated with parameters

        collected = defaultdict(dict)

        for key, val in parameters.iteritems():
            tok = key.split('.', 1)
            if len(tok) > 1:
                collected[tok[0]][tok[1]] = val
            else:
                collected[key] = val

        collected_arrays = defaultdict(dict)
        for key, val in collected.iteritems():
            tok = key.split('[', 1)
            if len(key.split('[', 1)) > 1:
                sub_key, n = key.split('[', 1)
                n = int(n[:-1])
                collected_arrays[sub_key][n] = val
            else:
                collected_arrays[key] = val

        built = {}

        def build(par):
            if isinstance(par, dict):
                reduce(lambda x, i: isinstance(i, int) and x,
                       par.keys(), True)
                d = [p[1] for p in sorted(par.iteritems(), key =
                                          lambda x: x[0])]
                return [build(p) for p in d]
            return par

        for key, val in collected_arrays.iteritems():
            built[key] = build(val)

        return cls(**built)



class SingleScatterer(Scatterer):
    def __init__(self, center = None):
        if center is not None and (np.isscalar(center) or len(center) != 3):
            raise ScattererDefinitionError("center specified as {0}, center "
                "should be specified as (x, y, z)".format(center), self)
        self.center = center

    def translated(self, x, y, z):
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
        new = copy(self)
        new.center = self.center + np.array([x, y, z])
        return new
    @property
    def x(self):
        return self.center[0]
    @property
    def y(self):
        return self.center[1]
    @property
    def z(self):
        return self.center[2]

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
    new = [[0, 0],[0, 0],[0, 0]]
    for i in range(3):
        new[i][0] = min(d1[i][0], d2[i][0])
        new[i][1] = max(d1[i][1], d2[i][1])
    return new

class Indicators(HoloPyObject):
    """
    Class holding functions describing a scatterer

    One or more functions (one per domain) that take Nx3 arrays of points and
    return a boolean array of membership in each domain. More than one indicator
    is allowed to return true for a given point, in that case the point is
    considered a member of the first domain with a true value.
    """
    def __init__(self, functions, bound = None):
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
