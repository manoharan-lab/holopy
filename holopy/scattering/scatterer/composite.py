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
Defines Scatterers, a scatterer that consists of other scatterers,
including scattering primitives (e.g. Sphere) or other Scatterers
scatterers (e.g. two trimers).

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
'''


from copy import copy
from numbers import Number
import warnings

import numpy as np

from holopy.scattering.scatterer.scatterer import Scatterer
from holopy.core.math import rotate_points
from holopy.core.utils import ensure_array, dict_without


class Scatterers(Scatterer):
    '''
    Contains optical and geometrical properties of a a composite
    scatterer.  A Scatterers can consist of multiple scattering
    primitives (e.g. Sphere) or other Scatterers scatterers.

    Attributes
    ----------
    scatterers : list
        List of scatterers that make up this object

    Methods
    -------
    parameters [property]
        Dictionary of composite's parameters
    add(scatterer)
        Adds a new scatterer to the composite.
    from_parameters
    translated
    rotated

    Notes
    -----
    Stores information about components in a tree.  This is the most
    generic container for a collection of scatterers.
    '''

    # this uses the composite design pattern
    # see http://en.wikipedia.org/wiki/Composite_pattern
    # and
    # http://stackoverflow.com/questions/1175110/python-classes-for-simple-gtd-app
    # for a python example

    def __init__(self, scatterers=None):
        '''
        Parameters
        ----------
        scatterers : list
            List of scatterers that make up this object
        '''
        if scatterers is None:
            scatterers = []
        self.scatterers = scatterers

    def add(self, scatterer):
        self.scatterers.append(scatterer)

    def __getitem__(self, key):
        return self.scatterers[key]

    def get_component_list(self):
        components = []
        for s in self.scatterers:
            if isinstance(s, self.__class__):
                components += s.get_component_list()
            else:
                components.append(s)
        return components

    @property
    def _parameters(self):
        parameters = {}
        for i, scatterer in enumerate(self.scatterers):
            single_scatterer_parameters = {
                '{0}:{1}'.format(i, key): val for key, val in
                scatterer._parameters.items()}
            parameters.update(single_scatterer_parameters)
        return parameters

    def from_parameters(self, new_parameters):
        '''
        Makes a new object similar to self with values as given in parameters.
        This returns a physical object, so any priors are replaced with their
        guesses if not included in passed-in parameters.

        Parameters
        ----------
        parameters : dict
            dictionary of parameters to use in the new object.
            Keys should match those of self.parameters.
        overwrite : bool (optional)
            if True, constant values are replaced by those in parameters
        '''
        n_scatterers = len(self.scatterers)
        collected = [{} for i in range(n_scatterers)]
        for key, val in new_parameters.items():
            parts = key.split(':', 1)
            if len(parts) == 2:
                n, par = parts
                collected[int(n)][par] = val
        scatterers = [scat.from_parameters(pars)
                         for scat, pars in zip(self.scatterers, collected)]
        self_dict = dict(self._iteritems())
        self_dict['scatterers'] = scatterers
        return type(self)(**self_dict)

    def _prettystr(self, level, indent="  "):
        '''
        Generate pretty string representation of object by recursion.
        Used by __str__.
        '''
        out = level*indent + self.__class__.__name__ + '\n'
        for s in self.scatterers:
            if isinstance(s, self.__class__):
                out = out + s._prettystr(level+1)
            else:
                out = out + (level+1)*indent + s.__str__() + '\n'
        return out

    def __str__(self):
        '''
        Pretty print the nested tree of scatterers
        '''
        return self._prettystr(0)


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
        if coord2 is None and len(ensure_array(coord1)==3):
            #entered translation vector
            trans_coords = ensure_array(coord1)
        elif coord2 is not None and coord3 is not None:
            #entered 3 coords
            trans_coords = np.array([coord1, coord2, coord3])
        else:
            raise InvalidScatterer(self, "Cannot interpret translation coordinates")

        trans = [s.translated(trans_coords) for s in self.scatterers]
        new = copy(self)
        new.scatterers = trans
        return new

    def rotated(self, ang1, ang2=None, ang3=None):

        if ang2 is None and len(ensure_array(ang1)==3):
            #entered rotation angle tuple
            alpha, beta, gamma = ang1
        elif ang2 is not None and ang3 is not None:
            #entered 3 angles
            alpha=ang1; beta=ang2; gamma=ang3
        else:
            raise InvalidScatterer(self, "Cannot interpret rotation coordinates")

        centers = np.array([s.center for s in self.scatterers])
        com = centers.mean(0)

        new_centers = com + rotate_points(centers - com, alpha, beta, gamma)

        scatterers = []

        for i in range(len(self.scatterers)):
            scatterers.append(self.scatterers[i].translated(
                *(new_centers[i,:] - centers[i,:])).rotated(alpha, beta, gamma))

        new = copy(self)
        new.scatterers = scatterers

        return new

    def in_domain(self, points):
        ind = self.scatterers[0].contains(points).astype('int')
        for i, s in enumerate(self.scatterers[1:]):
            contained = s.contains(points)
            nz = np.nonzero(contained)
            ind[nz] = i+1
        return ind

    def index_at(self, point):
        try:
            # This will pick out the first scatterer if you have
            # multiple overlapping ones. You shouldn't really have
            # overlapping scatterers with different indicies, so this
            # shouldn't be a problem
            return self.scatterers[self.in_domain(point)[0]].index_at(point)
        except TypeError:
            return None
