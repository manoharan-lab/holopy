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

import numpy as np


from . import Scatterer
from ...core.math import rotate_points
from ...core.utils import is_none, ensure_array

class Scatterers(Scatterer):
    '''
    Contains optical and geometrical properties of a a composite
    scatterer.  A Scatterers can consist of multiple scattering
    primitives (e.g. Sphere) or other Scatterers scatterers.

    Attributes
    ----------
    scatterers: list
       List of scatterers that make up this object

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
        if scatterers is None:
            self.scatterers = []
        else:
            self.scatterers = scatterers

    def add(self, scatterer):
        self.scatterers.append(scatterer)

    def get_component_list(self):
        components = []
        for s in self.scatterers:
            if isinstance(s, self.__class__):
                components += s.get_component_list()
            else:
                components.append(s)
        return components

    @property
    def parameters(self):
        d = {}
        for i, scatterer in enumerate(self.scatterers):
            for key, par in scatterer.parameters.items():
                d['{0}:{1}.{2}'.format(i, scatterer.__class__.__name__, key)] = par
        return dict(sorted(list(d.items()), key = lambda t: t[0]))

    @classmethod
    def from_parameters(cls, parameters):
        n_scatterers = len(set([p.split(':')[0] for p in list(parameters.keys())]))
        collected = [{} for i in range(n_scatterers)]
        types = [None] * n_scatterers
        for key, val in parameters.items():
            n, spec = key.split(':', 1)
            n = int(n)
            scat_type, par = spec.split('.', 1)

            collected[n][par] = val
            if types[n]:
                assert types[n] == scat_type
            else:
                types[n] = scat_type

        scatterers = []
        # pull in the scatterer package, this lets us grab scatterers by class
        # name
        # we have to do it here rather than at the top of the file because we
        # cannot import scatterer until it is done importing, which will not
        # happen until import of composite finishes.
        from .. import scatterer
        for i, scat_type in enumerate(types):
            scatterers.append(getattr(scatterer,
                              scat_type).from_parameters(collected[i]))

        return cls(scatterers)

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
        if is_none(coord2) and len(ensure_array(coord1)==3):
            #entered translation vector
            trans_coords = ensure_array(coord1)
        elif not is_none(coord2) and not is_none(coord3):
            #entered 3 coords
            trans_coords = np.array([coord1, coord2, coord3])
        else:
            raise InvalidScatterer(self, "Cannot interpret translation coordinates")

        trans = [s.translated(trans_coords) for s in self.scatterers]
        new = copy(self)
        new.scatterers = trans
        return new

    def rotated(self, ang1, ang2=None, ang3=None):

        if is_none(ang2) and len(ensure_array(ang1)==3):
            #entered rotation angle tuple
            alpha, beta, gamma = ang1
        elif not is_none(ang2) and not is_none(ang3):
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
