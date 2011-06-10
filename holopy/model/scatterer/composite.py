# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca
# W. Perry, Jerome Fung, and Ryan McGorty
#
# This file is part of Holopy.
#
# Holopy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Holopy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Holopy.  If not, see <http://www.gnu.org/licenses/>.

'''
Defines Composite, a scatterer that consists of other scatterers,
including scattering primitives (e.g. Sphere) or other Composite
scatterers (e.g. two trimers).

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
'''

import numpy as np
import pprint
from sphere import Sphere

class Composite(object):
    '''
    Contains optical and geometrical properties of a a composite
    scatterer.  A Composite can consist of multiple scattering
    primitives (e.g. Sphere) or other Composite scatterers. 

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

    def _contains_only_spheres(self):
        """
        Returns True if the Composite object contains only spheres.
        Used mainly for error checking by models that can only handle
        spheres
        """
        
        for scatterer in self.get_component_list():
            if not isinstance(scatterer, Sphere):
                return False
        return True

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
                out = out + (level+1)*indent + s.__class__.__name__ + '\n'
        return out
        
    def __str__(self):
        '''
        Pretty print the nested tree of scatterers
        '''
        return self._prettystr(0)

