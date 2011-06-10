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
Defines SphereCluster, a Composite scatterer consisting of Spheres

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
'''

import numpy as np
from sphere import Sphere
from composite import Composite
from holopy.utility.errors import SphereClusterDefError

class SphereCluster(Composite):
    '''
    Contains optical and geometrical properties of a cluster of spheres. 

    Attributes
    ----------
    spheres : list
       Interparticle gap distance ( = 0 at hard-sphere contact.) 

    Notes
    -----
    Inherited from Composite
    '''

    def __init__(self, spheres=None):
        if spheres is None:
            self.scatterers = []
        else: 
            self.scatterers = spheres

        # make sure all components are spheres
        for s in self.scatterers:
            if not isinstance(s, Sphere):
                raise SphereClusterDefError(repr(s)+' is not a Sphere')

