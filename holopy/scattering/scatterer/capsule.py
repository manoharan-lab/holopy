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
Defines capsule scatterers.

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
'''


import numpy as np

from .scatterer import CenteredScatterer, Indicators
from .sphere import Sphere
from ..errors import InvalidScatterer
from ...core.math import rotation_matrix
from numpy.linalg import norm


class Capsule(CenteredScatterer):
    """
        A cylinder with semi-spherical caps.

        A particle with no rotation has its long axis pointing along +z, 
        specify other orientations by euler angle rotations from that reference.

    Parameters
    ----------
    n : complex
        Index of refraction
    h : height of cylinder  
    d : diameter
    center : 3-tuple, list or numpy array
        specifies coordinates of center of the scatterer
    rotation : 3-tuple, list or numpy.array
        specifies the Euler angles (alpha, beta, gamma) in radians
    """

    def __init__(self, n=None, h=None, d=None, center=None,rotation=(0,0,0)):
        self.n = n
        self.d = d
        self.h = h

        if np.isscalar(rotation) or len(rotation) != 3:
            raise InvalidScatterer(self,"rotation specified as {0}; "
                                           "rotation should be "
                                           "specified as (alpha, beta, gamma)"
                                           "".format(rotation))
        self.rotation = rotation
        super().__init__(center)

    @property
    def indicators(self):
        normal = (self.h/2)*np.dot(rotation_matrix(*self.rotation),(0, 0, 1))
        s0 = Sphere(r = self.d/2, center = -normal)
        s1 = Sphere(r = self.d/2, center = normal)
        #TODO: check that this is the correct way to rotate a vector

        def cylinder(point): #actually makes cylinder with round tops
            subdivisions = point.shape #this gives number of subdivisions in x, y, z, 3
            point = point.reshape(-1, point.shape[-1]) #reshape into a list of coordinates, NxNyNz x 3 array
            flat_indicator_a = (norm(point,axis=1) < ((self.d/2)**2 + 
                                     (self.h/2)**2)**0.5) # sphere that circumscribes cylinder
            tiled_normals = np.tile(normal,(subdivisions[0] * subdivisions[1] *
                                            subdivisions[2], 1))
            # perpendicular distance to norm mustn't exceed r[0]
            flat_indicator_b = norm(np.transpose(
                    np.tile(np.dot(normal, np.transpose(point)),(3,1))) 
                    * tiled_normals/norm(normal)**2-point, axis=1) < self.d/2 
            flat_indicator_c = norm(np.transpose(
                    np.tile(-np.dot(-normal, np.transpose(point)),(3,1)))
                    * tiled_normals/norm(normal)**2-point, axis=1) < self.d/2
            flat_indicator = (flat_indicator_a &
                              (flat_indicator_b | flat_indicator_c))
            return flat_indicator.reshape(subdivisions[0], subdivisions[1],
                                          subdivisions[2])
        r = (self.h + 2 * self.d)/2
        return Indicators([cylinder, s0.contains, s1.contains],
                          [[-r, r], [-r, r], [-r, r]])
