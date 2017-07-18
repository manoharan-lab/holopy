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


## todo: doesn't work with DDA at the moment, but ok for T-matrix
'''
    Defines spheroidal scatterers.

    .. moduleauthor:: Anna Wang, Thomas G. Dimiduk
'''

import numpy as np

from .scatterer import CenteredScatterer
from ..errors import InvalidScatterer

class Spheroid(CenteredScatterer):
    """
    Scattering object representing spheroidal scatterers

    Attributes
    ----------
    n : complex
        Index of refraction
    r : (float, float)
        length of xy and z semi-axes of the spheroid
    rotation : 3-tuple, list or numpy array
        specifies the Euler angles (alpha, beta, gamma) in degrees
    center : 3-tuple, list or numpy array
        specifies coordinates of center of the scatterer
    """

    def __init__(self, n=None, r=None, rotation = (0, 0, 0), center=None):
        self.n = n

        if np.isscalar(r) or len(r) != 2:
            raise InvalidScatterer("r specified as {0}; "
                                           "r should be "
                                           "specified as (r_xy, r_z)"
                                           "".format(r), self)

        self.n = n
        self.r = r
        self.rotation = rotation
        self.center = center

    @property
    def indicators(self):
        inverserotate = np.linalg.inv(rotation_matrix(0, *self.rotation))
        threeaxes = np.array([self.r[0], self.r[0], self.r[1]])
        r = max(self.r)
        return Indicators(lambda point: ((np.dot(inverserotate, point.reshape(-1, 3).transpose()).transpose() / threeaxes)**2).sum(-1) < 1, [[-r, r], [-r, r], [-r, r]])
