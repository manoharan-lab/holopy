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
Defines ellipsoidal scatterers.

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
'''
from __future__ import division

import numpy as np

from .scatterer import CenteredScatterer, Indicators
from ..errors import ScattererDefinitionError

def isnumber(x):
    try:
        x + 1
        return True
    except TypeError:
        return False

def all_numbers(x):
    return reduce(lambda rest, i: isnumber(i) and rest, x, True)


class Ellipsoid(CenteredScatterer):
    """
    Scattering object representing ellipsoidal scatterers

    Parameters
    ----------
    n : complex
        Index of refraction
    r : float or (float, float, float)
        x, y, z semi-axes of the ellipsoid
    center : 3-tuple, list or numpy array
        specifies coordinates of center of the scatterer
    """

    def __init__(self, n=None, r=None, center=None):
        self.n = n

        if np.isscalar(r) or len(r) != 3:
            raise ScattererDefinitionError("r specified as {0}; "
                                           "r should be "
                                           "specified as (r_x, r_y, r_z)"
                                           "".format(center), self)

        self.r = r

        super(Ellipsoid, self).__init__(center)

    # TODO: does not handle rotations
    @property
    def indicators(self):
        return Indicators(lambda point: ((point / self.r) ** 2).sum() < 1,
                          [[-self.r[0], self.r[0]], [-self.r[1], self.r[1]],
                            [-self.r[2], self.r[2]]])
