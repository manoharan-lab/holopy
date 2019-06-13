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
Defines cylinder scatterers.

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
'''


import numpy as np

from .scatterer import CenteredScatterer, Indicators
from ..errors import InvalidScatterer

class Bisphere(CenteredScatterer):
    """
    Scattering object representing bisphere scatterers

    Parameters
    ----------
    n : complex
        Index of refraction
    h : distance between centers
    d : diameter
    center : 3-tuple, list or numpy array
        specifies coordinates of center of the scatterer
    rotation : 3-tuple, list or numpy.array
        specifies the Euler angles (alpha, beta, gamma) in radians 
        defined in a-dda manual section 8.1
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
