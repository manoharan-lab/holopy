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
Defines Sphere, a scattering primitive

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
'''

import numpy as np
from .scatterer import CenteredScatterer, Indicators
from ..errors import ScattererDefinitionError
from copy import copy

class Sphere(CenteredScatterer):
    '''
    Contains optical and geometrical properties of a sphere, a
    scattering primitive

    Attributes
    ----------
    n : complex
        Index of refraction of sphere
    r : float
        Radius of sphere
    center : 3-tuple, list or numpy array
        specifies coordinates of center of sphere

    '''

    def __init__(self, n = None, r = 0.5e-6, center = None):
        self.n = n
        self.r = r
        super(Sphere, self).__init__(center)

        if self.r < 0:
            raise ScattererDefinitionError("radius is negative", self)

    @property
    def indicators(self):
        if np.isscalar(self.r):
            r = self.r
            func = lambda point: (np.array(point)**2).sum() < self.r**2
        else:
            func = [lambda point: (np.array(point)**2).sum() < r**2 for r in self.r]
            r = max(self.r)

        return Indicators(func, [[-r, r], [-r, r], [-r, r]])

    def rotated(self, alpha, beta, gamma):
        return copy(self)
