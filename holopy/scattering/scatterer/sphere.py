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
Defines Sphere, a scattering primitive

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
'''

from copy import copy

import numpy as np

from holopy.scattering.scatterer.scatterer import CenteredScatterer, Indicators
from holopy.scattering.errors import InvalidScatterer
from holopy.core.utils import ensure_array, updated


class Sphere(CenteredScatterer):
    '''
    Contains optical and geometrical properties of a sphere, a
    scattering primitive.

    This can be a multiple layered sphere by making r and n lists.

    Attributes
    ----------
    n : complex or list of complex
        index of refraction of each layer of the sphere
    r : float or list of float
        radius of the sphere or outer radius of each sphere.
    center : length 3 listlike
        specifies coordinates of center of sphere

    '''

    def __init__(self, n=None, r=.5, center=None):
        self.n = n
        self.r = r
        super().__init__(center)

        try:
            if np.any(np.array(self.r) < 0):
                raise InvalidScatterer(self, "radius is negative")
        except TypeError:
            # Simplest solution to deal with spheres with a parameter or prior
            # as arguments, just don't check them. It might be worth doing some
            # testing of the guess, but for now I am not doing that to avoid
            # introducing a dependency on something in fit
            pass

    @property
    def indicators(self):
        rs = ensure_array(self.r)
        funcs = [
            (lambda points, ri=ri: (points**2).sum(-1) < ri**2) for ri in rs]
        r = max(rs)
        return Indicators(funcs, [[-r, r], [-r, r], [-r, r]])

    def rotated(self, alpha, beta, gamma):
        return copy(self)

    @property
    def num_domains(self):
        if self.n:
            if np.isscalar(self.n):
                return 1
            else:
                return len(self.n)
        else:
            return 0


class LayeredSphere(Sphere):
    """
    Alternative description of a sphere where you specify layer
    thicknesses instead of radii

    Attributes
    ----------
    n : list of complex
        Index of each each layer
    t : list of float
        Thickness of each layer
    center : length 3 listlike
        specifies coordinates of center of sphere
    """
    def __init__(self, n=None, t=None, center=None):
        self.n = ensure_array(n)
        self.t = ensure_array(t)
        self.center = center

    @property
    def r(self):
        r = np.zeros(len(self.t))
        r[0] = self.t[0]
        for i, t in enumerate(self.t[1:]):
            r[i+1] = r[i] + t
        return r

