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
Do Constructive Solid Geometry (CSG) with scatterers. Currently only useful with
the DDA th

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
'''
from holopy.scattering.errors import InvalidScatterer
from holopy.scattering.scatterer import Scatterer
from holopy.core.math import rotate_points

import numpy as np
from numpy import logical_and, logical_not, logical_or
from copy import copy

class CsgScatterer(Scatterer):
    def __init__(self, s1, s2):
        # For now we just treat s1's center as the center of the composite.
        # This is probably not the best way to do it, but it is simple to
        # implement for now.
        self.center = s1.center
        self.s1 = s1
        self.s2 = s2
        if s1.num_domains > 1 or s2.num_domains > 1:
            raise InvalidScatterer(self, "Scatterer CSG is not supported for multidomain scatterers")
        if s1.n is None:
            s1 = copy(s1)
            s1.n = s2.n
        self.n = s1.n
        if s1.n != s2.n:
            raise InvalidScatterer(self, "Components of a CSG scatterer must not have different indicies")

    @property
    def bounds(self):
        return [(min(b1[0], b2[0]), max(b1[1], b2[1])) for b1, b2 in zip(self.s1.bounds, self.s2.bounds)]

    def rotated(self, alpha, beta, gamma):
        centers = np.array([s.center for s in (self.s1, self.s2)])
        new_centers = self.center + rotate_points(centers - self.center, alpha, beta, gamma)

        s1, s2 = [s.translated(*(c-n)).rotated(alpha, beta, gamma) for s, c, n
                  in zip((self.s1, self.s2), centers, new_centers)]
        return self.__class__(s1, s2)


class Union(CsgScatterer):
    def in_domain(self, points):
        return np.logical_or(self.s1.in_domain(points), self.s2.in_domain(points))


class Difference(CsgScatterer):
    def in_domain(self, points):
        return np.logical_and(self.s1.in_domain(points), np.logical_not(self.s2.in_domain(points)))

    @property
    def bounds(self):
        # this isn't as good as we can do, but it is at least correct
        return self.s1.bounds

class Intersection(CsgScatterer):
    def in_domain(self, points):
        return np.logical_and(self.s1.in_domain(points), self.s2.in_domain(points))
