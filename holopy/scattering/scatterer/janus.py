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
Defines a janus sphere as a scattering primitive

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
'''

import numpy as np
from .scatterer import CenteredScatterer, Indicators
from .sphere import Sphere
from ...core.math import rotation_matrix

class JanusSphere(CenteredScatterer):
    def __init__(self, n = None, r = None, rotation = (0, 0), center = None):
        """
        A Janus (two faced) sphere

        A two layer particle with the outer layer only spanning one hemisphere. A particle
        with no rotation has its cap pointing along +z, specify other orientations by euler
        angle rotations from that reference.

        Parameters
        ----------
        n : complex, complex
            Index of refraction of each layer
        r : float, float
            Outer radius of each layer
        normal : (float, float)
            Euler angles beta and gamma to rotate from the reference position
        center : (float, float, float)
            The "center" of the janus sphere. This "center" is actually the center of the full
            sphere, and is at the center of curvature of the shell, but it is not actually the
            center of mass of the whole structure.
        """
        self.n = n
        self.r = r
        self.rotation = rotation
        self.center = center

    @property
    def indicators(self):
        s0 = Sphere(r = self.r[0], center = [0, 0, 0])
        s1 = Sphere(r = self.r[1], center = [0, 0, 0])
        #TODO: check that this is the correct way to rotate a vector
        normal = np.dot(rotation_matrix(0, *self.rotation),(0, 0, 1))
        def cap(point):
            return (np.dot(point, normal) > 0) & s1.contains(point)
        r = max(self.r)
        return Indicators([s0.contains, cap], [[-r, r], [-r, r], [-r, r]])
