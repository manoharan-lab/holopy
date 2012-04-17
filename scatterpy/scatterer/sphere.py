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
Defines Sphere, a scattering primitive

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
'''

from .abstract_scatterer import SphericallySymmetricScatterer

class Sphere(SphericallySymmetricScatterer):
    '''
    Contains optical and geometrical properties of a sphere, a
    scattering primitive

    Attributes
    ----------
    n : complex
        Index of refraction of sphere
    r : float
        Radius of sphere
    x : float
        x-component of center
    y : float
        y-component of center
    z : float
        z-component of center
    center : 3-tuple, list or numpy array (optional)
        specifies coordinates of center of sphere

    '''

    def __init__(self, n = 1.59, r = 0.5e-6, center = (0, 0, 0)):
        self.n = complex(n)
        self.r = r
        super(Sphere, self).__init__(center)

