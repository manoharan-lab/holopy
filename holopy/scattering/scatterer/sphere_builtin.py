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

from .scatterer import CenteredScatterer

class Sphere_builtin(CenteredScatterer):
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

    def __init__(self, n = None, r = None, center = None):
        self.n = n
        self.r = r
        super(Sphere_builtin, self).__init__(center)


