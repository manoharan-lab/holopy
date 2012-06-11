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
Defines CoatedSphere, a scattering primitive

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
'''
from __future__ import division

from holopy.utility.helpers import _ensure_array
from .abstract_scatterer import SphericallySymmetricScatterer

class CoatedSphere(SphericallySymmetricScatterer):
    '''
    Sphere with concentric layers of different index.  
    
    A CoatedSphere can contain any number of layers
    
    Attributes
    ----------
    n : array(complex)
        Indices of refraction of each layer, starting from the core
    r : array(float)
        Outer radius of the each layer, starting from the core
    center : 3-tuple, list or numpy array (optional)
        specifies coordinates of center of sphere

    '''
    def __init__(self, n = 1.59, r = 0.5e-6, center = (0.0, 0.0, 0.0)):
        self.n = _ensure_array(n)
        self.r = _ensure_array(r)
        super(CoatedSphere, self).__init__(center)
