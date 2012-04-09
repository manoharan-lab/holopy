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

import numpy as np
from holopy.utility.helpers import _ensure_array
from scatterpy.scatterer.ellipsoid import SingleCenterScatterer

#TODO: failed tests are all things I think we are refactoring away -
# tgd 2012-03-13

class CoatedSphere(SingleCenterScatterer):
    '''
    Contains optical and geometrical properties of a coated sphere, a
    scattering primitive.  Core and shell are concentric.

    Attributes
    ----------
    n : array(complex)
        Index of refraction of shell
    r : array(float)
        Radius of core sphere
    center : 3-tuple, list or numpy array (optional)
        specifies coordinates of center of sphere

    '''
    def __init__(self, n = 1.59, r = 0.5e-6, center = (0.0, 0.0, 0.0)):
        self.n = _ensure_array(n).astype('complex')
        self.r = _ensure_array(r)
        super(CoatedSphere, self).__init__(center)
