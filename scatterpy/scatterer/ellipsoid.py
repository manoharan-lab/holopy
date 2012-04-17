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
Defines ellipsiods and a base class for all regular scatterers that have a well
defined center.  

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
'''

import numpy as np

from scatterpy.scatterer.abstract_scatterer import SingleScatterer, all_numbers
from scatterpy.errors import ScattererDefinitionError

def isnumber(x):
    try:
        x + 1
        return True
    except TypeError:
        return False

def all_numbers(x):
    return reduce(lambda rest, i: isnumber(i) and rest, x, True)
    

class Ellipsoid(SingleScatterer):
    """
    Scattering object representing ellipsoidal scatterers

    Parameters
    ----------
    n: complex
        Index of refraction
    r: float or (float, float, float)
        x, y, z semi-axes of the ellipsiod
    center : 3-tuple, list or numpy array
        specifies coordinates of center of the scatterer
    """
    
    def __init__(self, n=None, r=None, center=None):
        if n is not None:
            self.n = complex(n)
        else:
            self.n = None

        if np.isscalar(center) or len(center) != 3 or not all_numbers(center):
            raise ScattererDefinitionError("r specified as {0}, r should be "
                                           "specified as (r_x, r_y, r_z)"
                                           "".format(center), self)
 
        self.r = r
        
        super(Ellipsoid, self).__init__(center)
