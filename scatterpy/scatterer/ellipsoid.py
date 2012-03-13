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
from scatterpy.scatterer import Scatterer
from scatterpy.errors import ScattererDefinitionError

class SingleCenterScatterer(Scatterer):
    """
    Base class for scattererers which are localized around some defined center.

    Attributes
    ----------
    x, y, z : float
        x, y, z-component of center
    center : 3-tuple, list or numpy array (optional)
        specifies coordinates of center of the scatterer
    """
    
    def __init__(self, x = None, y = None, z = None, center = None):
        self.center = self._construct_xyz_tuple(x, y, z, center, 'center')

    def _construct_xyz_tuple(self, x, y, z, xyz, name_for_errors='parameter'):
        """
        supports allowing users to specify values along the three axes either as
        individual function arguments x, y, z, or as a single tuple, xyz
        """
        if xyz is not None:
            if np.isscalar(xyz) or len(xyz) != 3:
                raise ScattererDefinitionError(
                    "{1} specified as {0}, {1} should be specified as (x,"
                    " y, z)".format(xyz, name_for_errors), self)
            return np.array(xyz)
        elif (x is not None) and (y is not None) and (z is not None):
            return np.array([x, y, z])
        else:
            raise ScattererDefinitionError(
                "Invalid {0} specification, neither valid {0} tuple or "
                "x, y, z values supplied".format(name_for_errors), self)

    @property
    def x(self):
        return self.center[0]
    @property
    def y(self):
        return self.center[1]
    @property
    def z(self):
        return self.center[2]        

    
class Ellipsoid(SingleCenterScatterer):
    """
    Scattering object representing ellipsoidal scatterers

    Parameters
    ----------
    n: float
        Index of refraction
    r_x, r_y, r_z : float
        x, y, z semi-axes
    x, y, z : float
        x,y,z-component of center
    r: float or (float, float, float)
        x, y, z semi-axes of the ellipsiod
    center : 3-tuple, list or numpy array (optional)
        specifies coordinates of center of the scatterer
    """
    
    def __init__(self, n, r_x=None, r_y=None, r_z=None, x=None, y=None, z=None,
                 r=None, center=None):
        self.n = n
        self.r = self._construct_xyz_tuple(r_x, r_y, r_z, r, 'r')
        
        super(Ellipsoid, self).__init__(x, y, z, center)

    def __repr__(self):
        return ('{0}(n={n}, r={r}, center={center})'.format(
            self.__class__.__name__, n=self.n, r = list(self.r),  center =
            list(self.center))) 
