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

from scatterpy.scatterer import Scatterer
from scatterpy.errors import ScattererDefinitionError
from scatterpy.scatterer.abstract_scatterer import xyzTriple, InvalidxyzTriple

class SingleCenterScatterer(Scatterer):
    """
    Base class for scattererers which are localized around some defined center.

    Attributes
    ----------
    center : 3-tuple, list or numpy array
        specifies coordinates of center of the scatterer
    """
    
    def __init__(self, center = None):
        try:
            self.center = xyzTriple(center)
        except InvalidxyzTriple as e:
            raise ScattererDefinitionError("center specified as {0}, center "
                "should be specified as (x, y, z)".format(e.xyz), self)

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
        self.r = xyzTriple(r)
        
        super(Ellipsoid, self).__init__(center)
