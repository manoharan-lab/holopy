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
'''

import numpy as np

class CoatedSphere(object):
    '''
    Contains optical and geometrical properties of a coated sphere, a
    scattering primitive.  Core and shell are concentric.

    Attributes
    ----------
    n1 : float or complex
        Index of refraction of core sphere
    n2 : float or complex
        Index of refraction of shell
    r1 : float
        Radius of core sphere
    r2 : float
        Radius of core+shell
    x : float
        x-component of center
    y : float
        y-component of center
    z : float
        z-component of center
    center : 3-tuple, list or numpy array (optional)
        specifies coordinates of center of sphere

    '''

    def __init__(self, n1 = 1.59, n2 = 1.33, r1 = 0.5e-6, r2 = 1e-6, 
                 x = 0.0, y = 0.0, z = 0.0, center = None):
        self.n1 = n1
        self.n2 = n2
        self.r1 = r1
        self.r2 = r2
        if center is not None:
            self.center = np.array(center)
        else:
            self.center = np.array([x, y, z])

    def __repr__(self):
        '''
        Outputs the object parameters in a way that can be typed into
        the python interpreter
        '''
        return "{c}(center={center}, n1={n1}, n2={n2}, r1={r1}, r2={r2})".format(c=self.__class__.__name__, center=repr(self.center), n1=self.n1, n2=self.n2, r1=self.r1, r2=self.r2)
