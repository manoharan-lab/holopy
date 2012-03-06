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
from scatterpy.scatterer import Scatterer
from scatterpy.scatterer.ellipsoid import SingleCenterScatterer

class CoatedSphere(SingleCenterScatterer):
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
        super(CoatedSphere, self).__init__(x, y, z, center)

    parameter_names_list = ['n1.real', 'n1.imag', 'n2.real', 'n2.imag', 'r1',
                            'r2', 'x', 'y', 'z'] 

    def __repr__(self):
        '''
        Outputs the object parameters in a way that can be typed into
        the python interpreter
        '''
        return "{c}(center={center}, n1={n1}, n2={n2}, r1={r1}, r2={r2})".format(c=self.__class__.__name__, center=repr(self.center), n1=self.n1, n2=self.n2, r1=self.r1, r2=self.r2)

    @property
    def parameter_list(self):
        return np.array([self.n1.real, self.n1.imag, self.n2.real, self.n2.imag,
                         self.r1, self.r2, self.x, self.y, self.z])
    
    @classmethod
    def make_from_parameter_list(cls, params):
        n1 = params[0] + 1.0j * params[1]
        n2 = params[2] + 1.0j * params[3]
        return cls(n1, n2, *params[4:])

    @property
    def r(self):
        # when someone asks us for r, they want to know our physical size, so
        # give them our larger radius
        return self.r2
    

class Shell(CoatedSphere):
    """
    A CoatedSphere that you specify in terms of thickness and radus instead of
    two radii
    """
    def __init__(self, n1, n2, t, r, x = 0.0, y = 0.0, z = 0.0, center = None):
        inner_r = r-t
        super(Shell, self).__init__(n1, n2, inner_r, r, x, y, z, center)

    parameter_names_list = ['n1.real', 'n1.imag', 'n2.real', 'n2.imag', 't',
                            'r', 'x', 'y', 'z'] 


    @property
    def t(self):
        return self.r2 - self.r1

    @property
    def parameter_list(self):
        return np.array([self.n1.real, self.n1.imag, self.n2.real, self.n2.imag,
                        self.r2-self.r1, self.r2, self.x, self.y, self.z])

    # We can just use the parent class make_from_parameter_list, its parameter
    # munging is correct for us, it will call our __init__ correctly
    
    def __repr__(self):
        '''
        Outputs the object parameters in a way that can be typed into
        the python interpreter
        '''
        return "{c}(center={center}, n1={n1}, n2={n2}, t={t}, r={r})".format(c=self.__class__.__name__, center=repr(self.center), n1=self.n1, n2=self.n2, t=self.t, r=self.r)
