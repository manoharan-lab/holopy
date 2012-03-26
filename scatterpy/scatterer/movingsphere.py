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
Defines MovingSphere, a sphere that moves with some velocity sufficiently
high that it cannot be treated as stationary during the integration time
of some optical sensor.

We treat MovingSphere as a Composite scatterer with overlaps. However,
any ScatteringTheory needs to superpose hologram intensities, not fields,
from its components.

.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>
'''

#COVERAGE: All lines not covered by tests should be on their way out in refactor

from scatterpy.scatterer.sphere import Sphere
from scatterpy.scatterer.composite import Composite
import numpy as np
from numpy import arange, floor

class MovingSphere(Sphere, Composite):
    '''
    Contains optical and geometric properties of a rapidly moving sphere.
    
    Attributes:
    ----------
    Same as sphere, except
    v_x : float
          x-component of sphere velocity
    v_y : float
          y-component of sphere velocity
    v_z : float
          z-component of sphere velocity
    int_time : float
          integration time of optical sensor
    n_smear : integer
          number of intensities that are summed together. For holograms,
          choose n_smear such that the particle moves << a wavelength 
          between frames.
    '''

    def __init__(self, n = 1.59, r = 0.5e-6, x = 0.0, y = 0.0, z = 0.0,
                 v_x = 0.0, v_y = 0.0, v_z = 0.0, int_time = 1e-6, 
                 n_smear = 10, center = None):
        Sphere.__init__(self, n, r, x, y, z, center)
        self.v_x = v_x
        self.v_y = v_y
        self.v_z = v_z
        self.int_time = int_time
        self.n_smear = n_smear
        # determine positions of the individual spheres
        if n_smear%2. == 0: #even number of steps
            n_step = arange(n_smear) - (n_smear/2. - 0.5)
        else:
            n_step = arange(n_smear) - floor(n_smear/2.)
        dt = int_time/n_smear
        self.x_centers = self.center[0] + n_step * v_x * dt
        self.y_centers = self.center[1] + n_step * v_y * dt
        self.z_centers = self.center[2] + n_step * v_z * dt
        self.scatterers = [Sphere(n, r, xc, yc, zc) for (xc, yc, zc) in 
                           zip(self.x_centers, self.y_centers, self.z_centers)]

    def __repr__(self):
        '''
        Outputs the object parameters in a way that can be typed into
        the python interpreter
        '''
        return "{c}(center={center}, n={n}, r={r})".format(
            c=self.__class__.__name__, center=repr(list(self.center)), n=self.n,
            r=self.r, v_x = self.v_x, v_y = self.v_y, v_z = self.v_z,
            int_time = self.int_time, n_smear = self.n_smear)

    @property
    def parameter_list(self):
        """
        Return moving sphere parameters in order: n, r, x, y, z, v_x, v_y, v_z,
        int_time, n_smear
        """
        return np.array([self.n.real, self.n.imag, self.r, self.x, self.y,
                         self.z, self.v_x, self.v_y, self.v_z, self.int_time,
                         self.n_smear])

    @property
    def parameter_names_list(self):
        return ['n.real', 'n.imag', 'r', 'x', 'y', 'z', 'v_x', 'v_y', 'v_z',
                'int_time', 'n_smear']



        
