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
Defines specific rigid sphere cluster geometries.

.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>
'''

import numpy as np
from numpy import array, sin, cos, dot, zeros, pi
from holopy.process.math import rotation_matrix
from sphere import Sphere
from spherecluster import SphereCluster
from scatterpy.errors import ScattererDefinitionError

class BentTrimer(SphereCluster):
    '''
    Defines a bent three-particle cluster.  Inherits from SphereCluster.
    
    The reference geometry is that in an unrotated cluster (euler_alpha = 
    euler_beta = euler_gamma = 0), particle #1 lies a distance 
    +(r_1 + r_2 + gap_01) xhat from the central particle (#0).  Particle #2 
    lies in the x-y plane; the angle defined by particles 2, 0, and 1 is theta.
    Euler angles follow the SCSMFO zyz convention. 

    Attributes added to SphereCluster
    ---------------------------------
    gap_01 : float
        gap distance between central sphere (#0) and #1
    gap_02 : float
        gap distance between spheres #0 and #2
    theta : float
        angle subtended by spheres 1, 0 and 2
    euler_alpha : float
        Euler angle for rotation about z axis, degrees
    euler_beta : float
        Euler angle for rotation about y axis, degrees
    euler_gamma : float
        Euler angle for rotation about z axis, degrees

    Notes
    -----
    Initialize by specifying array-like n (3 elements), array-like r, 
    and float positions xc, yc, zc of the central particle #0, as well as gaps,
    central angle, and orientation angles.
        
    '''
    def __init__(self, n = None, r = None, xc = None, yc = None, zc = None,
                 gap_01 = 0., gap_02 = 0., theta = 180., euler_alpha = 0.,
                 euler_beta = 0., euler_gamma = 0.):

        try:
            n = array(n).reshape(3)
            r = array(r).reshape(3)
        except ValueError:
            raise ScattererDefinitionError('n and r must be 3-element array-like objects', self)
        self.gap_01 = gap_01
        self.gap_02 = gap_02
        self.theta = theta
        self.euler_alpha = euler_alpha
        self.euler_beta = euler_beta
        self.euler_gamma = euler_gamma

        # define unrotated reference configuration here
        p0_coords = zeros(3)
        p1_coords = array([r[0] + r[1] + gap_01, 0., 0.])
        theta_rad = theta * pi/180.
        p2_coords = array([cos(theta_rad), sin(theta_rad), 0.]) * (r[0] +
                                                                   r[2] + 
                                                                   gap_02) 
           
        # compute rotation matrix and transform particle coords
        rot_mat = rotation_matrix(self.euler_alpha, self.euler_beta, 
                                  self.euler_gamma, False)
        # add displacement of particle 0
        p0_shift = array([xc, yc, zc])
        p0_rot = dot(rot_mat, p0_coords) + p0_shift
        p1_rot = dot(rot_mat, p1_coords) + p0_shift
        p2_rot = dot(rot_mat, p2_coords) + p0_shift
        centers = np.array([p0_rot, p1_rot, p2_rot])

        # make SphereCluster
        self.scatterers = []
        for i in range(3):
            s = Sphere(n = n[i], r = r[i], center = centers[i])
            self.scatterers.append(s)

    @property
    def parameter_names_list(self):
        return ['sphere_0.n.real', 'sphere_0.n.imag', 'sphere_0.r', 
                'sphere_0.x', 'sphere_0.y', 'sphere_0.z', 'sphere_1.n.real',
                'sphere_1.n.imag', 'sphere_1.r', 'sphere_2.n.real', 
                'sphere_2.n.imag', 'sphere_2.r', 'gap_01', 'gap_02',
                'theta', 'euler_alpha', 'euler_beta', 'euler_gamma']

    @property
    def parameter_list(self):
        spheres = self.get_component_list()
        parlist = spheres[0].parameter_list
        for sphere in spheres[1:]:
            parlist = np.append(parlist, sphere.parameter_list[0:3])
        other_params = array([self.gap_01, self.gap_02, self.theta,
                              self.euler_alpha, self.euler_beta,
                              self.euler_gamma])
        parlist = np.append(parlist, other_params)
        return parlist

    @classmethod
    def make_from_parameter_list(cls, params):
        n = array([params[0], params[6], params[9]]) + 1j * array([params[1],
                                                                   params[7],
                                                                   params[10]])
        r = array([params[2], params[8], params[11]])
        arg_arr= np.concatenate((params[3:6], params[12:]))
        return cls(n, r, *arg_arr)
        
