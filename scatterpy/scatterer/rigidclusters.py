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
from numpy import array, sin, cos, dot, zeros
from holopy.process.math import rotation_matrix
from spherecluster import SphereCluster
from scatterpy.errors import ScattererDefinitionError

class BentTrimer(SphereCluster):
    '''
    Defines a bent three-particle cluster.  Inherits from SphereCluster.
    
    The reference geometry is that in an unrotated cluster (euler_alpha = 
    euler_beta = euler_gamma = 0), particle #2 lies a distance 
    +(r_1 + r_2 + gap_12) xhat from the central particle (#1).  Particle #3 
    lies in the x-y plane; the angle defined by particles 3, 1, and 2 is theta.
    Euler angles follow the SCSMFO zyz convention. 

    Attributes added to SphereCluster
    ---------------------------------
    gap_12 : float
        gap distance between central sphere (#1) and #2
    gap_13 : float
        gap distance between spheres #1 and #3
    theta : float
        angle subtended by spheres 2, 1 and 3
    euler_alpha : float
        Euler angle for rotation about z axis, degrees
    euler_beta : float
        Euler angle for rotation about y axis, degrees
    euler_gamma : float
        Euler angle for rotation about z axis, degrees

    Notes
    -----
    Initialize by specifying array-like n (3 elements), array-like r, 
    and float positions of the central particle #1, as well as gaps,
    central angle, and orientation angles.
        
    '''
    def __init__(self, n = None, r = None, xc = None, yc = None, zc = None,
                 gap_12 = 0., gap_13 = 0., theta = 180., euler_alpha = 0.,
                 euler_beta = 0., euler_gamma = 0.):

        try:
            n = array(n).reshape(3)
            r = array(r).reshape(3)
        except ValueError:
            raise ScattererDefinitionError("n and r must be 3-element array-like
                                           objects", self)
        self.gap_12 = gap_12
        self.gap_13 = gap_13
        self.theta = theta
        self.euler_alpha = euler_alpha
        self.euler_beta = euler_beta
        self.euler_gamma = euler_gamma

        # define unrotated reference configuration here
        p1_coords = zeros(3)
        p2_coords = array([r[0] + r[1] + gap_12, 0., 0.])
        p3_coords = array([cos(theta), sin(theta), 0.]) * (r[0]+r[2]+gap_13) 
           
        # compute rotation matrix and transform particle coords
        rot_mat = rotation_matrix(euler_alpha, euler_beta, euler_gamma, False)
        # add displacement of particle 1
        p1_shift = array([xc, yc, zc])
        p1_rot = dot(rot_mat, p1_coords) + p1_shift
        p2_rot = dot(rot_mat, p2_coords) + p1_shift
        p3_rot = dot(rot_mat, p3_coords) + p1_shift
        centers = np.array([p1_rot, p2_rot, p3_rot])

        # make SphereCluster
        self.scatterers = []
        for i in range(3):
            s = Sphere(n = n[i], r = r[i], center = centers[i])
            self.scatterers.append(s)

    # TODO:
    # Override parameter_list, parameter_names_list, make_from_parameter_list
