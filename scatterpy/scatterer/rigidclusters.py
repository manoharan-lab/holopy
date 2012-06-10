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
from __future__ import division

#COVERAGE: This file is on its way out in refactoring

# VNM: convert all of these to models rather than scatterers

import numpy as np
from numpy import array, sin, cos, dot, zeros, pi, ones, sqrt, arccos
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


class RigidCluster(SphereCluster):
    '''
    Base class for regular rigid clusters where we will specify rotation about
    the geometric center rather than a specific particle.

    Geometrical regularity requires r to be identical for every particle, and
    requires the gap to be identical between every pair of adjacent particles.

    Subclasses (specific cluster geometries) need to call 
    RigidCluster.__init__() in their constructor. They also need to provide
    a @property reference_geometry. This should be a (n x 3) ndarray,
    where the nth row contains the x, y, and z coordinates of the nth particle
    in units of (r + gap/2).

    Euler angles follow the SCSMFO zyz convention.

    Attributes added to SphereCluster
    ---------------------------------
    n_spheres : int
        Number of spheres in cluster
    x_com : float
    y_com : float
    z_com : float
        x, y, z coords of cluster center of mass
    euler_alpha : float
        Euler angle for rotation about z axis, degrees
    euler_beta : float
        Euler angle for rotation about y axis, degrees
    euler_gamma : float
        Euler angle for rotation about z axis, degrees
    gap : float
        Gap distance between particle edges.
    reference_geometry : ndarray (n x 3)
        Reference configuration of cluster.

    '''
    def __init__(self, n_spheres = None, n = None, r = None, x_com = None, 
                 y_com = None, z_com = None, euler_alpha = None, 
                 euler_beta = None, euler_gamma = None, gap = None):
        '''
        Parameters:
        n_spheres : int
        n : float or ndarray(n_spheres)
        r : float
        '''
        # initialize 
        self.n_spheres = n_spheres
        n = array(n)
        if n.size == 1: # scalar
            n = n * ones(self.n_spheres)
        else:
            try:
                n = n.reshape(self.n_spheres)
            except ValueError:
                raise ScattererDefinitionError('Wrong number of n specified', 
                                               self)
        #print 'Made it here'
        self.x_com = x_com
        self.y_com = y_com
        self.z_com = z_com
        self.euler_alpha = euler_alpha
        self.euler_beta = euler_beta
        self.euler_gamma = euler_gamma
        self.gap = gap

        # compute reference geometry
        ref_positions = self.reference_geometry * (r + self.gap/2.)
        # compute rotation matrix and transform particle coords
        rot_mat = rotation_matrix(self.euler_alpha, self.euler_beta, 
                                  self.euler_gamma, False)
        # add displacement of com
        com_shift = array([self.x_com, self.y_com, self.z_com])
        centers = np.array([dot(rot_mat, pcoords) + com_shift for pcoords in
                            ref_positions])

        # make SphereCluster
        self.scatterers = []
        for i in range(self.n_spheres):
            s = Sphere(n = n[i], r = r, center = centers[i])
            self.scatterers.append(s)

    @property
    def reference_geometry(self):
        raise NotImplementedError

    @property
    def parameter_names_list(self):
        '''
        [sphere_i.nreal, sphere_i.nimag, r, x_com, y_com, z_com, ea, eb, eg,
        gap]
        '''
        parnames = []
        # add n for each sphere
        for i in xrange(self.n_spheres):
            parnames.append('sphere_' + str(i) + '.n.real')
            parnames.append('sphere_' + str(i) + '.n.imag')
        parnames.extend(['r', 'x_com', 'y_com', 'z_com', 'euler_alpha', 
                         'euler_beta', 'euler_gamma', 'gap'])
        return parnames

    @property
    def parameter_list(self):
        parlist = array([])
        spheres = self.get_component_list()
        for sphere in spheres: # indices
            parlist = np.append(parlist, sphere.parameter_list[0:2])
        parlist = np.append(parlist, array([self.r[0], self.x_com, self.y_com,
                                            self.z_com, self.euler_alpha, 
                                            self.euler_beta, self.euler_gamma, 
                                            self.gap]))
        return parlist


class Tetrahedron(RigidCluster):
    def __init__(self, n = None, r = None, x_com = None, 
                 y_com = None, z_com = None, euler_alpha = None, 
                 euler_beta = None, euler_gamma = None, gap = None):
        RigidCluster.__init__(self, n_spheres = 4, n = n,
                              r = r, x_com = x_com, y_com = y_com,
                              z_com = z_com, euler_alpha = euler_alpha,
                              euler_beta = euler_beta, 
                              euler_gamma = euler_gamma, gap = gap)
    
    @property
    def reference_geometry(self):
        '''
        This is different from Becca's definition (and from JF's prior 
        reference configuration) because it makes the symmetry more manifest.
        Projected onto the x-z plane, the particle projections lie on the 
        x and z axes.
        '''
        sphere_0 = array([-1., 0., sqrt(2.)/2.])
        sphere_1 = array([1., 0., sqrt(2.)/2.])
        sphere_2 = array([0., 1., -sqrt(2.)/2.])
        sphere_3 = array([0., -1., -sqrt(2.)/2.])
        return array([sphere_0, sphere_1, sphere_2, sphere_3]) 
    

class TrigonalBipyramid(RigidCluster):
    def __init__(self, n = None, r = None, x_com = None, 
                 y_com = None, z_com = None, euler_alpha = None, 
                 euler_beta = None, euler_gamma = None, gap = None):
        RigidCluster.__init__(self, n_spheres = 5, n = n,
                              r = r, x_com = x_com, y_com = y_com,
                              z_com = z_com, euler_alpha = euler_alpha,
                              euler_beta = euler_beta, 
                              euler_gamma = euler_gamma, gap = gap)

    @property
    def reference_geometry(self):
        '''
        Particles 1, 2, 3 lie in the y-z plane; particles 0 and 4 on the x-axis.
        '''
        sphere_0 = array([-2*sqrt(2./3.), 0., 0.])
        sphere_1 = array([0., 0., 2.*sqrt(3.)/3.])
        sphere_2 = array([0., 1., -sqrt(3.)/3.])
        sphere_3 = array([0., -1., -sqrt(3.)/3.])
        sphere_4 = array([2*sqrt(2./3.), 0., 0.])
        return array([sphere_0, sphere_1, sphere_2, sphere_3, sphere_4])


class Polytetrahedron(RigidCluster):
    def __init__(self, n = None, r = None, x_com = None, 
                 y_com = None, z_com = None, euler_alpha = None, 
                 euler_beta = None, euler_gamma = None, gap = None):
        RigidCluster.__init__(self, n_spheres = 6, n = n,
                              r = r, x_com = x_com, y_com = y_com,
                              z_com = z_com, euler_alpha = euler_alpha,
                              euler_beta = euler_beta, 
                              euler_gamma = euler_gamma, gap = gap)

    @property
    def reference_geometry(self):
        '''
        Particles 0-3 have the same relative positions as in the tetrahedron.
        Particles 4 and 5 have y = 0.  We have particles 0, 1, 4, and 5 with
        y = 0 (we project these onto the x-z plane to calculate the geometry),
        and particles 2 and 3 have x = 0.
        '''
        # Because the geometry is messy, don't get z-COM yet.
        x = (3.*arccos(1./3.) - pi)/2.
        sphere_0 = array([-1., 0, sqrt(2.)])
        sphere_1 = array([1., 0, sqrt(2.)])
        sphere_2 = array([0, 1., 0])
        sphere_3 = array([0, -1., 0])
        sphere_4 = array([-cos(x), 0, -sin(x)]) * sqrt(3.)
        sphere_5 = array([cos(x), 0, -sin(x)]) * sqrt(3.)
        spheres = array([sphere_0, sphere_1, sphere_2, sphere_3, sphere_4,
                         sphere_5])
        # calculate the offset needed such that the z-centroid = 0
        z_off = spheres[:,2].sum()/6. 
        return spheres - array([0, 0, z_off])
