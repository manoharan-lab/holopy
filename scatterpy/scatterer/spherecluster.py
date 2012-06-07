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
Defines SphereCluster, a Composite scatterer consisting of Spheres

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
'''
# COVERAGE: I think all uncovered code is either unreachable or due likely to be
# refactored away

import numpy as np
from sphere import Sphere
from composite import Composite
from scatterpy.errors import (ScattererDefinitionError,
                              InvalidScattererSphereOverlap, warning)
from holopy.process.math import cartesian_distance, rotate_points

class SphereCluster(Composite):
    '''
    Contains optical and geometrical properties of a cluster of spheres. 

    Attributes
    ----------
    spheres : list of Spheres
        Spheres which will make up the cluster

    Notes
    -----
    '''

    def __init__(self, spheres):
        # make sure all components are spheres
        for s in spheres:
            if not isinstance(s, Sphere):
                raise ScattererDefinitionError(
                    "SphereCluster expects all component " +
                    "scatterers to be Spheres.\n" + 
                    repr(s) + " is not a Sphere", self)
        self.scatterers = spheres

        if self.has_overlaps:
            warning("creating unphysical scatterer with overlapping spheres", self)

    @property
    def has_overlaps(self):
        overlaps = []
        for i, s1 in enumerate(self.scatterers):
            for j in range(i+1, len(self.scatterers)):
                s2= self.scatterers[j]
                if cartesian_distance(s1.center, s2.center) < (np.max(s1.r) + np.max(s2.r)):
                    overlaps.append((i, j))
        return len(overlaps) > 0
            
    def __repr__(self):
        return "{c}(spheres={spheres})".format(c=self.__class__.__name__,
                                       spheres=repr(self.scatterers))


    @property
    def n(self):
        return np.array([s.n for s in self.scatterers])
    @property
    def n_real(self):
        return np.array([s.n.real for s in self.scatterers])
    @property
    def n_imag(self):
        return np.array([s.n.imag for s in self.scatterers])
    @property
    def r(self):
        return np.array([s.r for s in self.scatterers])
    @property
    def x(self):
        return np.array([s.x for s in self.scatterers])
    @property
    def y(self):
        return np.array([s.y for s in self.scatterers])
    @property
    def z(self):
        return np.array([s.z for s in self.scatterers])
    @property
    def centers(self):
        return np.array([s.center for s in self.scatterers])

    @property
    def center(self):
        return self.centers.mean(0)

# TODO: Move this code out of scatterer? It sort of has more to do with how
# clusters move than pure geometry
    
def rotate(cluster, theta, phi, psi):
    com = cluster.centers.mean(0)
        
    return SphereCluster([Sphere(n=s.n, r=s.r, center =
                                 com+rotate_points(s.center-com, theta,
                                                   phi, psi)) for s in
                          cluster.scatterers])


class RotatedSphereCluster(SphereCluster):
    def __init__(self, orig_cluster, alpha, beta, gamma, com = None):
        self.com = orig_cluster.centers.mean(0)
        self.orig_cluster = SphereCluster([Sphere(n=s.n, r=s.r, center =
                                                  s.center-self.com) for s in
                                          orig_cluster.scatterers])
        # overwrite whatever com the particle originally had
        if com is not None:
            self.com = com
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __repr__(self):
        return "{s.__class__.__name__}(theta={s.theta}, phi={s.phi}, \
psi={s.psi}, com={s.com}, orig_cluster={o})".format(s=self, o=repr(self.orig_cluster))
        
    @property
    def scatterers(self):
        return [Sphere(n=s.n, r=s.r,
                       center = self.com + rotate_points(s.center, self.alpha,
                                                         self.beta, self.gamma))
                for s in self.orig_cluster.scatterers]

    @property
    def parameter_names_list(self):
        return ['com_x', 'com_y', 'com_z', 'alpha', 'beta', 'gamma']

    @property
    def parameter_list(self):
        return np.array([self.com[0], self.com[1], self.com[2], self.alpha, 
                         self.beta, self.gamma])

    # not a classmethod because the parameter list does not have enough
    # information to make a new one, need to reference an existing
    # RotatedSphereCluster to get a value for orig_cluster
    def make_from_parameter_list(self, params):
        return RotatedSphereCluster(self.orig_cluster, params[3], params[4],
                                    params[5], (params[0], params[1], params[2]))
