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

import numpy as np
from sphere import Sphere
from composite import Composite
from scatterpy.errors import ScattererDefinitionError, InvalidScattererSphereOverlap
from holopy.process.math import cartesian_distance, rotate_points

class SphereCluster(Composite):
    '''
    Contains optical and geometrical properties of a cluster of spheres. 

    Attributes
    ----------
    spheres : list
        Interparticle gap distance ( = 0 at hard-sphere contact.) 
    n : array-like
        indices of refraction of each sphere
    x : array-like
        x-position of each sphere 
    y : array-like
        y-position of each sphere
    z : array-like
        z-position of each sphere
    centers : Nx3 array (optional)
        array of sphere positions

    Notes
    -----
    Inherited from Composite.  You can specify the list of spheres in
    two ways: by giving a list of Sphere objects, or by giving lists
    of `n`, `r`, `x`, `y`, and `z`.  You can also specify the
    positions of the spheres in `centers`, an Nx3 array.  If you
    specify a list of Sphere objects, the arguments `n`, `r`, `x`,
    `y`, `z` and `centers` are ignored.
    '''

    def __init__(self, spheres=None, n=None, r=None, x=None, y=None,
                 z=None, centers=None):
        if spheres is None:
            self.scatterers = []
            # find number of spheres from n
            N = np.size(np.array(n))
            # below is a bunch of stuff that checks for all sorts of
            # errors that can occur when initializing.  This is
            # because the constructor needs to process several
            # different ways of specifying the cluster
            if n is not None:
                try:
                    n = np.array(n).reshape(N)
                    r = np.array(r).reshape(N)
                except ValueError:
                    raise ScattererDefinitionError(
                        "r must be the same size as n", self)
                if centers is not None:
                    centers = np.array(centers)
                    try:
                        centers = centers.reshape((N, 3))
                    except ValueError:
                        raise ScattererDefinitionError(
                            "parameter 'centers' must have Nx3 elements", 
                            self)
                else:
                    try: 
                        x = np.array(x).reshape(N)
                        y = np.array(y).reshape(N)
                        z = np.array(z).reshape(N)
                        centers = np.array([x, y, z]).transpose()
                    except ValueError:
                        raise ScattererDefinitionError(
                            "n, r, x, y, and z should all be of length N",
                            self)
                for i in range(N):
                    try:
                        s = Sphere(n=n[i], r=r[i], center=centers[i])
                    except IndexError:
                        raise ScattererDefinitionError(
                            "n, r, x, y, and z should all be "+
                            "of length N.", self)
                    self.scatterers.append(s)
        else: 
            # make sure all components are spheres
            for s in spheres:
                if not isinstance(s, Sphere):
                    raise ScattererDefinitionError(
                        "SphereCluster expects all component " +
                        "scatterers to be Spheres.\n" + 
                        repr(s) + " is not a Sphere", self)
            self.scatterers = spheres

        self._validate()

    def _validate(self):
        overlaps = []
        for i, s1 in enumerate(self.scatterers):
            for j in range(i+1, len(self.scatterers)):
                s2= self.scatterers[j]
                if cartesian_distance(s1.center, s2.center) < (s1.r + s2.r):
                    overlaps.append((i, j))

        if overlaps:
            raise InvalidScattererSphereOverlap(self, overlaps)

        return True

    def __repr__(self):
        return "{c}(spheres={spheres})".format(c=self.__class__.__name__,
                                       spheres=repr(self.get_component_list()))

    @property
    def parameter_list(self):
        """
        Return sphere parameters in order: n, r, x, y, z
        """
        spheres = self.get_component_list()
        parlist = spheres[0].parameter_list
        for sphere in spheres[1:]:
            parlist = np.append(parlist, sphere.parameter_list)
        return parlist

    @property
    def parameter_names_list(self):
        spheres = self.get_component_list()
        names = []
        for i, sphere in enumerate(spheres):
            names.extend(['sphere_{0}.{1}'.format(i, name) for name in
                            sphere.parameter_names_list])
        return names

    @classmethod
    def make_from_parameter_list(cls, params):
        sphere_params = 6
        num_spheres = len(params)/sphere_params
        s = []
        for i in range(num_spheres):
            s.append(Sphere.make_from_parameter_list(
                    params[i*sphere_params:(i+1)*sphere_params]))
        sc = cls(s)
        sc._validate()
        return sc
    
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

def rotate(cluster, theta, phi, psi):
    com = cluster.centers.mean(0)
        
    return SphereCluster([Sphere(n=s.n, r=s.r, center =
                                 com+rotate_points(s.center-com, theta,
                                                   phi, psi)) for s in
                          cluster.scatterers])


class RotatedSphereCluster(SphereCluster):
    def __init__(self, orig_cluster, theta, phi, psi, com = None):
        self.com = orig_cluster.centers.mean(0)
        self.orig_cluster = SphereCluster([Sphere(n=s.n, r=s.r, center =
                                                  s.center-self.com) for s in
                                          orig_cluster.scatterers])
        # overwrite whatever com the particle originally had
        if com is not None:
            self.com = com
        self.theta = theta
        self.phi = phi
        self.psi = psi


    @property
    def scatterers(self):
        return [Sphere(n=s.n, r=s.r,
                       center = self.com + rotate_points(s.center, self.theta,
                                                         self.phi, self.psi))
                for s in self.orig_cluster.scatterers]

    @property
    def parameter_names_list(self):
        return ['com_x', 'com_y', 'com_z', 'theta', 'phi', 'psi']

    @property
    def parameter_list(self):
        return np.array([self.com[0], self.com[1], self.com[2], self.theta, self.phi, self.psi])

    # not a classmethod because the parameter list does not have enough
    # information to make a new one, need to reference an existing
    # RotatedSphereCluster to get a value for orig_cluster
    def make_from_parameter_list(self, params):
        return RotatedSphereCluster(self.orig_cluster, params[1], params[2],
                                    params[3], params[0])
