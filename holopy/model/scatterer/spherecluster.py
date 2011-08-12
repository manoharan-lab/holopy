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
from holopy.model.errors import ScattererDefinitionError

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

            
    # convenience functions, defined so you can write, e.g., sc.n
    # instead of sc.get_n()
    @property
    def n(self):
        return [s.n for s in self.scatterers]
    @property
    def n_real(self):
        return [s.n.real for s in self.scatterers]
    @property
    def n_imag(self):
        return [s.n.imag for s in self.scatterers]
    @property
    def r(self):
        return [s.r for s in self.scatterers]
    @property
    def x(self):
        return [s.x for s in self.scatterers]
    @property
    def y(self):
        return [s.y for s in self.scatterers]
    @property
    def z(self):
        return [s.z for s in self.scatterers]
    @property
    def centers(self):
        return [s.center for s in self.scatterers]

