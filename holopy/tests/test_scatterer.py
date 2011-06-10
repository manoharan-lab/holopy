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
Test construction and manipulation of Scatterer objects.

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
'''

import numpy as np
import holopy
import nose
from nose.tools import raises, assert_raises
from numpy.testing import assert_, assert_equal, assert_array_almost_equal
import os
import string
from nose.plugins.attrib import attr

from holopy.model.scatterer import Sphere, CoatedSphere
from holopy.model.scatterer import Composite, SphereCluster
from holopy.utility.errors import SphereClusterDefError
#from holopy.model.scatterer import SphereDimer

def test_Sphere_construction():
    s = Sphere(n = 1.59, r = 5e-7, x = 1e-6, y = -1e-6, z = 10e-6)
    s = Sphere(n = 1.59, r = 5e-7)
    # index can be complex
    s = Sphere(n = 1.59+0.0001j, r = 5e-7)
    s = Sphere()

def test_Sphere_construct_list():
    # specify center as list
    center = [1e-6, -1e-6, 10e-6]
    s = Sphere(n = 1.59+0.0001j, r = 5e-7, center = center)
    assert_equal(s.center, np.array(center))

def test_Sphere_construct_tuple():
    # specify center as list
    center = (1e-6, -1e-6, 10e-6)
    s = Sphere(n = 1.59+0.0001j, r = 5e-7, center = center)
    assert_equal(s.center, np.array(center))

def test_Sphere_construct_array():
    # specify center as list
    center = np.array([1e-6, -1e-6, 10e-6])
    s = Sphere(n = 1.59+0.0001j, r = 5e-7, center = center)
    assert_equal(s.center, center)
    
def test_CoatedSphere_construction():
    cs = CoatedSphere(n1=1.59, n2=1.59, r1=5e-7, r2=1e-6, x=1e-6, 
                      y=-1e-6, z=10e-6) 
    cs = CoatedSphere(n1=1.59, n2=1.33, r1=5e-7, r2=1e-6)
    # index can be complex
    cs = CoatedSphere(n1 = 1.59+0.0001j, n2=1.33+0.0001j, r1=5e-7,
                      r2=1e-6) 
    center = np.array([1e-6, -1e-6, 10e-6])
    cs = CoatedSphere(n1 = 1.59+0.0001j, n2=1.33+0.0001j, r1=5e-7, 
                      r2=1e-6,
                      center = center) 
    cs = CoatedSphere()

def test_Composite_construction():
    # empty composite
    comp_empty = Composite()
    print comp_empty.get_component_list()
    
    # composite of multiple spheres
    s1 = Sphere(n = 1.59, r = 5e-7, x = 1e-6, y = -1e-6, z = 10e-6)
    s2 = Sphere(n = 1.59, r = 1e-6, center=[0,0,0])
    s3 = Sphere(n = 1.59+0.0001j, r = 5e-7, center=[5e-6,0,0])
    comp_spheres = Composite(scatterers=[s1, s2, s3])

    # heterogeneous composite
    cs = CoatedSphere(n1 = 1.59+0.0001j, n2=1.33+0.0001j, r1=5e-7, 
                      r2=1e-6,
                      center=[-5e-6, 0,0])
    comp = Composite(scatterers=[s1, s2, s3, cs])

    # multi-level composite (contains another composite)
    s4 = Sphere(center=[0, 5e-6, 0])
    comp_spheres.add(s4)
    comp2 = Composite(scatterers=[comp_spheres, comp])
    print comp2.get_component_list()

    # even more levels
    comp3 = Composite(scatterers=[comp2, cs])
    print comp3

def test_SphereCluster_construction():
    # empty cluster
    sc_empty = SphereCluster()
    print sc_empty.get_component_list()
    
    # cluster of multiple spheres
    s1 = Sphere(n = 1.59, r = 5e-7, x = 1e-6, y = -1e-6, z = 10e-6)
    s2 = Sphere(n = 1.59, r = 1e-6, center=[0,0,0])
    s3 = Sphere(n = 1.59+0.0001j, r = 5e-7, center=[5e-6,0,0])
    sc = SphereCluster(spheres=[s1, s2, s3])
    print sc.get_component_list()
    print sc

    # construct from an array
        
def test_SphereCluster__contains_only_spheres():
    s1 = Sphere(n = 1.59, r = 5e-7, x = 1e-6, y = -1e-6, z = 10e-6)
    s2 = Sphere(n = 1.59, r = 1e-6, center=[0,0,0])
    s3 = Sphere(n = 1.59+0.0001j, r = 5e-7, center=[5e-6,0,0])
    sc = SphereCluster(spheres=[s1, s2, s3])
    assert_(sc._contains_only_spheres() is True)

    sc.add(CoatedSphere())
    assert_(sc._contains_only_spheres() is False)

@raises(SphereClusterDefError)
def test_SphereCluster_construction_typechecking():
    # heterogeneous composite should raise exception, since a
    # sphere cluster must contain only Spheres
    s1 = Sphere(n = 1.59, r = 5e-7, x = 1e-6, y = -1e-6, z = 10e-6)
    s2 = Sphere(n = 1.59, r = 1e-6, center=[0,0,0])
    s3 = Sphere(n = 1.59+0.0001j, r = 5e-7, center=[5e-6,0,0])
    cs = CoatedSphere(n1 = 1.59+0.0001j, n2=1.33+0.0001j, r1=5e-7, 
                      r2=1e-6,
                      center=[-5e-6, 0,0])
    sc = SphereCluster(spheres=[s1, s2, s3, cs])
