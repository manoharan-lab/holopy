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
from nose.tools import raises, assert_raises
from numpy.testing import assert_, assert_equal, assert_almost_equal
from nose.plugins.attrib import attr

from scatterpy.scatterer import Sphere, CoatedSphere, Scatterer
from scatterpy.scatterer import Composite, SphereCluster
from scatterpy.errors import ScattererDefinitionError, InvalidScattererSphereOverlap
from common import ErrorExpected

@attr('fast')
def test_Scatterer_construction():
    assert_raises(NotImplementedError, lambda:  Scatterer())

@attr('fast')
def test_Sphere_construction():
    s = Sphere(n = 1.59, r = 5e-7, x = 1e-6, y = -1e-6, z = 10e-6)
    s = Sphere(n = 1.59, r = 5e-7)
    # index can be complex
    s = Sphere(n = 1.59+0.0001j, r = 5e-7)
    s = Sphere()

@attr('fast')
def test_Sphere_construct_list():
    # specify center as list
    center = [1e-6, -1e-6, 10e-6]
    s = Sphere(n = 1.59+0.0001j, r = 5e-7, center = center)
    assert_equal(s.center, np.array(center))

@attr('fast')
def test_Sphere_construct_tuple():
    # specify center as list
    center = (1e-6, -1e-6, 10e-6)
    s = Sphere(n = 1.59+0.0001j, r = 5e-7, center = center)
    assert_equal(s.center, np.array(center))

@attr('fast')
def test_Sphere_construct_array():
    # specify center as list
    center = np.array([1e-6, -1e-6, 10e-6])
    s = Sphere(n = 1.59+0.0001j, r = 5e-7, center = center)
    assert_equal(s.center, center)

    try:
        Sphere(center = 1)
        raise ErrorExpected('Sphere with only 1 center coordinate is not valid')
    except ScattererDefinitionError as e:
            assert_(str(e), "Error defining scatterer object of type Sphere. center specified as 1, center should be specified as (x, y, z)")

@attr('fast')
def test_Sphere_construct_params():
    params = np.array([1.59, 1e-4, 5e-7, 1e-6, -1e-6, 10e-6])
    s = Sphere.make_from_parameter_list(params)
    assert_equal(s.parameter_list, params)
    
@attr('fast')
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

@attr('fast')
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

@attr('fast')
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

    # construct from lists
    n = [1.59, 1.58, 1.57, 1.56]
    r = [0.5e-6, 0.49e-6, 0.48e-6, 0.47e-6]
    x = [0.0, 1.0e-6, 2.0e-6, 3.0e-6]
    y = [0.0, -1.0e-6, -2.0e-6, -3.0e-6]
    z = [10.0e-6, 11.0e-6, 12.0e-6, 13.0e-6]
    sc = SphereCluster(n=n, r=r, x=x, y=y, z=z)
    assert_((sc.n == n).all() and (sc.r == r).all() and (sc.x == x).all() and 
            (sc.y == y).all() and (sc.z == z).all())

    # construct from arrays
    na = np.array(n)
    ra = np.array(r)
    xa = np.array(x)
    ya = np.array(y)
    za = np.array(z)
    sc = SphereCluster(n=na, r=ra, x=xa, y=ya, z=za)
    assert_((sc.n == n).all() and (sc.r == r).all() and (sc.x == x).all() and 
            (sc.y == y).all() and (sc.z == z).all())

    # __init__ should throw an exception if arrays are wrong sizes
    assert_raises(ScattererDefinitionError, lambda:
                      SphereCluster(n=n, r=r, x=x, y=y, z=0))
    assert_raises(ScattererDefinitionError, lambda: 
                  SphereCluster(n=n, r=r, centers=[0,0,0]))
    
    # TODO: fix this test, it fails overlap checking and so will not run
    # should be okay if all arrays are the same size
    sc = SphereCluster(n=n, r=r, centers=np.ones((4,3)))
    assert_((sc.n == n).all() and (sc.r == r).all())
    assert_equal(sc.centers[0], np.ones(3))
    # but throw error if they're different
    assert_raises(ScattererDefinitionError, lambda: 
                  SphereCluster(n=n, r=r, centers=np.ones((3,3))))
    assert_raises(ScattererDefinitionError, lambda: 
                  SphereCluster(n=n, r=r, centers=np.ones((5,3))))

    # test for single sphere only
    sc = SphereCluster(n=1.59, r=1e-6, 
                       centers=np.array([x[0], y[0], z[0]]))
    assert_almost_equal(sc.n, 1.59)
    assert_almost_equal(sc.r, 1e-6)
    assert_equal(sc.centers[0], np.array([x[0], y[0], z[0]]))
    sc = SphereCluster(n=1.59, r=1e-6, 
                       centers=[x[0], y[0], z[0]])
    
    # now use an array to define the centers
    centers = np.array([[0,0,1], [0,0,2], [0,0,3], [0,0,4]])
    sc = SphereCluster(n=n, r=r, centers=centers)
    assert_equal(sc.centers[0], centers[0])
    print sc.get_component_list()

@attr('fast')
def test_SphereCluster_construct_params():
    params = [1.5891, 0.0001, 6.7e-07, 1.56e-05, 1.44e-05, 1.5e-05, 1.5891,
              0.0001, 6.5e-07, 3.42e-05, 3.17e-05, 1.0e-05]
    s = SphereCluster.make_from_parameter_list(params)
    assert_equal(s.parameter_list, params) 
    
@attr('fast')
def test_SphereCluster_contains_only_spheres():
    s1 = Sphere(n = 1.59, r = 5e-7, x = 1e-6, y = -1e-6, z = 10e-6)
    s2 = Sphere(n = 1.59, r = 1e-6, center=[0,0,0])
    s3 = Sphere(n = 1.59+0.0001j, r = 5e-7, center=[5e-6,0,0])
    sc = SphereCluster(spheres=[s1, s2, s3])
    assert_(sc.contains_only_spheres() is True)

    sc.add(CoatedSphere())
    assert_(sc.contains_only_spheres() is False)

    # a cluster with no spheres defined should return false
    sc = SphereCluster()
    assert_(sc.contains_only_spheres() is False)

@attr('fast')
@raises(ScattererDefinitionError)
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

@attr('fast')
def test_SphereCluster_ovelap_checking():
    s1 = Sphere(n = 1.59, r = 5e-7, x = 1e-6, y = -1e-6, z = 10e-6)
    sc = SphereCluster([s1, s1, s1])
    try:
       sc.validate()
       raise ErrorExpected('cluster with overlaping spheres should fail validation')
    except InvalidScattererSphereOverlap as e:
        assert_(str(e), "SphereCluster(spheres=[Sphere(center=[9.9999999999999995e-07, -9.9999999999999995e-07, 1.0000000000000001e-05], n=1.59, r=5e-07), Sphere(center=[9.9999999999999995e-07, -9.9999999999999995e-07, 1.0000000000000001e-05], n=1.59, r=5e-07), Sphere(center=[9.9999999999999995e-07, -9.9999999999999995e-07, 1.0000000000000001e-05], n=1.59, r=5e-07)]) has overlaps between spheres: [(0, 1), (0, 2), (1, 2)]")


@attr('fast')
def test_abstract_scatterer():

    class Dummy(Scatterer):
        def __init__(self):
            pass
    
    s = Dummy()

    assert_raises(NotImplementedError, lambda : s.parameter_list)
    assert_raises(NotImplementedError,
                  lambda : Scatterer.make_from_parameter_list([1, 2, 3]))
