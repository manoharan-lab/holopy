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
from numpy.testing import assert_equal
from nose.plugins.attrib import attr

from scatterpy.scatterer import (Sphere, CoatedSphere, Scatterer, Ellipsoid,
                                 Composite, MovingSphere, abstract_scatterer)

from scatterpy.errors import ScattererDefinitionError

@attr('fast')
def test_Scatterer_construction():
    assert_raises(NotImplementedError, Scatterer)

@attr('fast')
def test_Sphere_construction():
    s = Sphere(n = 1.59, r = 5e-7, x = 1e-6, y = -1e-6, z = 10e-6)
    s = Sphere(n = 1.59, r = 5e-7)
    # index can be complex
    s = Sphere(n = 1.59+0.0001j, r = 5e-7)
    s = Sphere()

    with assert_raises(ScattererDefinitionError) as cm:
        Sphere(n = 1.59, r = 5e-7, x = 1e-6, y = -1e-6, z = None)
    assert_equal(str(cm.exception), 'Error defining scatterer object of type '
                 'Sphere.\ncenter specified as x=1e-06, y=-1e-06, z=None, '
                 'center should be specified as (x, y, z)')

def test_Ellipsoid_construction():
    s = Ellipsoid(n = 1.57, r = (1, 2, 3), center = (3, 2, 1))

    assert_equal(str(s), 'Ellipsoid(center=[3, 2, 1], n=1.57, r=[1, 2, 3])')
    
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

    with assert_raises(ScattererDefinitionError) as cm:
        Sphere(center = 1)
    assert_equal(str(cm.exception), "Error defining scatterer object of type "
                 "Sphere.\ncenter specified as 1, center should be specified "
                 "as (x, y, z)")

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
def test_abstract_scatterer():

    class Dummy(Scatterer):
        def __init__(self):
            pass
    
    s = Dummy()

    assert_raises(NotImplementedError, lambda: s.parameter_list)
    assert_raises(NotImplementedError, Scatterer.make_from_parameter_list,
                  [1, 2, 3]) 


@attr('fast')
def test_MovingSphere():
    s = MovingSphere(center=(1e-6, -1e-6, 10e-6), v_x = 1.0, int_time=10e-6)
    # test an odd number of smear steps
    MovingSphere(center=(1e-6, -1e-6, 10e-6), v_x = 1.0, int_time=10e-6, n_smear=11)
    assert_equal(repr(s), 'MovingSphere(center=[9.9999999999999995e-07, '
                 '-9.9999999999999995e-07, 1.0000000000000001e-05], n=1.59, '
                 'r=5e-07)')

@attr('fast')
def test_xyzTriple():
    xyzTriple = abstract_scatterer.xyzTriple
    # Three ways to init an xyzTriple, should all give the same result
    c_1 = xyzTriple(1, 2, 3)
    c_2 = xyzTriple(x=1, y=2, z=3)
    c_3 = xyzTriple(xyz = (1, 2, 3))

    assert_equal(c_1, c_2)
    assert_equal(c_3, c_1)

    # Now confirm that it rejects bad input
    with assert_raises(abstract_scatterer.InvalidxyzTriple) as cm:
        xyzTriple(1)
    assert_equal(str(cm.exception), "Specification of xyzTriple(x=1, "
                 "y=None, z=None) is not valid, should be specified as "
                 "(x, y, z)")
    assert_raises(abstract_scatterer.InvalidxyzTriple, xyzTriple, xyz = 1)
    assert_raises(abstract_scatterer.InvalidxyzTriple, xyzTriple, xyz = [1])

    assert_equal(c_1.parameters, {'x': 1, 'y': 2, 'z' : 3})
    assert_equal(c_1.parameters_prefix('r'), {'r_x': 1, 'r_y': 2, 'r_z' : 3})
