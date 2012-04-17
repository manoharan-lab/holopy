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
.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
'''

from collections import OrderedDict

import numpy as np
from nose.tools import raises, assert_raises
from numpy.testing import assert_, assert_equal, assert_almost_equal
from nose.plugins.attrib import attr

from scatterpy.scatterer import Sphere, CoatedSphere
from scatterpy.scatterer import SphereCluster
from scatterpy.errors import ScattererDefinitionError, InvalidScattererSphereOverlap


@attr('fast')
def test_SphereCluster_construction():
    
    # cluster of multiple spheres
    s1 = Sphere(n = 1.59, r = 5e-7, center = (1e-6, -1e-6, 10e-6))
    s2 = Sphere(n = 1.59, r = 1e-6, center=[0,0,0])
    s3 = Sphere(n = 1.59+0.0001j, r = 5e-7, center=[5e-6,0,0])
    sc = SphereCluster(spheres=[s1, s2, s3])
    print sc.get_component_list()
    print sc

    # test attribute access
    sc.n
    sc.n_real
    sc.n_imag
    sc.r
    sc.x
    sc.y
    sc.z
    sc.centers
    sc.center


@attr('fast')
@raises(ScattererDefinitionError)
def test_SphereCluster_construction_typechecking():
    # heterogeneous composite should raise exception, since a
    # sphere cluster must contain only Spheres
    s1 = Sphere(n = 1.59, r = 5e-7, center = (1e-6, -1e-6, 10e-6))
    s2 = Sphere(n = 1.59, r = 1e-6, center=[0,0,0])
    s3 = Sphere(n = 1.59+0.0001j, r = 5e-7, center=[5e-6,0,0])
    cs = CoatedSphere(n=(1.59+0.0001j, 1.33+0.0001j), r=(5e-7, 1e-6),
                      center=[-5e-6, 0,0])
    sc = SphereCluster(spheres=[s1, s2, s3, cs])

@attr('fast')
def test_SphereCluster_ovelap_checking():
    s1 = Sphere(n = 1.59, r = 5e-7, center=(1e-6, -1e-6, 10e-6))
    sc = SphereCluster([s1, s1, s1])

    with assert_raises(InvalidScattererSphereOverlap) as cm:
        sc.validate()
    assert_equal(str(cm.exception),
        'SphereCluster(spheres=[Sphere(center=(1e-06, -1e-06, 1e-05), '
        'n=(1.59+0j), r=5e-07), Sphere(center=(1e-06, -1e-06, 1e-05), '
        'n=(1.59+0j), r=5e-07), Sphere(center=(1e-06, -1e-06, 1e-05), '
        'n=(1.59+0j), r=5e-07)]) has overlaps between spheres: [(0, 1), (0, 2), '
        '(1, 2)]')


def test_SphereCluster_parameters():
    s1 = Sphere(n = 1.59, r = 5e-7, center=[1e-6, -1e-6, 10e-6])
    s2 = Sphere(n = 1.59, r = 1e-6, center=[0,0,0])
    sc = SphereCluster(spheres = [s1, s2])

    assert_equal(sc.parameters, OrderedDict([('0:Sphere.center[0]',
    9.9999999999999995e-07), ('0:Sphere.center[1]', -9.9999999999999995e-07),
    ('0:Sphere.center[2]', 1.0000000000000001e-05), ('0:Sphere.n.imag', 0.0),
    ('0:Sphere.n.real', 1.59), ('0:Sphere.r', 5e-07), ('1:Sphere.center[0]', 0),
    ('1:Sphere.center[1]', 0), ('1:Sphere.center[2]', 0), ('1:Sphere.n.imag',
    0.0), ('1:Sphere.n.real', 1.59), ('1:Sphere.r', 1e-06)]))

    sc2 = SphereCluster.from_parameters(sc.parameters)

    assert_equal(sc.scatterers[0].r, sc2.scatterers[0].r)
    assert_equal(sc.scatterers[1].r, sc2.scatterers[1].r)
    assert_equal(sc.scatterers[0].n, sc2.scatterers[0].n)
    assert_equal(sc.scatterers[1].n, sc2.scatterers[1].n)
    assert_equal(sc.scatterers[0].center, sc2.scatterers[0].center)
    assert_equal(sc.scatterers[1].center, sc2.scatterers[1].center)
    
    
def test_SphereCluster_translation():
    s1 = Sphere(n = 1.59, r = 5, center=[1, -1, 10])
    s2 = Sphere(n = 1.59, r = 1, center=[0,0,0])
    sc = SphereCluster(spheres = [s1, s2])

    sc2 = sc.translated(1, 1, 1)

    assert_equal(sc.scatterers[0].r, sc2.scatterers[0].r)
    assert_equal(sc.scatterers[1].r, sc2.scatterers[1].r)
    assert_equal(sc.scatterers[0].n, sc2.scatterers[0].n)
    assert_equal(sc.scatterers[1].n, sc2.scatterers[1].n)
    assert_equal([2, 0, 11], sc2.scatterers[0].center)
    assert_equal([1, 1, 1], sc2.scatterers[1].center)

def test_SphereCluster_rotation():
    s1 = Sphere(n = 1.59, r = 1, center = [1, 0, 0])
    s2 = Sphere(n = 1.59, r = 1, center = [-1, 0, 1])
    sc = SphereCluster(spheres = [s1, s2])

    sc2 = sc.rotated(np.pi/2, 0, 0)

    assert_equal(sc.scatterers[0].r, sc2.scatterers[0].r)
    assert_equal(sc.scatterers[1].r, sc2.scatterers[1].r)
    assert_equal(sc.scatterers[0].n, sc2.scatterers[0].n)
    assert_equal(sc.scatterers[1].n, sc2.scatterers[1].n)
    assert_almost_equal([0, -1, 0], sc2.scatterers[0].center)
    assert_almost_equal([0, 1, 1], sc2.scatterers[1].center)
    

