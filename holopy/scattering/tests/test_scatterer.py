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
from __future__ import division

import numpy as np
from nose.tools import raises, assert_raises
from numpy.testing import assert_equal
from nose.plugins.attrib import attr

from ...core.helpers import OrderedDict

from ..scatterer import (Sphere, Scatterer, Ellipsoid,
                         Scatterers, scatterer)

from ..errors import ScattererDefinitionError
from .common import assert_allclose


@attr('fast')
def test_Sphere_construction():
    s = Sphere(n = 1.59, r = 5e-7, center = (1e-6, -1e-6, 10e-6))
    s = Sphere(n = 1.59, r = 5e-7)
    # index can be complex
    s = Sphere(n = 1.59+0.0001j, r = 5e-7)
    s = Sphere()

    with assert_raises(ScattererDefinitionError):
        Sphere(n=1.59, r = -2, center = (1, 1, 1))

    # now test multilayer spheres
    cs = Sphere(n=(1.59, 1.59), r=(5e-7, 1e-6), center=(1e-6, -1e-6, 10e-6))
    cs = Sphere(n=(1.59, 1.33), r=(5e-7, 1e-6))
    # index can be complex
    cs = Sphere(n = (1.59+0.0001j, 1.33+0.0001j), r=(5e-7, 1e-6))
    center = np.array([1e-6, -1e-6, 10e-6])
    cs = Sphere(n = (1.59+0.0001j, 1.33+0.0001j), r=(5e-7, 1e-6),
                      center = center)

def test_Ellipsoid():
    s = Ellipsoid(n = 1.57, r = (1, 2, 3), center = (3, 2, 1))
    assert_equal(s.n, 1.57)
    assert_equal(s.r, (1, 2, 3))
    assert_equal(s.center, (3, 2, 1))
    assert_equal(str(s)[0:9], 'Ellipsoid')

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
def test_Sphere_parameters():
    s = Sphere(n = 1.59+1e-4j, r = 5e-7, center=(1e-6, -1e-6, 10e-6))
    assert_equal(s.parameters, OrderedDict([('center[0]',
    1e-6), ('center[1]', -1e-6), ('center[2]',
    1e-5), ('n', 1.59+1e-4j), ('r',
    5e-07)]))

    sp = Sphere.from_parameters(s.parameters)
    assert_equal(s.r, sp.r)
    assert_equal(s.n, sp.n)
    assert_equal(s.center, sp.center)


@attr('fast')
def test_Composite_construction():
    # empty composite
    comp_empty = Scatterers()
    print comp_empty.get_component_list()

    # composite of multiple spheres
    s1 = Sphere(n = 1.59, r = 5e-7, center = (1e-6, -1e-6, 10e-6))
    s2 = Sphere(n = 1.59, r = 1e-6, center=[0,0,0])
    s3 = Sphere(n = 1.59+0.0001j, r = 5e-7, center=[5e-6,0,0])
    comp_spheres = Scatterers(scatterers=[s1, s2, s3])

    # heterogeneous composite
    cs = Sphere(n=(1.59+0.0001j, 1.33+0.0001j), r=(5e-7, 1e-6),
                      center=[-5e-6, 0,0])
    comp = Scatterers(scatterers=[s1, s2, s3, cs])

    # multi-level composite (contains another composite)
    s4 = Sphere(center=[0, 5e-6, 0])
    comp_spheres.add(s4)
    comp2 = Scatterers(scatterers=[comp_spheres, comp])
    print comp2.get_component_list()

    # even more levels
    comp3 = Scatterers(scatterers=[comp2, cs])
    print comp3

@attr('fast')
def test_like_me():
    s = Sphere(n = 1.59, r = .5, center = (1, -1, 10))
    s2 = s.like_me(center = (0, 2, 10))

    assert_equal(s.r, s2.r)
    assert_equal(s.n, s2.n)
    assert_equal(s2.center, (0, 2, 10))


@attr('fast')
def test_translate():
    s = Sphere(n = 1.59, r = .5, center = (0, 0, 0))
    s2 = s.translated(1, 1, 1)
    assert_equal(s.r, s2.r)
    assert_equal(s.n, s2.n)
    assert_allclose(s2.center, (1, 1, 1))
