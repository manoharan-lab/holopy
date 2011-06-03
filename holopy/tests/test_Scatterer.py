# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca W. Perry,
# Jerome Fung, and Ryan McGorty
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
from numpy.testing import assert_, assert_equal, assert_array_almost_equal
import os
import string
from nose.plugins.attrib import attr

from holopy.model.scatterer import Sphere, CoatedSphere
from holopy.model.scatterer import SphereDimer #, SphereCluster

class TestScatterer:

    def test_Sphere_construction(self):
        s = Sphere(n = 1.59, r = 5e-7, x = 1e-6, y = -1e-6, z = 10e-6)
        s = Sphere(n = 1.59, r = 5e-7)
        # index can be complex
        s = Sphere(n = 1.59+0.001j, r = 5e-7)
        s = Sphere()

    def test_Sphere_construct_list(self):
        # specify centerition as list
        center = [1e-6, -1e-6, 10e-6]
        s = Sphere(n = 1.59+0.001j, r = 5e-7, center = center)
        assert_equal(s.center, np.array(center))

    def test_Sphere_construct_tuple(self):
        # specify centerition as list
        center = (1e-6, -1e-6, 10e-6)
        s = Sphere(n = 1.59+0.001j, r = 5e-7, center = center)
        assert_equal(s.center, np.array(center))

    def test_Sphere_construct_array(self):
        # specify centerition as list
        center = np.array([1e-6, -1e-6, 10e-6])
        s = Sphere(n = 1.59+0.001j, r = 5e-7, center = center)
        assert_equal(s.center, center)
        
    def test_CoatedSphere_construction(self):
        cs = CoatedSphere(n = 1.59, r1 = 5e-7, r2 = 1e-6, x = 1e-6,
                    y = -1e-6, z = 10e-6) 
        cs = CoatedSphere(n = 1.59, r1 = 5e-7, r2 = 1e-6)
        # index can be complex
        cs = CoatedSphere(n = 1.59+0.001j, r1 = 5e-7)
        center = np.array([1e-6, -1e-6, 10e-6])
        cs = CoatedSphere(n = 1.59+0.001j, r1 = 5e-7, r2 = 1e-6,
                          center = center) 
        cs = CoatedSphere()

    def test_SphereDimer_construction(self):
        sd = SphereDimer()

    # def test_SphereCluster_construction(self):
    #     sc = SphereCluster(n = [1.59+0.001j, 1.59+0.001j], r = [])
