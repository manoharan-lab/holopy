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
Test construction and manipulation of scattering theory objects.

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
from holopy.model.scatterer.spherecluster import SphereClusterDefError

from holopy.model.theory import Mie
from holopy.model.calculate import calc_field, calc_holo, calc_intensity

class TestTheory:

    def test_Mie_construction(self):
        theory = Mie()
        theory = Mie(size=(256,256))
        theory = Mie(size=(256,256))

#    def test_SCSM_construction

#    def test_Mie_calc_field(self):
#        scatterer = Sphere(n=1.59, r=5e-7, x=1e-6, y=-1e-6, z=10e-6)
#        theory = Mie(size=256)
#        calc_field(scatterer, theory=theory, )
