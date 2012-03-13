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
from nose.tools import assert_raises
from numpy.testing import assert_equal
from nose.tools import with_setup
from nose.plugins.attrib import attr
from scatterpy.scatterer import Sphere
from common import assert_allclose

from scatterpy.theory.scatteringtheory import (ScatteringTheory, ElectricField,
                                               InvalidElectricFieldComputation)

import common

# nose setup/teardown methods
def setup_optics():
    # set up optics class for use in several test functions
    global optics
    optics = common.optics
    
def teardown_optics():
    global optics
    del optics


@attr('fast')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_abstract_theory():
    theory = ScatteringTheory(optics)

    assert_raises(NotImplementedError, theory.calc_field, Sphere())

@attr('fast')
def test_ElectricField():
    x = np.ones((5, 5))
    y = np.ones((5, 5))
    z = np.ones((5, 5))

    e1 = ElectricField(x, y, z, 0, .66)
    e2 = ElectricField(x, y, z, 0, .7)

    with assert_raises(InvalidElectricFieldComputation) as cm:
        e1 * e1
    assert_equal(str(cm.exception), "Invalid Electric Computation: "
                 "multiplication by nonscalar values not yet implemented")

    with assert_raises(InvalidElectricFieldComputation) as cm:
        e1 + e2
    assert_equal(str(cm.exception), "Invalid Electric Computation: "
                 "Superposition of fields with different wavelengths is not "
                 "implemented")

    assert_allclose(e1 * 2.0, 2.0 * e1)

    
