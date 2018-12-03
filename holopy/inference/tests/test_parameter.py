# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, Ryan McGorty, Anna Wang, Solomon Barkley
#
# This file is part of HoloPy.
#
# HoloPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HoloPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HoloPy.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from numpy.testing import assert_raises, assert_equal

from holopy.inference.prior import ComplexPrior, Uniform
from holopy.scattering.errors import ParameterSpecificationError
from holopy.core.tests.common import assert_obj_close


def test_parameter():

    assert_raises(ParameterSpecificationError, Uniform, 4, 6, 7)

    p4 = Uniform(-1, 1, 0)
    assert_equal(p4.scale_factor, 0.2)

    p5 = Uniform(1, 4)
    assert_equal(p5.scale_factor, 2.5)

    # if given a guess of 0 and no limits, we fall through to the
    # default of no scaling
    p6 = Uniform(0, np.inf)
    assert_equal(p6.scale_factor, 1)

def test_complex_parameter():
    p = ComplexPrior(1, 2)
    assert_obj_close(p.real, 1)
    assert_obj_close(p.imag, 2)

    assert_equal(p.guess, 1+2j)
