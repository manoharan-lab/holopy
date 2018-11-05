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


from numpy.testing import assert_raises, assert_equal

from ..parameter import ComplexParameter
from ..parameter import Parameter as par
from ...scattering.errors import ParameterSpecificationError
from ...core.tests.common import assert_obj_close


def test_parameter():

    assert_raises(ParameterSpecificationError, par, 7, [4, 6])

    assert_raises(ParameterSpecificationError, par, 6, 7)

    p4 = par(0, [-1, 1])
    assert_equal(p4.scale_factor, 0.2)

    p5 = par(limit = [1, 4])
    assert_equal(p5.scale_factor, 2.5)

    # if given a guess of 0 and no limits, we fall through to the
    # default of no scaling
    p6 = par(guess = 0)
    assert_equal(p6.scale_factor, 1)

def test_complex_parameter():
    p = ComplexParameter(1, 2)
    assert_obj_close(p.real, par(1, 1))
    assert_obj_close(p.imag, par(2, 2))

    assert_equal(p.guess, 1+2j)
