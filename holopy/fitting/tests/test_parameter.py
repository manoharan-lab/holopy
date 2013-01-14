# Copyright 2011-2013, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, and Ryan McGorty, Anna Wang
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
from __future__ import division

from nose.tools import assert_raises, assert_equal

from ..parameter import Parameter, par, ComplexParameter
from ..errors import GuessOutOfBoundsError
from ...core.tests.common import assert_obj_close


def test_parameter():
    # basic parameter
    p1 = Parameter(guess = 5, limit = [4, 6])
    # now make a par using shorthands
    p2 = par(5, [4,6])
    # they should be the same
    assert_obj_close(p1, p2)

    p3 = par(limit = 7)

    assert_equal(p3.guess, 7)

    assert_raises(GuessOutOfBoundsError, par, 7, [4, 6])

    assert_raises(GuessOutOfBoundsError, par, 6, 7)

    p4 = par(0, [-1, 1])
    assert_equal(p4.scale_factor, 0.2)

    p5 = par(limit = [1, 4])
    assert_equal(p5.scale_factor, 2.0)

    # if given a guess of 0 and no limits, we fall through to the
    # default of no scaling
    p6 = par(guess = 0)
    assert_equal(p6.scale_factor, 1)

def test_complex_parameter():
    p = ComplexParameter(1, 2)
    assert_obj_close(p.real, par(1, 1))
    assert_obj_close(p.imag, par(2, 2))

    assert_equal(p.guess, 1+2j)
