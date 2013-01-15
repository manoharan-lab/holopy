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

import tempfile

import numpy as np

from nose.plugins.attrib import attr
from numpy.testing import assert_equal
from ...scattering.theory import Mie
from ...scattering.scatterer import Sphere, Spheres, Scatterer
from .. import fit, par, Model
from ...core.tests.common import assert_obj_close, get_example_data


@attr('medium')
def test_Tying():

    #tied parameters
    n1 = par(1.59)
    sc = Spheres([Sphere(n = n1, r = par(0.5e-6), center = np.array([10., 10., 20.])),
              Sphere(n = n1, r = par(0.5e-6), center = np.array([9., 11., 21.]))])
    model = Model(sc, Mie.calc_holo, alpha = par(.7,[.6,1]))
              
    assert_equal(model.parameters[0].guess, 1.59)
    assert_equal(model.parameters[1].guess, 5e-7)
    assert_equal(len(model.parameters),4)