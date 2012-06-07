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
from __future__ import division
import numpy as np
from numpy.testing import assert_allclose, assert_equal

import scatterpy

    
def assert_parameters_allclose(actual, desired, rtol=1e-3):
    if isinstance(actual, scatterpy.scatterer.Scatterer):
        actual = actual.parameters
    if isinstance(actual, dict):
        actual = np.array([p[1] for p in actual.iteritems()])
    if isinstance(desired, scatterpy.scatterer.Scatterer):
        desired = desired.parameters
    if isinstance(desired, dict):
        desired = np.array([p[1] for p in desired.iteritems()])
    assert_allclose(actual, desired, rtol=rtol)

def assert_obj_close(actual, desired, rtol=1e-3):
    if isinstance(actual, (scatterpy.scatterer.Scatterer, dict)):
        assert_parameters_allclose(actual, desired, rtol)
    elif hasattr(actual, '__dict__'):
        for key, val in actual.__dict__.iteritems():
            assert_obj_close(getattr(actual, key), getattr(desired, key), rtol)
    elif actual is not None and not np.isscalar(actual):
        for i, item in enumerate(actual):
            assert_obj_close(actual[i], desired[i], rtol)
    else:
        assert_equal(actual, desired)
        
