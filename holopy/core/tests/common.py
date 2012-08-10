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
import tempfile
import os
from ..io import load, save
from numpy.testing import assert_equal

from ...scattering import scatterer

try:
    from numpy.testing import assert_allclose
except ImportError:
    from numpy import allclose
    def assert_allclose(actual, desired, rtol=1e-07, atol=0, err_msg='',
                        verbose=True):
        if not allclose(actual, desired, rtol=rtol, atol=atol):
            raise AssertionError("""Assertion Error:
Not equal to tolerance rtol={0}, atol={1}

 actual: {2}
 desired: {3}""".format(rtol, atol, actual, desired))

def get_example_data_path(name):
    path = os.path.abspath(__file__)
    path = os.path.join(os.path.split(path)[0], 'exampledata')
    return os.path.join(path, name)

def get_example_data(name, optics_name):
    return load(get_example_data_path(name), optics = get_example_data_path(optics_name))
        
def assert_read_matches_write(o):
    tempf = tempfile.TemporaryFile()
    save(tempf, o)
    tempf.flush()
    tempf.seek(0)
    loaded = load(tempf)
    assert_obj_close(o, loaded)
    
def assert_parameters_allclose(actual, desired, rtol=1e-7, atol = 0):
    if isinstance(actual, scatterer.Scatterer):
        actual = actual.parameters
    if isinstance(actual, dict):
        actual = np.array([p[1] for p in actual.iteritems()])
    if isinstance(desired, scatterer.Scatterer):
        desired = desired.parameters
    if isinstance(desired, dict):
        desired = np.array([p[1] for p in desired.iteritems()])
    if actual.dtype == 'object':
        # regular allclose will probably fail on objects, so if the scatterer
        # contains objects (like say paramters), compare them with our assert_obj_close
        assert_obj_close(actual, desired)
    else:
        assert_allclose(actual, desired, rtol=rtol, atol = atol)

def assert_obj_close(actual, desired, rtol=1e-7, atol = 0, context = None):
    if context is None:
        context = 'tested_object'
    if isinstance(actual, (scatterer.Scatterer, dict)):
        assert_parameters_allclose(actual, desired, rtol, atol)
    elif isinstance(actual, (list, tuple)):
        assert_equal(len(actual), len(desired))
        for i, item in enumerate(actual):
            assert_obj_close(actual[i], desired[i], context =
                             '{0}[{1}]'.format(context, i))
    elif hasattr(actual, '__dict__'):
        for key, val in actual.__dict__.iteritems():
            assert_obj_close(getattr(actual, key), getattr(desired, key), rtol,
                             context = context+'.'+key)
    elif actual is not None and not np.isscalar(actual):
        for i, item in enumerate(actual):
            assert_obj_close(actual[i], desired[i], rtol, atol, context)
    else:
        try:
            assert_equal(actual, desired)
        except AssertionError as e:
            raise AssertionError("\nIn {0}{1}".format(context, str(e)))
