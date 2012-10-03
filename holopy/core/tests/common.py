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
import types
import inspect
import yaml
from numpy.testing import assert_equal, assert_almost_equal
from ..holopy_object import OrderedDict

from ..io import load, save
from ..metadata import Optics
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

def get_example_data(name, optics):
    if not isinstance(optics, Optics):
        optics = get_example_data_path(optics)
    return load(get_example_data_path(name), optics = optics)
        
def assert_read_matches_write(o):
    tempf = tempfile.NamedTemporaryFile()
    save(tempf, o)
    tempf.flush()
    tempf.seek(0)
    loaded = load(tempf)
    assert_obj_close(o, loaded)

# TODO: rewrite this.  I don't think this the right way to do things anymore, we
# should be comparing _dict entries
def assert_parameters_allclose(actual, desired, rtol=1e-7, atol = 0):
    if isinstance(actual, scatterer.Scatterer):
        actual = actual.parameters
    if isinstance(actual, OrderedDict):
        actual = np.array([p[1] for p in actual.iteritems()])
    if isinstance(desired, scatterer.Scatterer):
        desired = desired.parameters
    if isinstance(desired, OrderedDict):
        desired = np.array([p[1] for p in desired.iteritems()])
    if getattr(actual, 'dtype', None) == 'object' or getattr(desired, 'dtype', None) == 'object':
        # regular allclose will probably fail on objects, so if the scatterer
        # contains objects (like say paramters), compare them with our assert_obj_close
        assert_obj_close(actual, desired)
    else:
        assert_allclose(actual, desired, rtol=rtol, atol = atol)
            
        
def assert_obj_close(actual, desired, rtol=1e-7, atol = 0, context = 'tested_object'):
    if isinstance(actual, np.ndarray) or isinstance(desired, np.ndarray):
        assert_allclose(actual, desired, rtol = rtol, atol = atol, err_msg=context)

    if isinstance(actual, dict) and isinstance(desired, dict):
        for key, val in actual.iteritems():
            assert_obj_close(actual[key], desired[key], context =
                             '{0}[{1}]'.format(context, key))
    elif hasattr(actual, '_dict') and hasattr(desired, '_dict'):
        assert_obj_close(actual._dict, desired._dict, rtol=rtol, atol=atol,
                         context = "{0}._dict".format(context))
    elif isinstance(actual, (list, tuple)):
        assert_equal(len(actual), len(desired), err_msg=context)
        for i, item in enumerate(actual):
            assert_obj_close(actual[i], desired[i], context =
                             '{0}[{1}]'.format(context, i))
    elif isinstance(actual, types.MethodType):
        assert_equal(actual.im_func.func_name, desired.im_func.func_name,
                     err_msg=context)

        # We want to treat Mie.calc_holo and Mie().calc_holo as equal, this code
        # here instantiates a class if possible so these match
        act_obj = actual.im_self
        try:
            act_obj = act_obj()
        except TypeError:
            pass
        des_obj = desired.im_self
        try:
            des_obj = des_obj()
        except TypeError:
            pass

        # now actually compare things 
        assert_obj_close(act_obj, des_obj, rtol=rtol, atol=atol,
                         context = context)
    elif hasattr(actual, '__dict__'):
        for key, val in actual.__dict__.iteritems():
            assert_obj_close(getattr(actual, key), getattr(desired, key), rtol, atol,
                             context = context+'.'+key)
    else:
        try:
            assert_allclose(actual, desired, rtol=rtol, atol=atol,
                            err_msg=context)
        except (TypeError, NotImplementedError):
            assert_equal(actual, desired, err_msg=context)

def verify(result, name, rtol=1e-7):
    location = os.path.split(os.path.abspath(__file__))[0]
    # This gets the filename for the context verify was called from.  It feels
    # really hacky, but it should get the job done.  
    filename = inspect.currentframe().f_back.f_code.co_filename
    location =  os.path.split(filename)[0]
    gold_dir = os.path.join(location, 'gold')
    gold_name = os.path.join(location, 'gold', 'gold_'+name)
    gold_yaml = yaml.load(file(gold_name+'.yaml'))

    full = os.path.join(gold_dir, 'full_data', 'gold_full_{0}.yaml'.format(name))
    if os.path.exists(full):
        assert_obj_close(result, load(full), rtol)


    if isinstance(result, dict):
        assert_obj_close(result, gold_yaml, rtol)
    else:
        for key, val in gold_yaml.iteritems():
            assert_almost_equal(getattr(result, key)(), val, decimal=int(-np.log10(rtol)))

# TODO: update me
def make_golds(result, name):
    '''
    Make new golds for a test

    Parameters
    ----------
    result: Marray
        A result that you want to make the new gold (try to make sure it is
        correct)
    name: string
        The name for the result (this should be something like the test name)
    '''
    
    gold_name = 'gold_'+name

    full = 'gold_full_{0}.yaml'.format(name)
    save(full, result)
    
    gold_dict = {}

    checks = ['min', 'max', 'mean', 'std']

    for check in checks:
        gold_dict[check] = getattr(result, check)()

    simple_checks = 'gold_{0}.yaml'.format(name)
    save(simple_checks, gold_dict)

    print('move {0} to the gold/ directory, and {1} to the gold/full_data/ '
          'directory to regold your local test.  You should also see about '
          'getting the full data somewhere useful for fetching by '
          'others'.format(simple_checks, gold_name))
