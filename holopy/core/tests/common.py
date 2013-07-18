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

import numpy as np
import tempfile
import os
import types
import inspect
import yaml
import shutil
from numpy.testing import assert_equal, assert_almost_equal


from .. import load, save
from ..io import get_example_data, get_example_data_path

# tests should fail if they give warnings
import warnings
warnings.simplefilter("error")


from numpy.testing import assert_allclose

def assert_read_matches_write(o):
    tempf = tempfile.NamedTemporaryFile()
    save(tempf, o)
    tempf.flush()
    tempf.seek(0)
    loaded = load(tempf)
    assert_obj_close(o, loaded)

def assert_obj_close(actual, desired, rtol=1e-7, atol = 0, context = 'tested_object'):
    # we go ahead and try to compare anything using numpy's assert allclose, if
    # it fails it probably gives more useful error messages than later options,
    # and catching NotImplementedError and TypeError should cause this to
    # silently fall through for other types
    try:
        assert_allclose(actual, desired, rtol = rtol, atol = atol, err_msg=context)
    except (NotImplementedError, TypeError):
        pass

    if isinstance(actual, dict) and isinstance(desired, dict):
        for key, val in actual.iteritems():
            assert_obj_close(actual[key], desired[key], context = '{0}[{1}]'.format(context, key),
                             rtol = rtol, atol = atol)
    elif hasattr(actual, '_dict') and hasattr(desired, '_dict'):
        assert_obj_close(actual._dict, desired._dict, rtol=rtol, atol=atol,
                         context = "{0}._dict".format(context))
    elif isinstance(actual, (list, tuple)):
        assert_equal(len(actual), len(desired), err_msg=context)
        for i, item in enumerate(actual):
            assert_obj_close(actual[i], desired[i], context = '{0}[{1}]'.format(context, i),
                             rtol = rtol, atol = atol)
    elif isinstance(actual, types.MethodType):
        assert_method_equal(actual, desired, context)
    elif hasattr(actual, '__dict__'):
        assert_obj_close(actual.__dict__, desired.__dict__, rtol = rtol,
                         atol = atol, context = context + '.__dict__')
    else:
        try:
            assert_allclose(actual, desired, rtol=rtol, atol=atol, err_msg=context)
        except (TypeError, NotImplementedError):
            assert_equal(actual, desired, err_msg=context)

def assert_method_equal(actual, desired, context):
    # Check that the functions are the same
    assert_equal(actual.im_func.func_name, desired.im_func.func_name, err_msg=context)

    # check that the objects are the same

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
    assert_obj_close(act_obj, des_obj, context = context)

def verify(result, name, rtol=1e-7, atol=1e-8):
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
        assert_obj_close(result, gold_yaml, rtol, atol)
    else:
        for key, val in gold_yaml.iteritems():
            assert_almost_equal(getattr(result, key)(), val, decimal=int(-np.log10(rtol)))

# TODO: update me
def make_golds(result, name, moveto=None):
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

    if moveto:
        root = os.path.split(os.path.abspath(__file__))[0]
        location = os.path.join(root, 'holopy', moveto, 'tests')
        gold_dir = os.path.join(location, 'gold')
        full_dir = os.path.join(gold_dir, 'full_data')
        shutil.move(simple_checks, gold_dir)
        shutil.move(gold_name, full_dir)
    else:
        print('move {0} to the gold/ directory, and {1} to the gold/full_data/ '
              'directory to regold your local test.  You should also see about '
              'getting the full data somewhere useful for fetching by '
              'others'.format(simple_checks, gold_name))
