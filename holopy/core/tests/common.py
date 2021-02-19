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

import tempfile
import os
import warnings
import types
import inspect
import yaml
import shutil
import pickle

import xarray as xr
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from nose.plugins import Plugin

from holopy.core.io import load, save, get_example_data
from holopy.core.errors import PerformanceWarning


class HoloPyCatchWarnings(Plugin):
    name='holopycatchwarnings'

    def options(self, parser, env=os.environ):
        super(HoloPyCatchWarnings, self).options(parser, env=env)

    def configure(self, options, conf):
        super(HoloPyCatchWarnings, self).configure(options, conf)
        if not self.enabled:
            return

    def beforeTest(self, test):
        warnings.simplefilter("error")
        warnings.simplefilter(action="ignore", category=PerformanceWarning)
        warnings.simplefilter(action="ignore", category=FutureWarning)


def assert_read_matches_write(original):
    with tempfile.NamedTemporaryFile(suffix='.h5') as tempf:
        save(tempf.name, original)
        loaded = load(tempf.name)
    # For now our code for writing xarrays to hdf5 ends up with them
    # picking up a name attribute that their predecessor may not have,
    # so correct for that if it is true.
    if hasattr(loaded, 'name') and original.name is None:
        loaded.name = None
    assert_obj_close(original, loaded)


def assert_pickle_roundtrip(o, cPickle_only=False):
    # TODO: Our pickling code currently works for cPickle but fails in
    # a memoization check in regular pickle., for now I am testing
    # cPickle only in those cases, but it would be good to fix that
    # and test both in all cases
    if not cPickle_only:
        assert_obj_close(o, pickle.loads(pickle.dumps(o)))
    assert_obj_close(o, pickle.loads(pickle.dumps(o)))



def assert_obj_close(actual, desired, rtol=1e-7, atol = 0, context = 'tested_object'):
    # we go ahead and try to compare anything using numpy's assert allclose, if
    # it fails it probably gives more useful error messages than later options,
    # and catching NotImplementedError and TypeError should cause this to
    # silently fall through for other types
    if isinstance(actual, np.ndarray) and isinstance(desired, np.ndarray):
        assert_allclose(actual, desired, rtol = rtol, atol = atol, err_msg=context)

    if (isinstance(desired, xr.DataArray) and isinstance(actual, xr.DataArray)
            and hasattr(actual, "_indexes")):
        # as of xarray 0.12.1, saved and reloaded objects do not maintain
        # the redundant ._indexes attribute
        desired._indexes = actual._indexes

    # if None, let some things that are functially equivalent to None pass
    nonelike = [None, {}]
    if actual is None or desired is None:
        if actual in nonelike and desired in nonelike:
            return

    if isinstance(actual, dict) and isinstance(desired, dict):
        for key, val in actual.items():
            if key in ['_id', '_encoding', '_coords']:
                # these are implementation specific dict keys that we
                # shouldn't expect to be identical, so ignore them
                continue
            assert_obj_close(actual[key], desired[key], rtol=rtol, atol=atol,
                             context='{0}[{1}]'.format(context, key))
    elif hasattr(actual, '_dict') and hasattr(desired, '_dict'):
        assert_obj_close(actual._dict, desired._dict, rtol=rtol, atol=atol,
                         context = "{0}._dict".format(context))
    elif isinstance(actual, (list, tuple)):
        assert_equal(len(actual), len(desired), err_msg=context)
        for i, item in enumerate(actual):
            assert_obj_close(actual[i], desired[i], rtol=rtol, atol=atol,
                             context = '{0}[{1}]'.format(context, i))
    elif isinstance(actual, types.MethodType):
        assert_method_equal(actual, desired, context)
    elif hasattr(actual, '__dict__') and hasattr(desired, '__dict__'):
        assert_obj_close(actual.__dict__, desired.__dict__, rtol = rtol,
                         atol = atol, context = context + '.__dict__')
    else:
        try:
            assert_allclose(actual, desired, rtol=rtol, atol=atol, err_msg=context)
        except (TypeError, NotImplementedError):
            assert_equal(actual, desired, err_msg=context)

def assert_method_equal(actual, desired, context):
    # Check that the functions are the same
    assert_equal(actual.__func__.__name__, desired.__func__.__name__, err_msg=context)

    # check that the objects are the same

    # We want to treat Mie.calc_holo and Mie().calc_holo as equal, this code
    # here instantiates a class if possible so these match
    act_obj = actual.__self__
    try:
        act_obj = act_obj()
    except TypeError:
        pass
    des_obj = desired.__self__
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
    with open(gold_name+'.yaml') as gold_file:
        gold_yaml = yaml.safe_load(gold_file)

    full = os.path.join(gold_dir, 'full_data', 'gold_full_{0}'.format(name))
    if os.path.exists(full):
        try:
            assert_obj_close(result, load(full), rtol)
        except OSError:
            # This will happen if you don't have git lfs installed and we
            # attempt to open the placeholder file. In that case we just test
            # the summary data, same as if the full data isn't present.
            pass

    if isinstance(result, dict):
        assert_obj_close(result, gold_yaml, rtol, atol)
    else:
        for key, val in gold_yaml.items():
            assert_obj_close(getattr(result, key)(), val, rtol, atol)


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
        print(('move {0} to the gold/ directory, and {1} to the gold/full_data/ '
              'directory to regold your local test.  You should also see about '
              'getting the full data somewhere useful for fetching by '
              'others'.format(simple_checks, gold_name)))
