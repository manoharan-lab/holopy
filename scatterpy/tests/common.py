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

import holopy
import scatterpy
import os
import numpy as np
import yaml
from scatterpy.theory.scatteringtheory import ElectricField


from numpy.testing import assert_almost_equal, assert_allclose, assert_equal

wavelen = 658e-9
ypolarization = [0., 1.0] # y-polarized
xpolarization = [1.0, 0.] # x-polarized
divergence = 0
pixel_scale = [.1151e-6, .1151e-6]
index = 1.33

yoptics = holopy.optics.Optics(wavelen=wavelen, index=index,
                               pixel_scale=pixel_scale,
                               polarization=ypolarization,
                               divergence=divergence)
    
xoptics = holopy.optics.Optics(wavelen=wavelen, index=index,
                               pixel_scale=pixel_scale,
                               polarization=xpolarization,
                               divergence=divergence)

optics=yoptics


def verify(result, name):
    scatterpy_location = os.path.split(os.path.abspath(scatterpy.__file__))[0]
    gold_name = os.path.join(scatterpy_location, 'tests', 'gold', 'gold_'+name)
    if os.path.exists(gold_name + '.npy'):
        gold = np.load(gold_name + '.npy')
        assert_allclose(result, gold)

    gold = yaml.load(file(gold_name+'.yaml'))

    for key, val in gold.iteritems():
        if isinstance(result, ElectricField):
            comp, check = key.split('.')
            assert_almost_equal(getattr(getattr(result, comp), check)(), val)
        else:
            assert_almost_equal(getattr(result, key)(), val)
    

def make_golds(result, name):
    '''
    Make new golds for a test

    Parameters
    ----------
    result: Hologram or ElectricField
        A result that you want to make the new gold (try to make sure it is
        correct)
    name: string
        The name for the result (this should be something like the test name)
    '''
    
    gold_name = 'gold_'+name
    if isinstance(result, ElectricField):
        np.save(gold_name+'.npy', result._array())
    else:
        np.save(gold_name+'.npy', result)

    gold_dict = {}

    checks = ['min', 'max', 'mean', 'std']


    for check in checks:
        if isinstance(result, ElectricField):
            comps = ['x_comp', 'y_comp', 'z_comp']
            for comp in comps:
                res = getattr(getattr(result, comp), check)()
                gold_dict['{0}.{1}'.format(comp, check)] = res
        else:
            gold_dict[check] = getattr(result, check)()

    yaml.dump(gold_dict, file(gold_name+'.yaml','w'))
        
def assert_parameters_allclose(actual, desired, rtol=1e-7, atol = 0):
    if isinstance(actual, scatterpy.scatterer.Scatterer):
        actual = actual.parameters
    if isinstance(actual, dict):
        actual = np.array([p[1] for p in actual.iteritems()])
    if isinstance(desired, scatterpy.scatterer.Scatterer):
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
    if isinstance(actual, (scatterpy.scatterer.Scatterer, dict)):
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
