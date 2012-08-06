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

import os
import numpy as np
import yaml

from numpy.testing import assert_almost_equal

from ...core import Optics

from ...core.tests.common import assert_allclose, assert_obj_close

wavelen = 658e-9
ypolarization = [0., 1.0] # y-polarized
xpolarization = [1.0, 0.] # x-polarized
divergence = 0
pixel_scale = [.1151e-6, .1151e-6]
index = 1.33

yoptics = Optics(wavelen=wavelen, index=index,
                 pixel_scale=pixel_scale,
                 polarization=ypolarization,
                 divergence=divergence)
    
xoptics = Optics(wavelen=wavelen, index=index,
                 pixel_scale=pixel_scale,
                 polarization=xpolarization,
                 divergence=divergence)

optics=yoptics


def verify(result, name, rtol=1e-7):
    location = os.path.split(os.path.abspath(__file__))[0]
    gold_name = os.path.join(location, 'gold', 'gold_'+name)
    gold_yaml = yaml.load(file(gold_name+'.yaml'))
    if isinstance(result, dict):
        assert_obj_close(result, gold_yaml, rtol)
        return
    if os.path.exists(gold_name + '.npy'):
        gold = np.load(gold_name + '.npy')
        arr = gold
        assert_allclose(arr, gold, rtol)


    for key, val in gold_yaml.iteritems():
        lookup = {'x_comp': 0, 'y_comp': 1, 'z_comp': 2}
        if hasattr(result, 'components'):
            comp, check = key.split('.')
            assert_almost_equal(getattr(result[...,lookup[comp]], check)(), val,
                                decimal=int(-np.log10(rtol)),
                                err_msg = "for {0} {1}".format(name, key))
        else:
            assert_almost_equal(getattr(result, key)(), val, decimal=int(-np.log10(rtol)))

# TODO: update me
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
