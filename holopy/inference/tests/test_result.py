# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
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

import xarray as xr
import numpy as np
from numpy.testing import assert_raises, assert_equal, assert_almost_equal

from holopy.inference.prior import Uniform
from holopy.inference.result import UncertainValue, FitResult, SamplingResult
from holopy.inference import AlphaModel, CmaStrategy, EmceeStrategy
from holopy.scattering import Sphere
from holopy.scattering.errors import MissingParameter
from holopy.core.metadata import detector_grid, update_metadata
from holopy.core.tests.common import assert_read_matches_write


data = update_metadata(detector_grid(shape=10, spacing=2), 1.33, 0.660, (0,1))
par_s = Sphere(n=Uniform(1.5, 1.65), r=Uniform(0.5, 0.7), center=[10, 10, 10])
model = AlphaModel(par_s, alpha=Uniform(0.6, 0.9, guess=0.8))
strat = CmaStrategy()
intervals = [UncertainValue(1.6, 0.1, name='n'), UncertainValue(0.6,
                        0.1, name='r'), UncertainValue(0.7, 0.1, name='alpha')]
samples = xr.DataArray([[1,2,3],[4,5,6]], dims=['dim1', 'dim2'], 
                coords={'dim1':['left', 'right'], 'dim2':['r', 'b', 'g']})

def test_UncertainValue():
    uncval1 = UncertainValue([10], np.array(2))
    uncval2 = UncertainValue(10, 2, 2)
    assert_equal(uncval1, uncval2)

def test_FitResult():
    # test no intervals
    assert_raises(MissingParameter, FitResult, data, model, strat, 10)
    
    # test properties
    result = FitResult(data, model, CmaStrategy(), 10, {'intervals':intervals})
    assert_equal(result.guess, [1.6, 0.6, 0.7])
    assert_equal(result._names, ['n', 'r', 'alpha'])
    assert_equal(result.parameters, {'r':0.6, 'n':1.6, 'alpha':0.7})
    assert_equal(result.scatterer, Sphere(n=1.6, r=0.6, center=[10, 10, 10]))
    
    # test calculations
    assert_almost_equal(result.max_lnprob, -138.17557, decimal=5)
    assert hasattr(result, '_max_lnprob')
    assert_equal(result.best_fit.shape, (10,10,1))
    assert_almost_equal(result.best_fit.mean(), 1.005387, decimal=6)
    assert hasattr(result, '_best_fit')
    
    # test add_attr
    result.add_attr({'foo':'bar', 'foobar':7})
    assert_equal(result.foo, 'bar')

def test_io():
    # test base io
    result = FitResult(data, model, CmaStrategy(), 10, {'intervals':intervals})
    assert_read_matches_write(result)
    
    # test datarray attr
    result.add_attr({'samples': samples})
    assert_read_matches_write(result)
    
    # test saved calculations
    result.best_fit
    result.max_lnprob
    assert_read_matches_write(result)
    
def test_SamplingResult():
    samples = np.array([[[1, 2], [11, 12], [3, 3]], [[0, 3], [1, 3], [5, 6]]])
    samples = xr.DataArray(samples, dims = ['walker', 'chain', 'parameter'],
                                        coords={'parameter':['p1','p2']})
    lnprobs = xr.DataArray([[10, 9, 8], [7, 6, 5]], dims = ['walker', 'chain'])
    result = SamplingResult(data, model, EmceeStrategy(), 10, 
                                kwargs={'samples':samples, 'lnprobs':lnprobs})
    assert_equal(result.intervals[0].guess, 1)
    assert_equal(result.intervals[1].guess, 2)
    assert_equal(result.intervals[0].name, 'p1')
    assert_equal(result.intervals[1].name, 'p2')
    
    result = result.burn_in(1)
    assert_equal(result.intervals[0].guess, 11)
    assert_equal(result.intervals[1].guess, 12)
    assert_equal(result.intervals[0].name, 'p1')
    assert_equal(result.intervals[1].name, 'p2')
    assert_equal(result.samples.shape, (2, 2, 2))
    
