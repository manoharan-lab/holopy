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

import numpy as np
from numpy.testing import assert_allclose

from holopy.fitting.model import BaseModel
from holopy.inference import prior, AlphaModel
from holopy.inference.sample import sample_emcee, EmceeStrategy, tempered_sample
from holopy.core.metadata import detector_grid
from holopy.fitting import make_subset_data
from holopy.scattering import Sphere, calc_holo

class SimpleModel:
    def __init__(self, x=None):
        self.parameters = [x]

    def lnposterior(self, par_vals, data):
        x = par_vals
        return -((x-data)**2).sum()

data = np.array(.5)

def test_sample_emcee():
    nwalkers = 10
    ndim = 1
    mod = SimpleModel()
    p0 = np.linspace(0, 1, 10).reshape((nwalkers, ndim))
    r = sample_emcee(mod, data, nwalkers, 500, p0, threads=None, seed=40)
    assert_allclose(r.chain[r.lnprobability==r.lnprobability.max()], .5, rtol=.001)

def test_EmceeStrategy():
    mod = SimpleModel(prior.Uniform(0, 1))
    strat = EmceeStrategy(10, None, None, seed=40)
    r = strat.sample(mod, data, 500)
    assert_allclose(r.MAP, .5, rtol=.001)
