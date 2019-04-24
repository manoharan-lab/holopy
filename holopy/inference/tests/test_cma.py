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

from holopy.inference.cmaes import run_cma, CmaStrategy
from holopy.inference.model import BaseModel
from holopy.inference import prior

class SimpleModel(BaseModel):
    def __init__(self, x=prior.Uniform(0,1), y=prior.Uniform(0,1)):
        self._parameters = [x,y]

    def lnposterior(self, par_vals, data, dummy):
        x = par_vals
        return -((x[None]-data)**2).sum()

def simplefunc(x):
    return -np.array((x-0.5)**2).mean()

def weightfunc(x, popsize):
    if x == popsize-1:
        return 0
    return 1/(x+1)

data = np.array(.5)
tols = {'maxiter':2}

def test_run_cma():
    popsize = 10
    pars = [prior.Uniform(0,1), prior.Uniform(0,1)]
    ndim = len(pars)
    p0 = np.linspace(0, 1, popsize*ndim).reshape((popsize, ndim))
    r = run_cma(simplefunc, pars, p0, weightfunc, tols, seed=1)
    assert_allclose(r.logger.xrecent.mean(), 3, rtol=.01)

def test_CmaStrategy():
    mod = SimpleModel()
    strat = CmaStrategy(seed=18, tols=tols)
    r = strat.optimize(mod, data, 5)
    assert_allclose(np.mean(r.guess), .55, atol=.001)
