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
from numpy.testing import assert_allclose, assert_equal

from holopy.inference.cmaes import run_cma, CmaStrategy
from holopy.inference.model import Model
from holopy.inference import prior
from holopy.inference.tests.common import SimpleModel


def simplefunc(x):
    """Has a global minium at x = 0.5"""
    return np.array((x - 0.5)**2).mean()


def weightfunc(x, popsize):
    if x == popsize-1:
        return 0
    return 1/(x+1)

data = np.array(.5)
tols = {'maxiter': 2}


def test_run_cma_returns_reproducible_answer():
    # If this test fails, it could be either a change in holopy code
    # or a change in the cma package code.
    popsize = 10
    pars = [prior.Uniform(0, 1), prior.Uniform(0, 1)]
    ndim = len(pars)
    p0 = np.linspace(0, 1, popsize*ndim).reshape((popsize, ndim))

    r = run_cma(simplefunc, pars, p0, weightfunc, tols, seed=1)
    found = r.logger.xrecent.mean()
    correct = 2.871557
    assert_allclose(found, correct, rtol=1e-3)


def test_CmaStrategy():
    mod = SimpleModel()
    strat = CmaStrategy(seed=18, tols=tols, popsize=5)
    r = strat.fit(mod, data)
    assert_allclose(np.mean(r._parameters), .522794, atol=.001)


def test_default_popsize():
    npars = 2
    mod = SimpleModel(npars)
    strat = CmaStrategy(seed=18, tols=tols, popsize=None)
    strat.fit(mod, data)
    assert_equal(strat.popsize, int(2 + npars + np.sqrt(npars)))

