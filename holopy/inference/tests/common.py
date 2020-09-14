# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, and Ryan McGorty, Anna Wang, Solomon Barkley
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

from holopy.inference import prior
from holopy.inference.model import Model
from holopy.inference.result import SamplingResult
from holopy.inference.emcee import EmceeStrategy
from holopy.scattering import Sphere

class SimpleModel(Model):
    def __init__(self, npars=2, noise_sd=1):
        self._parameters = [prior.Uniform(0, 1),
                            prior.Uniform(0, 1)][:npars]
        self._parameter_names = ['x', 'y'][:npars]
        self.constraints = []
        self._maps = {'optics': {}}

    def _residuals(self, pars, data, noise):
        return self.lnposterior(pars, data, None)

    def _lnposterior(self, par_vals, data, dummy):
        x = par_vals
        data = np.array(data)
        return -((x[-1] - data)**2).sum()

    def sample(self, data, strategy=None):
        strategy = self.validate_strategy(strategy, 'sample')
        kwargs = {'samples': None, 'lnprobs': None, 'intervals': None}
        return SamplingResult(data, self, strategy, 0, kwargs)
