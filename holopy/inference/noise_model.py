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

from holopy.fitting.model import BaseModel
from holopy.fitting.parameter import Parameter
from holopy.core.holopy_object import HoloPyObject
from holopy.scattering.errors import MultisphereFailure, InvalidScatterer

import numpy as np
import pandas as pd
from copy import copy
from holopy.scattering.calculations import calc_field, calc_holo

class NoiseModel(BaseModel):
    def __init__(self, scatterer, noise_sd, medium_index=None, illum_wavelen=None, illum_polarization=None, theory='auto'):
        super().__init__(scatterer, medium_index=medium_index, illum_wavelen=illum_wavelen, illum_polarization=illum_polarization, theory=theory)
        self._use_parameter(noise_sd, 'noise_sd')

    def _pack(self, vals):
        return {par.name: val for par, val in zip(self.parameters, vals)}

    def lnprior(self, par_vals):
        if isinstance(par_vals, dict):
            return sum([p.lnprob(par_vals[p.name]) for p in self.parameters])
        else:
            return sum([p.lnprob(v) for p, v in zip(self.parameters, par_vals)])

    def lnposterior(self, par_vals, data):
        lnprior = self.lnprior(par_vals)
        # prior is sometimes used to forbid thing like negative radius
        # which will fail if you attempt to compute a hologram of, so
        # don't try to compute likelihood where the prior already
        # forbids you to be
        if lnprior == -np.inf:
            return lnprior
        else:
            return lnprior + self.lnlike(par_vals, data)

    def _fields(self, pars, schema):
        def get_par(name):
            return pars.pop(name, self.par(name, schema))
        optics, scatterer = self._optics_scatterer(pars, schema)
        try:
            return calc_field(schema, scatterer, theory=self.theory, **optics)
        except (MultisphereFailure, InvalidScatterer):
            return -np.inf

    def _lnlike(self, pars, data):
        noise_sd = pars.pop('noise_sd', self.noise_sd)
        holo = self._holo(pars, data)
        N = data.size
        return (-N*np.log(noise_sd*np.sqrt(2*np.pi)) -
                ((holo-data)**2).sum()/(2*noise_sd**2))

    def lnlike(self, par_vals, data):
        return self._lnlike(self._pack(par_vals), data)




class AlphaModel(NoiseModel):
    def __init__(self, scatterer, noise_sd, alpha, medium_index=None, illum_wavelen=None, illum_polarization=None, theory='auto'):
        super().__init__(scatterer, medium_index=medium_index, illum_wavelen=illum_wavelen, illum_polarization=illum_polarization, theory=theory, noise_sd=noise_sd)
        self._use_parameter(alpha, 'alpha')

    def _holo(self, pars, schema, alpha=None):
        if alpha is not None:
            alpha = alpha
        else:
            alpha = self.get_par('alpha', pars)

        optics, scatterer = self._optics_scatterer(pars, schema)

        try:
            return calc_holo(schema, scatterer, theory=self.theory, **optics)
        except (MultisphereFailure, InvalidScatterer):
            return -np.inf
