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
from __future__ import division

from holopy.fitting.parameter import Parameter
from holopy.inference.noise_model import AlphaModel

from collections import defaultdict
import itertools

class TimeEvolution(Parameter):
    def __init__(self, initial):
        self.initial = initial

    @property
    def name(self):
        return self.initial.name

    @name.setter
    def name(self, value):
        self.initial.name = value

class TimeIndependent(TimeEvolution):
    pass

class Diffusive(TimeEvolution):
    def __init__(self, initial, diffusion_constant):
        super(Diffusive, self).__init__(initial)
        self.diffusion_constant = diffusion_constant

class TimeSeriesAlphaModel(AlphaModel):
    def __init__(self, scatterer, noise_sd, alpha, n_frames, medium_index=None, wavelen=None, optics=None, theory='auto'):
        self.n_frames = n_frames
        super(TimeSeriesAlphaModel, self).__init__(scatterer, medium_index=medium_index, wavelen=wavelen, optics=optics, noise_sd=noise_sd, alpha=alpha, theory=theory)
        self.time_independent_pars = []
        self.time_dependent_pars = []
        for par in self._parameters:
            if isinstance(par, TimeIndependent):
                self.time_independent_pars.append(par)
            else:
                self.time_dependent_pars.append(par)

    @property
    def parameters(self):
        for par in self.time_independent_pars:
            yield par.initial
        for i in range(self.n_frames):
            for par in self.time_dependent_pars:
                # This order is less efficent on this end, but makes
                # working with the parameters much more convenient
                cls = par.__class__
                d = par._dict
                name = d.pop('name')
                yield cls(name='{}@{}'.format(name, i),  **d)

    def lnlike(self, par_vals, data):
        time_independent_pars = {p.name: v for p, v in zip(self.time_independent_pars, par_vals)}
        time_dependent_pars = defaultdict(dict)

        for t in range(self.n_frames):
            for i, par in enumerate(self.time_dependent_pars):
                time_dependent_pars[t][par.name] = par_vals[t*self.n_frames+i]

        lnlike = 0
        for time in time_dependent_pars:
            d = dict(time_independent_pars, **time_dependent_pars[time])
            lnlike += self._lnlike(d, data[time])

        return lnlike


    def make_frame_models(self, pars):
        pars = self._pack(pars)
        const_pars = {}
        frames = defaultdict(dict)
        for par in pars:
            if '@' in par.name:
                name, t = par.name.split('@')
                t = int(t)
                frames[t][name] = par
            else:
                const_pars[par.name] = par
        models = {}
        for frame in frames.keys():
            d = dict(const_pars, **frames[frame])
            alpha = d.pop('alpha', self.alpha)
            s = model.scatterer.make_from(d)
            args = dict(model._dict, alpha=alpha, scatterer=s)
            models[frame] = AlphaModel(**args)

        return models


def fit_series(model, data, **kwargs):
    frames = data.shape[2]
