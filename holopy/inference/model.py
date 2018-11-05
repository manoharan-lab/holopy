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

from copy import copy

import numpy as np
import xarray as xr

from holopy.core.metadata import dict_to_array
from holopy.core.utils import ensure_array, ensure_listlike
from holopy.core.holopy_object import HoloPyObject
from holopy.scattering.errors import (MultisphereFailure,
                                InvalidScatterer, MissingParameter)
from holopy.scattering.calculations import calc_holo
from holopy.scattering.theory import MieLens
from holopy.scattering.scatterer import (_expand_parameters,
                                         _interpret_parameters)
from holopy.fitting import make_subset_data
from holopy.inference.prior import Prior, Fixed

class BaseModel(HoloPyObject):
    """Model probabilites of observing data

    Compute probabilities that observed data could be explained by a set of
    scatterer and observation parameters.
    """
    def __init__(self, scatterer, noise_sd=None, medium_index=None,
                 illum_wavelen=None, illum_polarization=None, theory='auto',
                 constraints=[]):
        self.scatterer = scatterer
        self.constraints = ensure_listlike(constraints)
        self._parameters = []
        self._use_parameters(scatterer.parameters, False)
        if not np.isscalar(noise_sd):
            noise_sd = ensure_array(noise_sd)
        self._use_parameters({'medium_index': medium_index,
                             'illum_wavelen':illum_wavelen,
                             'illum_polarization':illum_polarization,
                             'theory':theory, 'noise_sd':noise_sd})

    @property
    def parameters(self):
        return {par.name:par for par in self._parameters}

    def get_parameter(self, name, pars, schema=None):
        if name in pars.keys():
            return pars[name]
        elif hasattr(self, name):
            return getattr(self, name)
        try:
            getattr(schema, name)
        except:
            raise MissingParameter(name)

    def _use_parameters(self, parameters, as_attr=True):
        if as_attr:
            for name, par in parameters.items():
                if par is not None:
                    setattr(self, name, par)
        parameters = dict(_expand_parameters(parameters.items()))
        for key, val in parameters.items():
            if isinstance(val, Prior) and not isinstance(val, Fixed):
                self._parameters.append(copy(val))
                self._parameters[-1].name = key

    def _optics_scatterer(self, pars, schema):
        optics_keys = ['medium_index', 'illum_wavelen', 'illum_polarization']
        optics = {key:self.get_parameter(key, pars, schema)
                            for key in optics_keys}
        scatterer = self.scatterer.from_parameters(pars)
        return optics, scatterer


    def lnprior(self, par_vals):
        for constraint in self.constraints:
            tocheck = self.scatterer.from_parameters(par_vals)
            if not constraint.check(tocheck):
                return -np.inf

        if isinstance(par_vals, dict):
            return sum([p.lnprob(par_vals[p.name]) for p in self._parameters])
        else:
            return sum([p.lnprob(v) for p, v in zip(self._parameters, par_vals)])

    def lnposterior(self, par_vals, data, pixels=None):
        lnprior = self.lnprior(par_vals)
        # prior is sometimes used to forbid thing like negative radius
        # which will fail if you attempt to compute a hologram of, so
        # don't try to compute likelihood where the prior already
        # forbids you to be
        if lnprior == -np.inf:
            return lnprior
        else:
            if pixels is not None:
                data = make_subset_data(data, pixels=pixels)
            return lnprior + self.lnlike(par_vals, data)

    def forward(self, pars, detector):
        raise NotImplementedError("Implement in subclass")


    def lnlike(self, pars, data):
        """
        Compute the likelihood for pars given data

        Parameters
        -----------
        pars: dict(string, float)
            Dictionary containing values for each parameter
        data: xarray
            The data to compute likelihood against
        """
        if not isinstance(pars, dict):
            pars = {par.name:val for par, val in zip(self._parameters, pars)}
        noise_sd = dict_to_array(data,
                                self.get_parameter('noise_sd', pars, data))
        forward_model = self.forward(pars, data)
        N = data.size
        log_likelihood = np.asscalar(
            -N/2 * np.log(2 * np.pi) -
            N * np.mean(np.log(ensure_array(noise_sd))) -
            ((forward_model - data)**2 / (2 * noise_sd**2)).values.sum())
        return log_likelihood


class LimitOverlaps(HoloPyObject):
    """
    Constraint prohibiting overlaps beyond a certain tolerance.
    fraction is the largest overlap allowed, in terms of sphere diameter.

    """
    def __init__(self, fraction=.1):
        self.fraction = fraction

    def check(self, s):
        return s.largest_overlap() <= ((np.min(s.r) * 2) * self.fraction)


class AlphaModel(BaseModel):
    def __init__(self, scatterer, noise_sd=None, alpha=1, medium_index=None,
                 illum_wavelen=None, illum_polarization=None, theory='auto',
                 constraints=[]):
        super().__init__(scatterer, noise_sd, medium_index, illum_wavelen,
                         illum_polarization, theory, constraints)
        self._use_parameters({'alpha':alpha})

    def forward(self, pars, detector):
        """
        Compute a hologram from pars with dimensions and metadata of detector,
        scaled by alpha.

        Parameters
        -----------
        pars: dict(string, float)
            Dictionary containing values for each parameter used to compute
            the hologram. Possible parameters are given by self.parameters.
        detector: xarray
            dimensions of the resulting hologram. Metadata taken from
            detector if not given explicitly when instantiating self.
        """
        alpha = self.get_parameter('alpha', pars, detector)
        optics, scatterer = self._optics_scatterer(pars, detector)
        try:
            return calc_holo(detector, scatterer, theory=self.theory,
                             scaling=alpha, **optics)
        except (MultisphereFailure, InvalidScatterer):
            return -np.inf


# TODO: Change the default theory (when it is "auto") to be
# selected by the model.
# -- this is a little trickier than it sounds, because
# hlopy.scattering.determine_theory picks based off of whether the
# object is 1 sphere or a collection of spheres etc. So you can't
# pass MieLens as a theory
# For now it would be OK since PerfectLensModel only works with single
# spheres or superpositions, but I'm going to leave this for later.
class ExactModel(BaseModel):
    def __init__(self, scatterer, calc_func=calc_holo, noise_sd=None,
                 medium_index=None, illum_wavelen=None,
                 illum_polarization=None, theory='auto', constraints=[]):
        super().__init__(scatterer, noise_sd, medium_index, illum_wavelen,
                         illum_polarization, theory, constraints)

    def forward(self, pars, detector):
        """
        Compute a forward model (the hologram)

        Parameters
        -----------
        pars: dict(string, float)
            Dictionary containing values for each parameter used to compute
            the hologram. Possible parameters are given by self.parameters.
        detector: xarray
            dimensions of the resulting hologram. Metadata taken from
            detector if not given explicitly when instantiating self.
        """
        optics, scatterer = self._optics_scatterer(pars, detector)
        try:
            return calc_func(detector, scatterer, theory=self.theory, **optics)
        except (MultisphereFailure, InvalidScatterer):
            return -np.inf


class PerfectLensModel(BaseModel):
    theory_params = ['lens_angle']
    def __init__(self, scatterer, noise_sd=None, lens_angle=1.0,
                 medium_index=None, illum_wavelen=None, theory='auto',
                 illum_polarization=None, constraints=[]):
        super().__init__(scatterer, noise_sd, medium_index, illum_wavelen,
                         illum_polarization, theory, constraints)
        self._use_parameters({'lens_angle':lens_angle})

    def forward(self, pars, detector):
        """
        Compute a forward model (the hologram)

        Parameters
        -----------
        pars: dict(string, float)
            Dictionary containing values for each parameter used to compute
            the hologram. Possible parameters are given by self.parameters.
        detector: xarray
            dimensions of the resulting hologram. Metadata taken from
            detector if not given explicitly when instantiating self.
        """
        optics_kwargs, scatterer = self._optics_scatterer(pars, detector)
        # We need the lens parameter(s) for the theory:
        theory_kwargs = {name:self.get_parameter(name, pars, detector)
                                    for name in self.theory_params}
        # FIXME would be nice to have access to the interpolator kwargs
        theory = MieLens(**theory_kwargs)
        try:
            return calc_holo(detector, scatterer, theory=theory,
                             scaling=1.0, **optics_kwargs)
        except InvalidScatterer:
            return -np.inf

# TODO:
# Make some unit tests for ExactModel, then for PerfectLensModel
# It would be nice if some of the unittests for fitting were also
# applicable to the inference models. This should be changed later,
# when the two fitting approaches are unified.
