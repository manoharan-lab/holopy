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

from holopy.core.metadata import dict_to_array, make_subset_data
from holopy.core.utils import ensure_array, ensure_listlike, ensure_scalar
from holopy.core.holopy_object import HoloPyObject
from holopy.scattering.errors import (MultisphereFailure, TmatrixFailure,
                                      InvalidScatterer, MissingParameter)
from holopy.scattering.interface import calc_holo
from holopy.scattering.theory import MieLens
from holopy.scattering.scatterer import (_expand_parameters,
                                         _interpret_parameters)
from holopy.inference.prior import Prior, Uniform, generate_guess


class Model(HoloPyObject):
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
        if not (np.isscalar(noise_sd) or isinstance(noise_sd, (Prior, dict))):
            noise_sd = ensure_array(noise_sd)
        parameters_to_use = {
            'medium_index': medium_index,
            'illum_wavelen': illum_wavelen,
            'illum_polarization': illum_polarization,
            'theory': theory,
            'noise_sd': noise_sd,
            }
        self._check_parameters_are_not_xarray(parameters_to_use)
        self._use_parameters(parameters_to_use)

    @property
    def parameters(self):
        return {par.name: par for par in self._parameters}

    def _get_parameter(self, name, pars, schema=None):
        interpreted_pars = _interpret_parameters(pars)
        if name in pars.keys():
            return pars[name]
        elif name in interpreted_pars.keys():
            return interpreted_pars[name]
        elif hasattr(self, name):
            return getattr(self, name)
        try:
            return getattr(schema, name)
        except:
            raise MissingParameter(name)

    def _use_parameters(self, parameters, as_attr=True):
        if as_attr:
            for name, par in parameters.items():
                if par is not None:
                    setattr(self, name, par)
        parameters = dict(_expand_parameters(parameters.items()))
        for key, val in parameters.items():
            if isinstance(val, Prior):
                self._parameters.append(copy(val))
                self._parameters[-1].name = key

    def _optics_scatterer(self, pars, schema):
        optics_keys = ['medium_index', 'illum_wavelen', 'illum_polarization']
        optics = {key: self._get_parameter(key, pars, schema)
                  for key in optics_keys}
        scatterer = self.scatterer.from_parameters(pars)
        return optics, scatterer

    def generate_guess(self, n=1, scaling=1, seed=None):
        return generate_guess(self._parameters, n, scaling, seed)

    def lnprior(self, par_vals):
        """
        Compute the log-prior probability of par_vals

        Parameters
        -----------
        par_vals: dict(string, float)
            Dictionary containing values for each parameter
        Returns
        -------
        lnprior: float
        """
        if hasattr(self, 'scatterer'):
            try:
                par_scat = self.scatterer.from_parameters(par_vals)
            except InvalidScatterer:
                return -np.inf

        for constraint in self.constraints:
            if not constraint.check(par_scat):
                return -np.inf

        return sum([p.lnprob(par_vals[p.name]) for p in self._parameters])

    def lnposterior(self, par_vals, data, pixels=None):
        """
        Compute the log-posterior probability of pars given data

        Parameters
        -----------
        pars: dict(string, float)
            Dictionary containing values for each parameter
        data: xarray
            The data to compute posterior against
        pixels: int(optional)
            Specify to use a random subset of all pixels in data

        Returns
        --------
        lnposterior: float
        """
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

    def _find_noise(self, pars, data):
        noise = dict_to_array(
            data, self._get_parameter('noise_sd', pars, data))
        if noise is None:
            if np.all([isinstance(par, Uniform) for par in self._parameters]):
                noise = 1
            else:
                raise MissingParameter('noise_sd for non-uniform priors')
        return noise

    def _residuals(self, pars, data, noise):
        forward_model = self.forward(pars, data)
        return ((forward_model - data) / noise).values

    def lnlike(self, pars, data):
        """
        Compute the log-likelihood for pars given data

        Parameters
        -----------
        pars: dict(string, float)
            Dictionary containing values for each parameter
        data: xarray
            The data to compute likelihood against

        Returns
        --------
        lnlike: float
        """
        noise_sd = self._find_noise(pars, data)
        N = data.size
        log_likelihood = ensure_scalar(
            -N/2 * np.log(2 * np.pi) -
            N * np.mean(np.log(ensure_array(noise_sd))) -
            0.5 * (self._residuals(pars, data, noise_sd)**2).sum())
        return log_likelihood

    def fit(self, data, strategy=None):
        from holopy.fitting import fit_warning
        from holopy.inference.interface import validate_strategy
        fit_warning('holopy.fit()', 'model.fit()')
        strategy = validate_strategy(strategy, 'fit')
        return strategy.fit(self, data)

    def sample(self, data, strategy=None):
        from holopy.fitting import fit_warning
        from holopy.inference.interface import validate_strategy
        fit_warning('holopy.sample()', 'model.sample()')
        strategy = validate_strategy(strategy, 'sample')
        return strategy.sample(self, data)

    def _check_parameters_are_not_xarray(self, parameters_to_use):
        for key, value in parameters_to_use.items():
            if isinstance(value, xr.DataArray):
                msg = ("{} cannot be an xarray.DataArray due to ".format(key) +
                       "limitations in holopys ability to save objects.")
                raise ValueError(msg)


class LimitOverlaps(HoloPyObject):
    """
    Constraint prohibiting overlaps beyond a certain tolerance.
    fraction is the largest overlap allowed, in terms of sphere diameter.

    """
    def __init__(self, fraction=.1):
        self.fraction = fraction

    def check(self, s):
        return s.largest_overlap() <= ((np.min(s.r) * 2) * self.fraction)


class AlphaModel(Model):
    """
    Model of hologram image formation with scaling parameter alpha.
    """
    def __init__(self, scatterer, alpha=1, noise_sd=None, medium_index=None,
                 illum_wavelen=None, illum_polarization=None, theory='auto',
                 constraints=[]):
        super().__init__(scatterer, noise_sd, medium_index, illum_wavelen,
                         illum_polarization, theory, constraints)
        additional_parameters_to_use = {'alpha': alpha}
        self._use_parameters(additional_parameters_to_use)
        self._check_parameters_are_not_xarray(additional_parameters_to_use)

    def forward(self, pars, detector):
        """
        Compute a hologram from pars with dimensions and metadata of detector,
        scaled by self.alpha.

        Parameters
        -----------
        pars: dict(string, float)
            Dictionary containing values for each parameter used to compute
            the hologram. Possible parameters are given by self.parameters.
        detector: xarray
            dimensions of the resulting hologram. Metadata taken from
            detector if not given explicitly when instantiating self.
        """
        alpha = self._get_parameter('alpha', pars, detector)
        optics, scatterer = self._optics_scatterer(pars, detector)
        try:
            return calc_holo(detector, scatterer, theory=self.theory,
                             scaling=alpha, **optics)
        except (MultisphereFailure, TmatrixFailure, InvalidScatterer):
            return -np.inf


# TODO: Change the default theory (when it is "auto") to be
# selected by the model.
# -- this is a little trickier than it sounds, because
# hlopy.scattering.determine_theory picks based off of whether the
# object is 1 sphere or a collection of spheres etc. So you can't
# pass MieLens as a theory
# For now it would be OK since PerfectLensModel only works with single
# spheres or superpositions, but let's leave this for later.
class ExactModel(Model):
    """
    Model of arbitrary scattering function given by calc_func.
    """
    def __init__(self, scatterer, calc_func=calc_holo, noise_sd=None,
                 medium_index=None, illum_wavelen=None,
                 illum_polarization=None, theory='auto', constraints=[]):
        super().__init__(scatterer, noise_sd, medium_index, illum_wavelen,
                         illum_polarization, theory, constraints)
        self.calc_func = calc_func

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
            return self.calc_func(detector, scatterer, theory=self.theory, **optics)
        except (MultisphereFailure, InvalidScatterer):
            return -np.inf


class PerfectLensModel(Model):
    """
    Model of hologram image formation through a high-NA objective.
    """
    theory_params = ['lens_angle']

    def __init__(self, scatterer, alpha=1.0, lens_angle=1.0, noise_sd=None,
                 medium_index=None, illum_wavelen=None, theory='auto',
                 illum_polarization=None, constraints=[]):
        super().__init__(scatterer, noise_sd, medium_index, illum_wavelen,
                         illum_polarization, theory, constraints)
        additional_parameters_to_use = {
            'alpha': alpha, 'lens_angle': lens_angle}
        self._use_parameters(additional_parameters_to_use)
        self._check_parameters_are_not_xarray(additional_parameters_to_use)

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
        alpha = self._get_parameter('alpha', pars, detector)
        theory_kwargs = {name: self._get_parameter(name, pars, detector)
                         for name in self.theory_params}
        # FIXME would be nice to have access to the interpolator kwargs
        theory = MieLens(**theory_kwargs)
        try:
            return calc_holo(detector, scatterer, theory=theory,
                             scaling=alpha, **optics_kwargs)
        except InvalidScatterer:
            return -np.inf

