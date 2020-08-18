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
from holopy.inference.prior import Prior, Uniform, ComplexPrior, generate_guess


def make_xarray(dim_name, keys, values):
    if isinstance(values[0], xr.DataArray):
        new_dim = xr.DataArray(keys, dims=[dim_name], name=dim_name)
        return xr.concat(values, dim=new_dim)
    else:
        return xr.DataArray(np.array(values), coords=[keys], dims=dim_name)


def read_map(map_entry, parameter_values):
    if isinstance(map_entry, str) and map_entry[:11] == '_parameter_':
        return parameter_values[int(map_entry[11:])]
    elif isinstance(map_entry, list):
        if len(map_entry) == 2 and callable(map_entry[0]):
            func, args = map_entry
            return func(*[read_map(arg, parameter_values) for arg in args])
        else:
            return [read_map(item, parameter_values) for item in map_entry]
    else:
        return map_entry


def edit_map_indices(map_entry, indices):
    if isinstance(map_entry, list):
        return [edit_map_indices(item, indices) for item in map_entry]
    elif isinstance(map_entry, str) and map_entry[:11] == '_parameter_':
        old_index = int(map_entry.split("_")[-1])
        if old_index in indices:
            new_index = indices[0]
        elif old_index < indices[0]:
            new_index = old_index
        else:
            shift = (np.array(indices) < old_index).sum() - 1
            new_index = old_index - shift
        return '_parameter_{}'.format(new_index)
    else:
        return map_entry


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
        self._parameter_names = []
        self._scatterer_map = self._convert_to_map(scatterer.parameters)
        self._parameters = [parameter.renamed(name) for parameter, name
            in zip(self._parameters, self._parameter_names)]
        del self._parameter_names
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
        """
        dictionary of the model's parameters
        """
        return {par.name: par for par in self._parameters}

    @property
    def initial_guess(self):
        """
        dictionary of initial guess value for each parameter
        """
        return {par.name: par.guess for par in self._parameters}

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

    def _convert_to_map(self, parameter, name=''):
        if isinstance(parameter, (list, tuple, np.ndarray)):
            mapped = self._iterate_mapping(name + '.', enumerate(parameter))
        elif isinstance(parameter, dict):
            mapped = self._map_dictionary(parameter, name)
        elif isinstance(parameter, xr.DataArray):
            mapped = self._map_xarray(parameter, name)
        elif isinstance(parameter, ComplexPrior):
            mapped = self._map_complex(parameter, name)
        elif isinstance(parameter, Prior):
            index = self._check_for_ties(parameter, name)
            mapped = '_parameter_{}'.format(index)
        else:
            mapped = parameter
        return mapped
    # still need to reconcile with use_parameters

    def _iterate_mapping(self, prefix, pairs):
        return [self._convert_to_map(parameter, prefix + str(suffix))
                for suffix, parameter in pairs]

    def _map_dictionary(self, parameter, name):
        mapping = map(lambda x: x[::-1], parameter.items())
        prefix = name + "." if len(name) > 0 else ""
        values_map = self._iterate_mapping(prefix, parameter.items())
        iterator = zip(parameter.keys(), values_map)
        return [dict, [[[key, val] for key, val in iterator]]]

    def _map_xarray(self, parameter, name):
        dim_name = parameter.dims[0]
        coord_keys = list(parameter.coords[dim_name].values)
        if len(parameter.dims) == 1:
            values = parameter.values
        else:
            values = [parameter.loc[{dim_name: key}] for key in coord_keys]
        values_map = self._iterate_mapping(name + '.', zip(coord_keys, values))
        return [make_xarray, [dim_name, coord_keys, values_map]]

    def _map_complex(self, parameter, name):
        mapping = ((key, getattr(parameter, key)) for key in ['real', 'imag'])
        return [complex, self._iterate_mapping(name + '.', mapping)]

    def _check_for_ties(self, parameter, name):
        tied = False
        for index, existing in enumerate(self._parameters):
            # can't simply check parameter in self._parameters because
            # then two priors defined separately, but identically will
            # match whereas this way they are counted as separate objects.
            if existing is parameter:
                tied = True
                shared_name = self._parameter_names[index].split(':', 1)[-1]
                if shared_name not in self._parameter_names:
                    self._parameter_names[index] = shared_name
                break
        if not tied:
            index = len(self._parameters)
            self._parameters.append(parameter)
            if parameter.name is not None:
                name = parameter.name
            if name in self._parameter_names:
                name += '_0'
            while name in self._parameter_names:
                counter, reversename = name[::-1].split("_", 1)
                name = reversename[::-1] + "_" + str(int(counter[::-1]) + 1)
            self._parameter_names.append(name)
        return index

    def add_tie(self, parameters_to_tie, new_name=None):
        """
        Defines new ties between model parameters

        Parameters
        ----------
        parameters_to_tie: listlike
            names of parameters to tie, as given by keys in model.parameters
        new_name: string, optional
            the name for the new tied parameter
        """
        indices = []
        parameter_names = [par.name for par in self._parameters]
        for par in parameters_to_tie:
            if par not in parameter_names:
                msg = ("Cannot tie parameter {}. It is not present in "
                       "parameters {}").format(par, parameter_names)
                raise ValueError(msg)
            first_value = self.parameters[parameters_to_tie[0]].renamed(None)
            if not self.parameters[par].renamed(None) == first_value:
                msg = "Cannot tie unequal parameters {} and {}".format(
                        par, parameters_to_tie[0])
                raise ValueError(msg)
            indices.append(parameter_names.index(par))
        indices.sort()
        for index in indices[:0:-1]:
            del(self._parameters[index])
        if new_name is not None:
            self._parameters[indices[0]].name = new_name
        self._scatterer_map = edit_map_indices(self._scatterer_map, indices)
        # need to do any other maps here besides scatterer

    def scatterer_from_parameters(self, pars):
        """
        Creates a scatterer by setting values for model parameters

        Parameters
        ----------
        pars: dict
            values to create scatterer. Keys should match model.parameters

        Returns
        -------
        scatterer
        """
        pars = [pars[par.name] for par in self._parameters]
        scatterer_parameters = read_map(self._scatterer_map, pars)
        return self.scatterer.from_parameters(scatterer_parameters)

    def _optics_scatterer(self, pars, schema):
        optics_keys = ['medium_index', 'illum_wavelen', 'illum_polarization']
        optics = {key: self._get_parameter(key, pars, schema)
                  for key in optics_keys}
        scatterer = self.scatterer_from_parameters(pars)
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
                par_scat = self.scatterer_from_parameters(par_vals)
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

