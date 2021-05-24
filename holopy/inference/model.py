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

import warnings

import numpy as np

from holopy.core.metadata import make_subset_data
from holopy.core.utils import ensure_array, ensure_listlike, ensure_scalar
from holopy.core.holopy_object import HoloPyObject
from holopy.core.errors import raise_fitting_api_error
from holopy.scattering.errors import (MultisphereFailure, TmatrixFailure,
                                      InvalidScatterer, MissingParameter)
from holopy.scattering.interface import calc_holo, interpret_theory
from holopy.inference import prior
from holopy.core.mapping import Mapper, read_map, edit_map_indices


OPTICS_KEYS = ['medium_index', 'illum_wavelen',
               'illum_polarization', 'noise_sd']


class Model(HoloPyObject):
    """Model probabilites of observing data

    Compute probabilities that observed data could be explained by a set of
    scatterer and observation parameters.
    """
    _model_parameters = {}

    def __init__(self, scatterer, noise_sd=None, medium_index=None,
                 illum_wavelen=None, illum_polarization=None, theory='auto',
                 constraints=[]):
        self._dummy_scatterer = self._create_dummy_scatterer(scatterer)
        self.theory = interpret_theory(self._dummy_scatterer, theory)
        self.constraints = ensure_listlike(constraints)
        if not (np.isscalar(noise_sd)
                or isinstance(noise_sd, (prior.Prior, dict))):
            noise_sd = ensure_array(noise_sd)
        optics = [medium_index, illum_wavelen, illum_polarization, noise_sd]
        optics_parameters = {key: val for key, val in zip(OPTICS_KEYS, optics)}
        mapper = Mapper()
        self._maps = {'scatterer': mapper.convert_to_map(scatterer.parameters),
                      'theory': mapper.convert_to_map(self.theory.parameters),
                      'optics': mapper.convert_to_map(optics_parameters),
                      'model': mapper.convert_to_map(self._model_parameters)}
        self._parameters = mapper.parameters
        self._parameter_names = mapper.parameter_names

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
        for par in parameters_to_tie:
            if par not in self._parameter_names:
                msg = ("Cannot tie parameter {}. It is not present in "
                       "parameters {}").format(par, self._parameter_names)
                raise ValueError(msg)
            first_value = self.parameters[parameters_to_tie[0]].renamed(None)
            if not self.parameters[par].renamed(None) == first_value:
                msg = "Cannot tie unequal parameters {} and {}".format(
                        par, parameters_to_tie[0])
                raise ValueError(msg)
            indices.append(self._parameter_names.index(par))
        indices.sort()
        for index in indices[:0:-1]:
            del(self._parameters[index])
            del(self._parameter_names[index])
        if new_name is not None:
            self._parameter_names[indices[0]] = new_name
        self._maps = {key: edit_map_indices(val, indices)
                      for key, val in self._maps.items()}

    def _iteritems(self):
        keys = ['_dummy_scatterer', 'theory', '_parameters',
                '_parameter_names', '_maps']
        for key in keys:
            item = getattr(self, key)
            if isinstance(item, np.ndarray) and item.ndim == 1:
                item = list(item)
            yield key, item

    @classmethod
    def from_yaml(cls, loader, node):
        fields = loader.construct_mapping(node, deep=True)
        parameters = fields['_parameters']
        maps = fields['_maps']

        dummy_scatterer = fields['_dummy_scatterer']
        scatterer_parameters = read_map(maps['scatterer'], parameters)
        scatterer = dummy_scatterer.from_parameters(scatterer_parameters)
        kwargs = {'scatterer': scatterer, 'theory': fields['theory']}
        for key in ['optics', 'model', 'theory']:
            kwargs.update(read_map(maps[key], parameters))
        model = cls(**kwargs)
        if model._parameters == parameters:
            model._parameter_names = fields['_parameter_names']
        else:
            msg = ("Detected inconsistencies when reloading Model. "
                   "It may differ from previously saved object")
            warnings.warn(msg, UserWarning)
        return model

    @property
    def parameters(self):
        """
        dictionary of the model's parameters
        """
        return {name: par for name, par in zip(self._parameter_names,
                                               self._parameters)}

    @property
    def initial_guess(self):
        """
        dictionary of initial guess values for each parameter
        """
        return {name: par.guess for name, par in zip(self._parameter_names,
                                                     self._parameters)}

    @property
    def initial_guess_scatterer(self):
        return self.scatterer_from_parameters(self.initial_guess)

    @property
    def medium_index(self):
        return self._find_optics(self._parameters, None)['medium_index']

    @property
    def illum_wavelen(self):
        return self._find_optics(self._parameters, None)['illum_wavelen']

    @property
    def illum_polarization(self):
        return self._find_optics(self._parameters, None)['illum_polarization']

    @property
    def noise_sd(self):
        return self._find_noise(self._parameters, None)

    @property
    def scatterer(self):
        return self._scatterer_from_parameters(self._parameters)

    def theory_from_parameters(self, pars):
        pars = self.ensure_parameters_are_listlike(pars)
        formatted = read_map(self._maps['theory'], pars)
        return self.theory.from_parameters(formatted)

    def scatterer_from_parameters(self, pars):
        """
        Creates a scatterer by setting values for model parameters

        Parameters
        ----------
        pars: dict or list
            list - values for each parameter in the order of self._parameters
            dict - keys should match self.parameters
        Returns
        -------
        scatterer
        """
        pars = self.ensure_parameters_are_listlike(pars)
        return self._scatterer_from_parameters(pars)

    def _scatterer_from_parameters(self, pars):
        """
        Internal function taking pars as a list only
        """
        scatterer_parameters = read_map(self._maps['scatterer'], pars)
        return self._dummy_scatterer.from_parameters(scatterer_parameters)

    @classmethod
    def _create_dummy_scatterer(cls, scatterer):
        # this assumes that scatterer parameters are 1D list-likes or scalars
        # if they are not, this method needs to be changed.
        dummy_parameters = dict()
        for key, value in scatterer.parameters.items():
            parameter_is_1d_group = hasattr(value, '__len__')
            if parameter_is_1d_group:
                dummy_parameters[key] = [0 for _ in value]
            else:
                dummy_parameters[key] = 0
        return scatterer.from_parameters(dummy_parameters)

    def ensure_parameters_are_listlike(self, pars):
        if isinstance(pars, dict):
            pars = [pars[name] for name in self._parameter_names]
        return pars

    def _find_optics(self, pars, schema):
        """
        Creates optics dictionary by setting values for model parameters

        Parameters
        ----------
        pars: list
            values to create optics. Order should match model._parameters
        """
        mapped_optics = read_map(self._maps['optics'], pars)

        def find_parameter(key):
            if key in mapped_optics and mapped_optics[key] is not None:
                val = mapped_optics[key]
            elif hasattr(schema, key) and getattr(schema, key) is not None:
                val = getattr(schema, key)
            else:
                raise MissingParameter(key)
            return val
        return {key: find_parameter(key) for key in OPTICS_KEYS[:-1]}

    def _find_noise(self, pars, schema):
        """
        finds appropriate noise_sd for residuals calculations

        Parameters
        ----------
        pars: list
            values to create noise_sd. Order should match model._parameters
        """
        optics_map = read_map(self._maps['optics'], pars)
        if 'noise_sd' in optics_map and optics_map['noise_sd'] is not None:
            val = optics_map['noise_sd']
        elif hasattr(schema, 'noise_sd'):
            val = schema.noise_sd
        else:
            raise MissingParameter('noise_sd')
        if val is None:
            if np.all([isinstance(par, prior.Uniform)
                       for par in self._parameters]):
                val = 1
            else:
                raise MissingParameter('noise_sd for non-uniform priors')
        return val

    def generate_guess(self, n=1, scaling=1, seed=None):
        return prior.generate_guess(self._parameters, n, scaling, seed)

    def lnprior(self, pars):
        """
        Compute the log-prior probability of pars

        Parameters
        -----------
        pars: dict or list
            list - values for each parameter in the order of self._parameters
            dict - keys should match self.parameters
        Returns
        -------
        lnprior: float
        """
        pars = self.ensure_parameters_are_listlike(pars)
        return self._lnprior(pars)

    def _lnprior(self, pars):
        """
        Internal function taking pars as a list only
        """
        if 'scatterer' in self._maps:
            try:
                par_scat = self._scatterer_from_parameters(pars)
            except InvalidScatterer:
                return -np.inf

        for constraint in self.constraints:
            if not constraint.check(par_scat):
                return -np.inf
        return sum([p.lnprob(val) for p, val in zip(self._parameters, pars)])

    def lnposterior(self, pars, data, pixels=None):
        """
        Compute the log-posterior probability of pars given data

        Parameters
        -----------
        pars: dict or list
            list - values for each parameter in the order of self._parameters
            dict - keys should match self.parameters
        data: xarray
            The data to compute posterior against
        pixels: int(optional)
            Specify to use a random subset of all pixels in data

        Returns
        --------
        lnposterior: float
        """
        pars = self.ensure_parameters_are_listlike(pars)
        return self._lnposterior(pars, data, pixels)

    def _lnposterior(self, pars, data, pixels=None):
        """
        Internal function taking pars as a list only
        """
        lnprior = self._lnprior(pars)
        # prior is sometimes used to forbid thing like negative radius
        # which will fail if you attempt to compute a hologram of, so
        # don't try to compute likelihood where the prior already
        # forbids you to be
        if lnprior == -np.inf:
            return lnprior
        else:
            if pixels is not None:
                data = make_subset_data(data, pixels=pixels)
            return lnprior + self._lnlike(pars, data)

    def forward(self, pars, detector):
        pars = self.ensure_parameters_are_listlike(pars)
        return self._forward(pars, detector)

    def _forward(self, pars, detector):
        raise NotImplementedError("Implement in subclass")

    def _residuals(self, pars, data, noise):
        forward_model = self._forward(pars, data)
        return ((forward_model - data) / noise).values

    def lnlike(self, pars, data):
        """
        Compute the log-likelihood for pars given data

        Parameters
        -----------
        pars: dict or list
            list - values for each parameter in the order of self._parameters
            dict - keys should match self.parameters
        data: xarray
            The data to compute likelihood against

        Returns
        --------
        lnlike: float
        """
        pars = self.ensure_parameters_are_listlike(pars)
        return self._lnlike(pars, data)

    def _lnlike(self, pars, data):
        """
        Internal function taking pars as a list only
        """
        noise_sd = self._find_noise(pars, data)
        N = data.size
        log_likelihood = ensure_scalar(
            -N/2 * np.log(2 * np.pi) -
            N * np.mean(np.log(ensure_array(noise_sd))) -
            0.5 * (self._residuals(pars, data, noise_sd)**2).sum())
        return log_likelihood

    def fit(self, data, strategy=None):
        raise_fitting_api_error('holopy.fit()', 'Model.fit()')

    def sample(self, data, strategy=None):
        raise_fitting_api_error('holopy.sample()', 'Model.sample()')


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
        self._model_parameters = {'alpha': alpha}
        super().__init__(scatterer, noise_sd, medium_index, illum_wavelen,
                         illum_polarization, theory, constraints)

    @property
    def alpha(self):
        return read_map(self._maps['model'], self._parameters)['alpha']

    def _forward(self, pars, detector):
        """
        Compute a hologram from pars with dimensions and metadata of detector,
        scaled by self.alpha.

        Parameters
        -----------
        pars: list
            Values for each parameter used to compute the hologram. Ordering
            is given by self._parameters
        detector: xarray
            dimensions of the resulting hologram. Metadata taken from
            detector if not given explicitly when instantiating self.
        """
        alpha = read_map(self._maps['model'], pars)['alpha']
        optics = self._find_optics(pars, detector)
        scatterer = self._scatterer_from_parameters(pars)
        theory = self.theory_from_parameters(pars)
        try:
            return calc_holo(detector, scatterer, theory=theory,
                             scaling=alpha, **optics)
        except (MultisphereFailure, TmatrixFailure, InvalidScatterer):
            return -np.inf


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

    def _forward(self, pars, detector):
        """
        Compute a forward model (the hologram)

        Parameters
        -----------
        pars: list
            Values for each parameter used to compute the hologram. Ordering
            is given by self._parameters
        detector: xarray
            dimensions of the resulting hologram. Metadata taken from
            detector if not given explicitly when instantiating self.
        """
        optics = self._find_optics(pars, detector)
        scatterer = self._scatterer_from_parameters(pars)
        theory = self.theory_from_parameters(pars)
        try:
            return self.calc_func(detector, scatterer, theory=theory, **optics)
        except (MultisphereFailure, InvalidScatterer):
            return -np.inf
