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

from holopy.core.holopy_object import SerializableMetaclass
from holopy.scattering import Scatterer, Scatterers
from holopy.inference.model import Model, AlphaModel
from holopy.inference import prior
from holopy.inference.nmpfit import NmpfitStrategy
from holopy.inference.scipyfit import LeastSquaresScipyStrategy
from holopy.inference.cmaes import CmaStrategy
from holopy.inference.emcee import EmceeStrategy, TemperedStrategy

COORD_KEYS = ['x', 'y', 'z']
DEFAULT_STRATEGY = {'fit': 'nmpfit', 'sample': 'emcee'}
ALL_STRATEGIES = {'fit': {'nmpfit': NmpfitStrategy,
                          'scipy lsq': LeastSquaresScipyStrategy,
                          'cma': CmaStrategy},
                  'sample': {'emcee': EmceeStrategy,
                             'subset tempering': TemperedStrategy,
                             'parallel tempering': NotImplemented}}

available_fit_strategies = ALL_STRATEGIES['fit']
available_sampling_strategies = ALL_STRATEGIES['sample']


def sample(data, model, strategy=None):
    if isinstance(model, Model):
        strategy = validate_strategy(strategy, 'sample')
        return strategy.sample(model, data)
    else:
        msg = "Sampling model {} is not a HoloPy Model object.".format(model)
        raise ValueError(msg)


def fit(data, model, parameters=None, strategy=None):
    if isinstance(model, Scatterer):
        model = make_default_model(model, parameters)
    elif parameters is not None:
        warnings.warn("Ignoring parameters {} in favour of model {}.".format(
                      parameters, model), UserWarning)
    strategy = validate_strategy(strategy, 'fit')
    return strategy.fit(model, data)


def validate_strategy(strategy, operation):
    if strategy is None:
        strategy = DEFAULT_STRATEGY[operation]
    if isinstance(strategy, str):
        strategy = ALL_STRATEGIES[operation][strategy]
    if not hasattr(strategy, operation):
        raise ValueError("Cannot {} with Strategy of type {}.".format(
            operation, type(strategy).__name__))
    if isinstance(strategy, SerializableMetaclass):
        strategy = strategy()
    return strategy


def make_default_model(base_scatterer, fitting_parameters=None):
    if fitting_parameters is None:
        fitting_parameters = base_scatterer.parameters.keys()
    scatterer = parameterize_scatterer(base_scatterer, fitting_parameters)
    alpha_prior = prior.Uniform(0.5, 1, name='alpha')
    return AlphaModel(scatterer, noise_sd=1, alpha=alpha_prior)


def parameterize_scatterer(base_scatterer, fitting_parameters):
    if isinstance(fitting_parameters, str):
        fitting_parameters = [fitting_parameters]
    parameters = base_scatterer.parameters
    variable_parameters = {par_name: make_uniform(parameters, par_name)
                           for par_name in fitting_parameters}
    parameters.update(variable_parameters)
    for i, key in enumerate(COORD_KEYS):
        if isinstance(base_scatterer, Scatterers):
            for j in range(len(base_scatterer.scatterers)):
                replace_center(parameters, "{}:{}".format(j, key))
        else:
            replace_center(parameters, key)
    return base_scatterer.from_parameters(parameters)


def replace_center(parameters, key):
    index = COORD_KEYS.index(key[-1])
    prefix = key[:-1]
    try:
        val = parameters.pop(key)
    except KeyError:
        pass
    else:
        parameters[prefix + 'center'][index] = val


def make_uniform(guesses, key):
    suffix = key.split(":")[-1]
    prefix = key[:-len(suffix)]
    if suffix in COORD_KEYS:
        guess_value = guesses[prefix + 'center'][COORD_KEYS.index(suffix)]
    else:
        try:
            guess_value = guesses[key]
        except KeyError:
            msg = 'Parameter {} not found in scatterer parameters.'.format(key)
            raise ValueError(msg)
    minval = 0 if suffix in ['n', 'r'] else -np.inf
    if isinstance(guess_value, (list, tuple, np.ndarray)):
        if suffix == 'center':
            subkeys = COORD_KEYS
        else:
            subkeys = [suffix + '.' + str(i) for i in range(len(guess_value))]
        return [prior.Uniform(minval, np.inf, guess_val, prefix + subkey)
                for guess_val, subkey in zip(guess_value, subkeys)]
    else:
        return prior.Uniform(minval, np.inf, guess_value, key)
