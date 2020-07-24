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
from holopy.scattering import Scatterer
from holopy.scattering.scatterer import _interpret_parameters
from holopy.inference.model import Model, AlphaModel
from holopy.inference.prior import Uniform
from holopy.inference.nmpfit import NmpfitStrategy
from holopy.inference.scipyfit import LeastSquaresScipyStrategy
from holopy.inference.cmaes import CmaStrategy
from holopy.inference.emcee import EmceeStrategy, TemperedStrategy

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


def make_default_model(base_scatterer, fitting_parameters):
    if fitting_parameters is None:
        fitting_parameters = base_scatterer.parameters.keys()
    scatterer = parameterize_scatterer(base_scatterer, fitting_parameters)
    alpha_prior = Uniform(0, 1, guess=0.7, name='alpha')
    return AlphaModel(scatterer, noise_sd=1, alpha=alpha_prior)


def parameterize_scatterer(base_scatterer, fitting_parameters):
    parameters = base_scatterer.guess.parameters
    variable_parameters = {par_name: make_uniform(parameters, par_name)
                           for par_name in rename_xyz(fitting_parameters)}
    parameters.update(variable_parameters)
    return type(base_scatterer)(**_interpret_parameters(parameters, True))


def rename_xyz(parameters_list):
    for i, key in enumerate(['x', 'y', 'z']):
        new_key = 'center.{}'.format(i)
        if key in parameters_list:
            loc = parameters_list.index(key)
            parameters_list[loc] = new_key
    return parameters_list


def make_uniform(guesses, key):
    try:
        guess_value = guesses[key]
    except KeyError:
        msg = 'Parameter {} not found in scatterer parameters.'.format(key)
        raise ValueError(msg)
    minval = 0 if key in ['n', 'r'] else -np.inf
    return Uniform(minval, np.inf, guess_value, key)
