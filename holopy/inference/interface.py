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

from holopy.scattering import Scatterer
from holopy.scattering.scatterer import _interpret_parameters
from holopy.inference.model import ALL_STRATEGIES, Model, ExactModel
from holopy.inference.prior import Uniform

available_fit_strategies = ALL_STRATEGIES['fit']
available_sampling_strategies = ALL_STRATEGIES['sample']


def sample(data, model):
    if isinstance(model, Model):
        return model.sample(data)
    else:
        msg = "Sampling model {} is not a HoloPy Model object.".format(model)
        raise ValueError(msg)

def fit(data, model, parameters = None):
    if isinstance(model, Scatterer):
        model = make_default_model(model, parameters)
    elif parameters is not None:
        warnings.warn("Ignoring parameters {} in favour of model {}.".format(
                      parameters, model), UserWarning)
    return model.fit(data)

def make_default_model(base_scatterer, fitting_parameters):
    if fitting_parameters is None:
        fitting_parameters = base_scatterer.parameters.keys()
    scatterer = parameterize_scatterer(base_scatterer, fitting_parameters)
    return ExactModel(scatterer, noise_sd = 1)

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
    
