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


import numpy as np
import xarray as xr

from holopy.core.holopy_object import HoloPyObject
from holopy.core.prior import Prior, TransformedPrior


def make_xarray(dim_name, keys, values):
    '''
    Packs values into xarray with new dim and coords (keys)
    '''
    if isinstance(values[0], xr.DataArray):
        new_dim = xr.DataArray(keys, dims=[dim_name], name=dim_name)
        return xr.concat(values, dim=new_dim)
    else:
        return xr.DataArray(np.array(values), coords=[keys], dims=dim_name)


def transformed_prior(transformation, base_priors):
    if any([isinstance(bp, Prior) for bp in base_priors]):
        return TransformedPrior(transformation, base_priors)
    else:
        return transformation(*base_priors)


def read_map(map_entry, parameter_values):
    '''
    Reads a map to create an object

    Parameters
    ----------
    map_entry:
        map or subset of map created by model methods
    parameter_values: listlike
        values to replace map placeholders in final object
    '''
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
    '''
    Adjusts a map to account for ties between parameters

    Parameters
    ----------
    map_entry:
        map or subset of map created by model methods
    indices: listlike
        indices of parameters to be tied
    '''
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


class Mapper(HoloPyObject):
    '''
    Creates "maps" from objects containing priors that retain their
    hierarchical structure (including ties) but are easily serializable. The
    main entry point is through `convert_to_map`, which returns a map of the
    object and also updates the Mapper `parameter` and `parameter_names`
    attributes so they can be extracted for later use.
    '''
    def __init__(self):
        self.parameters = []
        self.parameter_names = []

    def convert_to_map(self, parameter, name=''):
        if isinstance(parameter, (list, tuple, np.ndarray)):
            mapped = self.iterate_mapping(name + '.', enumerate(parameter))
        elif isinstance(parameter, dict):
            mapped = self.map_dictionary(parameter, name)
        elif isinstance(parameter, xr.DataArray):
            mapped = self.map_xarray(parameter, name)
        elif isinstance(parameter, TransformedPrior):
            mapped = self.map_transformed_prior(parameter, name)
        elif isinstance(parameter, Prior):
            index = self.get_parameter_index(parameter, name)
            mapped = '_parameter_{}'.format(index)
        else:
            mapped = parameter
        return mapped

    def iterate_mapping(self, prefix, pairs):
        return [self.convert_to_map(parameter, prefix + str(suffix))
                for suffix, parameter in pairs]

    def map_dictionary(self, parameter, name):
        prefix = name + "." if len(name) > 0 else ""
        values_map = self.iterate_mapping(prefix, parameter.items())
        iterator = zip(parameter.keys(), values_map)
        dict_args = [[key, val] for key, val in iterator if val is not None]
        return [dict, [dict_args]]

    def map_xarray(self, parameter, name):
        dim_name = parameter.dims[0]
        coord_keys = parameter.coords[dim_name].values.tolist()
        if len(parameter.dims) == 1:
            values = parameter.values
        else:
            values = [parameter.loc[{dim_name: key}] for key in coord_keys]
        values_map = self.iterate_mapping(name + '.', zip(coord_keys, values))
        return [make_xarray, [dim_name, coord_keys, values_map]]

    def map_transformed_prior(self, parameter, name):
        name = name if parameter.name is None else parameter.name
        name = name + '.' if len(parameter.base_prior) > 1 else name
        mapped_priors = self.iterate_mapping(name, parameter.map_keys)
        return [transformed_prior, [parameter.transformation, mapped_priors]]

    def get_parameter_index(self, parameter, name):
        index = self.check_for_ties(parameter)
        if index is None:
            index = len(self.parameters)
            self.add_parameter(parameter, name)
        else:
            shared_name = self.parameter_names[index].split(':', 1)[-1]
            really_shared = name.split(':', 1)[-1] == shared_name
            if really_shared and shared_name not in self.parameter_names:
                self.parameter_names[index] = shared_name
        return index

    def check_for_ties(self, parameter):
        for index, existing in enumerate(self.parameters):
            # can't simply check parameter in self.parameters because
            # then two priors defined separately, but identically will
            # match whereas this way they are counted as separate objects.
            if existing is parameter:
                return index

    def add_parameter(self, parameter, name):
        self.parameters.append(parameter)
        if parameter.name is not None:
            name = parameter.name
        if name in self.parameter_names:
            name += '_0'
        while name in self.parameter_names:
            counter, reversename = name[::-1].split("_", 1)
            name = reversename[::-1] + "_" + str(int(counter[::-1]) + 1)
        self.parameter_names.append(name)
