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
"""
Results of sampling

.. moduleauthor:: Thomas G. Dimiduk <tom@dimiduk.net>
"""
from copy import copy
from warnings import warn

import yaml
import xarray as xr
import pandas as pd
import numpy as np


from holopy.core.metadata import detector_grid, copy_metadata
from holopy.core.holopy_object import HoloPyObject, FullLoader
from holopy.core.io.io import pack_attrs, unpack_attrs
from holopy.core.utils import dict_without, ensure_scalar
from holopy.core.errors import raise_fitting_api_error
from holopy.scattering.errors import MissingParameter


class FitResult(HoloPyObject):
    def __init__(self, data, model, strategy, time, kwargs={}):
        self.data = data
        self.model = model
        self.strategy = strategy
        if hasattr(strategy, 'parallel') and hasattr(strategy.parallel, 'map'):
            self.strategy.parallel = 'external_pool'
        self.time = time
        self._kwargs_keys = []
        self.add_attr(kwargs)
        needs_intervals = not isinstance(self, SamplingResult)
        if needs_intervals and not hasattr(self, 'intervals'):
            raise MissingParameter('intervals')

    @property
    def _parameters(self):
        return [val.guess for val in self.intervals]

    @property
    def _names(self):
        return [val.name for val in self.intervals]

    @property
    def parameters(self):
        return {name: val for name, val in zip(self._names, self._parameters)}

    @property
    def guess_parameters(self):
        return {name: val.guess for name, val in self.model.parameters.items()}

    @property
    def scatterer(self):
        return self.model.scatterer_from_parameters(self._parameters)

    @property
    def guess_scatterer(self):
        return self.model.scatterer_from_parameters(self.model.initial_guess)

    @property
    def hologram(self):
        def calculation():
            return self.forward(self._parameters)
        return self._calculate_first_time("_hologram", calculation)

    @property
    def guess_hologram(self):
        def calculation():
            return self.forward(self.model.initial_guess)
        return self._calculate_first_time("_guess_hologram", calculation)

    @property
    def max_lnprob(self):
        def calculation():
            return self.model.lnposterior(self._parameters, self.data)
        return self._calculate_first_time("_max_lnprob", calculation)

    def _calculate_first_time(self, attr_name, long_calculation):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, long_calculation())
            self._kwargs_keys.append(attr_name)
        return getattr(self, attr_name)

    def add_attr(self, kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)
            self._kwargs_keys.append(key)

    def forward(self, pars):
        if hasattr(self.data, 'original_dims'):
            # dealing with subset data
            original_dims = self.data.original_dims
            # can't currently handle non-0 values of z, as in detector_grid
            x = original_dims['x']
            y = original_dims['y']
            shape = (len(x), len(y))
            spacing = (np.diff(x)[0], np.diff(y)[0])
            extra_dims = dict_without(original_dims, ['x', 'y', 'z'])
            schema = detector_grid(shape, spacing, extra_dims=extra_dims)
            schema = copy_metadata(self.data, schema, do_coords=False)
            schema['x'] = x
            schema['y'] = y
        else:
            schema = self.data
        return self.model.forward(pars, schema)

    @property
    def _source_class(self):
        return "holopy.inference.{}".format(self.__class__.__name__)

    def _serialize_as_dataset(self):
        dataset = xr.Dataset({'data': self.data})
        if 'flat' in dataset:
            dataset.data.attrs['_flat'] = [
                list(f) for f in dataset.data.flat.values]
            dataset = dataset.rename({'flat': 'point'})
            new_coords = {'point': np.arange(len(dataset.point))}
            dataset = dataset.assign_coords(new_coords)
        dataset.data.attrs = pack_attrs(dataset.data)
        attrs = ['model', 'strategy', 'time', '_source_class']

        def make_yaml(key):
            attr = getattr(self, key)
            if isinstance(attr, HoloPyObject):
                attr = yaml.dump(attr, default_flow_style=True)
            return str(attr)

        attrs = {str(key): make_yaml(key) for key in attrs}
        xr_kw = {}
        yaml_kw = {}
        for key in self._kwargs_keys:
            attr = getattr(self, key)
            kwdict = xr_kw if isinstance(attr, xr.DataArray) else yaml_kw
            kwdict[key] = copy(attr)
        attrs['_kwargs'] = yaml.dump(yaml_kw, default_flow_style=True)
        for key, val in xr_kw.items():
            xr_kw[key].attrs = pack_attrs(val)
        dataset = xr.merge([dataset, xr_kw])
        dataset.attrs = attrs
        return dataset

    def _save(self, filename, **kwargs):
        dataset = self._serialize_as_dataset()
        dataset.to_netcdf(filename, engine='h5netcdf', **kwargs)

    def best_fit(self):
        # this method is published in the HoloPy paper so it needs
        # an informative error message:
        raise_fitting_api_error(
            'FitResult.hologram', 'FitResult.best_fit()')

    @classmethod
    def _unserialize(cls, dataset):
        data = dataset.data
        data.attrs = unpack_attrs(data.attrs)
        if '_flat' in data.attrs.keys():
            flats = np.array(data.attrs['_flat']).T
            levels = [data.original_dims[key] for key in ['x', 'y', 'z']]
            codes = [[level.index(f) for f in flat]
                     for level, flat in zip(levels, flats)]
            flat_index = pd.MultiIndex(levels, codes, names=['x', 'y', 'z'])
            coordnames = list(data.coords)
            coordnames.remove('point')
            coords = {coord: data[coord] for coord in coordnames}
            coords['flat'] = flat_index
            data = xr.DataArray(data.values, dims=coordnames + ['flat'],
                                coords=coords, attrs=data.attrs)
        model = yaml.load(dataset.attrs['model'], Loader=FullLoader)
        strategy = yaml.load(dataset.attrs['strategy'], Loader=FullLoader)
        outlist = [data, model, strategy]
        outlist.append(yaml.safe_load(dataset.attrs['time']))
        kwargs = yaml.safe_load(dataset.attrs['_kwargs'])

        for key in ['lnprobs', 'samples', '_best_fit']:
            try:
                kwargs[key] = getattr(dataset, key)
                kwargs[key].attrs = unpack_attrs(kwargs[key].attrs)
            except AttributeError:
                pass
        outlist.append(kwargs)
        return outlist

    @classmethod
    def _load(cls, ds, **kwargs):
        with xr.open_dataset(ds, engine='h5netcdf', **kwargs) as ds:
            args = cls._unserialize(ds.load())
        return cls(*args)


class SamplingResult(FitResult):
    def __init__(self, data, model, strategy, time, kwargs={}):
        super().__init__(data, model, strategy, time, kwargs)
        if not hasattr(self, 'intervals'):
            self.intervals = self._calc_intervals()

    def _calc_intervals(self):
        P_LOW = 15.865525393145708  # 100*(1-scipy.special.erf(1/np.sqrt(2)))/2
        map_val = self.samples[np.unravel_index(np.argmax(self.lnprobs.data), self.lnprobs.shape)]
        minus = map_val - self.samples.reduce(
            np.percentile, q=P_LOW, dim=['walker', 'chain'])
        plus = -map_val + self.samples.reduce(
            np.percentile, q=(100 - P_LOW), dim=['walker', 'chain'])

        def make_uncertain_value(p):
            return UncertainValue(
                map_val.loc[[p]], plus.loc[[p]], minus.loc[[p]], p)

        return [make_uncertain_value(p) for p in self.samples.parameter.values]

    def burn_in(self, sample_number):

        def cut_start(array):
            return array.isel(chain=slice(sample_number, None))

        burned_in = copy(self)
        burned_in.samples = cut_start(burned_in.samples)
        burned_in.lnprobs = cut_start(burned_in.lnprobs)
        burned_in.intervals = burned_in._calc_intervals()
        return burned_in


GROUPNAME = 'stage_results[{}]'


class TemperedSamplingResult(SamplingResult):
    def __init__(self, end_result, stage_results, strategy, time):
        kwargs = {'lnprobs': end_result.lnprobs, 'samples': end_result.samples}
        super().__init__(end_result.data, end_result.model, strategy, time,
                         kwargs)
        self.stage_results = stage_results

    def _save(self, filename):
        for i, ds in enumerate(self.stage_results):
            ds._save(filename, group=GROUPNAME.format(i), mode='a')
        super()._save(filename, mode='a')

    @classmethod
    def _load(cls, filename):
        ds = SamplingResult._load(filename)
        stages = [SamplingResult._load(filename, group=GROUPNAME.format(i))
                  for i in range(len(ds.strategy.stage_strategies) - 1)]
        return cls(ds, stages, ds.strategy, ds.time)


class UncertainValue(HoloPyObject):
    """
    Represent an uncertain value

    Parameters
    ----------
    value: float
        The value
    plus: float
        The plus n_sigma uncertainty (or the uncertainty if it is symmetric)
    minus: float or None
        The minus n_sigma uncertainty, or None if the uncertainty is symmetric
    n_sigma: int (or float)
        The number of sigma the uncertainties represent
    """
    def __init__(self, guess, plus, minus=None, name=None):
        self.guess = ensure_scalar(guess)
        self.plus = ensure_scalar(plus)
        if minus is None:
            self.minus = self.plus
        else:
            self.minus = ensure_scalar(minus)
        self.name = name

    def _repr_latex_(self):
        from IPython.display import Math
        confidence = ""
        if self.n_sigma != 1:
            confidence = " (\mathrm{{{}\ sigma}})".format(self.n_sigma)
        display_precision = int(
            round(np.log10(self.guess/(min(self.plus, self.minus))) + .6))
        guess_fmt = "{{:.{}g}}".format(max(display_precision, 2))
        guess = guess_fmt.format(self.guess)
        return "${guess}^{{+{s.plus:.2g}}}_{{-{s.minus:.2g}}}{conf}$".format(
            s=self, conf=confidence, guess=guess)
