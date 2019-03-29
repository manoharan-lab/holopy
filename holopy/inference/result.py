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
from collections import OrderedDict
from warnings import warn

import yaml
import xarray as xr
import pandas as pd
import numpy as np
import scipy.special
import h5py

from holopy.core.metadata import detector_grid, copy_metadata
from holopy.core.holopy_object import HoloPyObject, FullLoader
from holopy.core.io.io import pack_attrs, unpack_attrs
from holopy.core.utils import dict_without, ensure_scalar
from holopy.scattering.errors import MissingParameter


warn_text = 'Loading a legacy (pre-3.3) HoloPy file. Please \
                            save a new copy to ensure future compatibility'
# anywhere warn_text appears, there's an if-statement that can be removed.
# it is also used once in hp.core.io.io.unpack_attrs

def get_strategy(strategy):
    try:
        return yaml.load(strategy, Loader=FullLoader)
    except:
        # old file
        warn(warn_text)
    index = strategy.find('pixel')
    if index > -1 and strategy[index-1] != 'n':
        strategy = strategy[:index] + 'n' + strategy[index:]
    index = strategy.find('sample')
    if index > -1:
        strategy = strategy[:index] + 'emcee' + strategy[index+6:]
    return yaml.load(strategy, Loader=FullLoader)

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
        if not (isinstance(self,SamplingResult) or hasattr(self,'intervals')):
            raise MissingParameter('intervals')

    @property
    def guess(self):
        return [val.guess for val in self.intervals]

    @property
    def _names(self):
        return [val.name for val in self.intervals]

    @property
    def parameters(self):
        return {name: val for name, val in zip(self._names, self.guess)}

    @property
    def scatterer(self):
        return self.model.scatterer.from_parameters(self.parameters)

    @property
    def best_fit(self):
        # calculate the first time it's called and then store it
        try:
            return self._best_fit
        except AttributeError:
            pass
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
        self._best_fit = self.model.forward(self.parameters, schema)
        self._kwargs_keys.append('_best_fit')
        return self.best_fit

    @property
    def max_lnprob(self):
        # calculate the first time it's called and then store it
        try:
            return self._max_lnprob
        except AttributeError:
            pass
        self._max_lnprob = self.model.lnposterior(self.parameters, self.data)
        self._kwargs_keys.append('_max_lnprob')
        return self.max_lnprob

    def add_attr(self, kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)
            self._kwargs_keys.append(key)

    @property
    def _source_class(self):
        return "holopy.inference.{}".format(self.__class__.__name__)

    def _serialization_ds(self):
        ds = xr.Dataset({'data':self.data})
        if 'flat' in ds:
            ds.data.attrs['_flat'] = [list(f) for f in ds.data.flat.values]
            ds = ds.rename({'flat': 'point'})
            ds['point'].values = np.arange(len(ds.point))
        ds.data.attrs = pack_attrs(ds.data)
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
        ds = xr.merge([ds, xr_kw])
        ds.attrs = attrs
        return ds

    def _save(self, filename, **kwargs):
        ds = self._serialization_ds()
        ds.to_netcdf(filename, engine='h5netcdf', **kwargs)

    @classmethod
    def _unserialize(cls, ds):
        data = ds.data
        data.attrs = unpack_attrs(data.attrs)
        if '_flat' in data.attrs.keys():
            flats = np.array(data.attrs['_flat']).T
            levels = [data.original_dims[key] for key in ['x','y','z']]
            codes = [[level.index(f) for f in flat]
                                for level, flat in zip(levels, flats)]
            flat_index = pd.MultiIndex(levels, codes, names=['x','y','z'])
            coordnames = list(data.coords).remove('point')
            coords = {coord: data[coord] for coord in coordnames}
            coords['flat'] = flat_index
            data = xr.DataArray(data.values, dims=coordnames + ['flat'],
                                        coords=coords, attrs=data.attrs)
        model = yaml.load(ds.attrs['model'], Loader=FullLoader)
        strategy = get_strategy(ds.attrs['strategy'])
        outlist = [data, model, strategy]
        try:
            outlist.append(yaml.safe_load(ds.attrs['time']))
        except KeyError:
            outlist.append(None)
            warn(warn_text)
        try:
            kwargs = yaml.safe_load(ds.attrs['_kwargs'])
        except KeyError:
            warn(warn_text)
            kwargs = {}
        try:
            kwargs['intervals'] = yaml.load(ds.attrs['intervals'],
                                            Loader=FullLoader)
            warn(warn_text)
        except:
            pass
        for key in ['lnprobs', 'samples', '_best_fit']:
            try:
                kwargs[key] = getattr(ds, key)
                try:
                    kwargs[key].attrs = unpack_attrs(kwargs[key].attrs)
                except KeyError:
                    warn(warn_text)
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
        self.intervals = self._calc_intervals()

    def _calc_intervals(self):
        P_LOW = 100 * 0.158655253931457 # (1-scipy.special.erf(1/np.sqrt(2)))/2
        map_val = self.samples[np.unravel_index(
                                    self.lnprobs.argmax(), self.lnprobs.shape)]
        minus = map_val - self.samples.reduce(np.percentile, q=P_LOW,
                                                    dim=['walker', 'chain'])
        plus = -map_val + self.samples.reduce(np.percentile, q=(100 - P_LOW),
                                                    dim=['walker', 'chain'])
        return [UncertainValue(map_val.loc[p], plus.loc[p], minus.loc[p], p)
                                        for p in self.samples.parameter.values]

    def burn_in(self, sample_number):

        def cut_start(array):
            if len(array.chain.coords)==0:
                array['chain'] = ('chain', array.chain)
                array.set_index(chain='chain')
            return array.sel(chain = slice(sample_number, None))

        burned_in = copy(self)
        burned_in.samples = cut_start(burned_in.samples)
        burned_in.lnprobs = cut_start(burned_in.lnprobs)
        burned_in.intervals = burned_in._calc_intervals()
        return burned_in

    # deprecated methods as of 3.3
    def MAP(self):
        from holopy.fitting import fit_warning
        fit_warning('SamplingResult.guess', 'SamplingResult.MAP')
        return self.guess

    def values(self):
        from holopy.fitting import fit_warning
        fit_warning('SamplingResult.intervals', 'SamplingResult.values')
        return self.intervals


GROUPNAME = 'stage_results[{}]'
class TemperedSamplingResult(SamplingResult):
    def __init__(self, end_result, stage_results, strategy, time):
        kwargs = {'lnprobs': end_result.lnprobs, 'samples':end_result.samples}
        super().__init__(end_result.data, end_result.model, strategy, time,
                        kwargs)
        self.stage_results = stage_results

    def _save(self, filename):
        for i, ds in enumerate(self.stage_results):
            ds._save(filename, group = GROUPNAME.format(i), mode='a')
        super()._save(filename, mode='a')

    @classmethod
    def _load(cls, filename):
        try:
            ds = SamplingResult._load(filename)
        except AttributeError:
            # old file
            warn(warn_text)
            ds = SamplingResult._load(filename, group = 'end_result')
            with xr.open_dataset(filename, engine='h5netcdf') as top:
                ds.strategy = get_strategy(top.attrs['strategy'])
        stages = [SamplingResult._load(filename, group=GROUPNAME.format(i))
                        for i in range(len(ds.strategy.stage_strategies)-1)]
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
        confidence=""
        if self.n_sigma != 1:
            confidence=" (\mathrm{{{}\ sigma}})".format(self.n_sigma)
        display_precision = int(round(np.log10(self.guess/(min(self.plus, self.minus))) + .6))
        guess_fmt = "{{:.{}g}}".format(max(display_precision, 2))
        guess = guess_fmt.format(self.guess)
        return "${guess}^{{+{s.plus:.2g}}}_{{-{s.minus:.2g}}}{confidence}$".format(s=self, confidence=confidence, guess=guess)
