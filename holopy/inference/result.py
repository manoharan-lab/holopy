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

import yaml
import xarray as xr
import numpy as np
import scipy.special
import h5py

from holopy.core.metadata import detector_grid, copy_metadata
from holopy.core.holopy_object import HoloPyObject
from holopy.core.io.io import pack_attrs, unpack_attrs
from holopy.core.utils import dict_without, ensure_array

class InferenceResult(HoloPyObject):
    def __init__(self, data, model, strategy, intervals, time):
        self.data = data
        self.model = model
        self.strategy = strategy
        self.intervals = intervals
        self.time = time

    @property
    def guess(self):
        return [val.guess for val in self.intervals]

    @property
    def _names(self):
        return [val.name for val in self.intervals]

    @property
    def parameters(self):
        return {name:val for name, val in zip(self._names, self.guess)}

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
        original_dims = yaml.load(self.data.original_dims)
        # can't currently handle non-0 values of z, as in detector_grid
        x = original_dims['x']
        y = original_dims['y']
        shape = (len(x), len(y))
        spacing = (np.diff(x)[0], np.diff(y)[0])
        extra_dims = dict_without(original_dims,['x', 'y', 'z'])
        schema = detector_grid(shape, spacing, extra_dims = extra_dims)
        schema = copy_metadata(self.data, schema, do_coords=False)
        schema['x']=x
        schema['y']=y
        self._best_fit = self.model.forward(self.parameters(), schema)
        return self.best_fit()

    @property
    def max_lnprob(self):
        # calculate the first time it's called and then store it
        try:
            return self._max_lnprob
        except AttributeError:
            pass
        self._max_lnprob = self.model.lnposterior(self.parameters, self.data)
        return self.max_lnprob

    @property
    def _source_class(self):
        return "holopy.inference.{}".format(self.__class__.__name__)

    def _serialization_ds(self):
        ds = xr.Dataset({'data':self.data})
        ds.data.attrs = pack_attrs(ds.data)
        ds_attrs = ['model', 'strategy', 'intervals', 'time', '_source_class']
        for attr_name in ds_attrs:
            attr = getattr(self, attr_name)
            if isinstance(attr, HoloPyObject) or (isinstance(attr,list) and 
                                            isinstance(attr[0],HoloPyObject)):
                attr = yaml.dump(attr)
            ds.attrs[attr_name] = str(attr)
        if 'flat' in ds:
            ds.rename({'flat': 'point'}, inplace=True)
            ds['point'].values = np.arange(len(ds.point))
        return ds

    def _save(self, filename, **kwargs):
        ds = self._serialization_ds()
        ds.to_netcdf(filename, engine='h5netcdf', **kwargs)

    @classmethod
    def _unserialize(cls, ds):
        data = ds.data
        data.attrs = unpack_attrs(data.attrs)
        model = yaml.load(ds.attrs.pop('model'))
        strategy = yaml.load(ds.attrs.pop('strategy'))
        intervals = yaml.load(ds.attrs.pop('intervals'))
        intervals=None
        time = float(ds.attrs.pop('time'))
        return [data, model, strategy, intervals, time]

    @classmethod
    def _load(cls, ds, **kwargs):
       with xr.open_dataset(ds, engine='h5netcdf', **kwargs) as ds:
            args = cls._unserialize(ds)
       return cls(*args)


class SamplingResult(InferenceResult):
    def __init__(self, data, model, strategy, time, lnprobs, samples):
        self.samples = samples
        self.lnprobs = lnprobs
        intervals = self._calc_intervals()
        super().__init__(data, model, strategy, intervals, time)

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
            array.sel(chain = slice(sample_number, None))

        burned_in = copy(self)
        cut_start(burned_in.samples)
        cut_start(burned_in.lnprobs)
        burned_in.intervals = burned_in._calc_intervals()
        return burned_in

    def _serialization_ds(self):
        ds = super()._serialization_ds()
        attrs = ds.attrs
        ds = xr.merge([ds, {'lnprobs':self.lnprobs, 'samples':self.samples}])
        ds.attrs = attrs
        return ds

    @classmethod
    def _unserialize(cls, ds):
        args = super()._unserialize(ds)
        del args[3] # intervals
        return args + [ds.lnprobs, ds.samples]

GROUPNAME = 'stage_results[{}]'
class TemperedSamplingResult(SamplingResult):
    def __init__(self, end_result, stage_results, strategy, time):
        super().__init__(end_result.data, end_result.model, strategy, time, 
                        end_result.lnprobs, end_result.samples)
        self.stage_results = stage_results

    def _save(self, filename):
        super()._save(filename)
        for i, ds in enumerate(self.stage_results):
            ds._save(filename, group = GROUPNAME.format(i), mode='a')

    @classmethod
    def _load(cls, filename):
        ds = SamplingResult._load(filename)
        stages = [SamplingResult._load(filename, group=GROUPNAME.format(i))
                        for i in range(len(ds.strategy.stage_strategies))]
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
        self.guess = np.asscalar(ensure_array(guess))
        self.plus = np.asscalar(ensure_array(plus))
        if minus is None:
            self.minus = self.plus
        else:
            self.minus = np.asscalar(ensure_array(minus))
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
