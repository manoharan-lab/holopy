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

_res_ds_contents = ['samples', 'lnprobs', 'data']

class SamplingResult(HoloPyObject):
    def __init__(self, dataset, model, strategy):
        self.dataset = dataset
        self.model = model
        self.strategy = strategy

    @property
    def samples(self):
        return self.dataset.samples

    @property
    def lnprobs(self):
        return self.dataset.lnprobs

    @property
    def data(self):
        return self.dataset.data

    @property
    def MAP(self):
        m = self.samples[np.unravel_index(self.lnprobs.argmax(), self.lnprobs.shape)]
        return xr.DataArray(m, dims=['parameter'], coords={'parameter': m.parameter}, attrs={})

    @property
    def mean(self):
        return self.samples.mean(dim=['walker', 'chain'])

    @property
    def median(self):
        return self.samples.median(dim=['walker', 'chain'])

    def values(self, sigma_interval=1):
        def uncval(mp, interval):
            mp = float(mp)
            minus = mp - float(interval.sel(sigma=-sigma_interval))
            plus = float(interval.sel(sigma=sigma_interval)) - mp
            return UncertainValue(mp, plus=plus, minus=minus)


        return {p.name: uncval(MAP, si) for p, MAP, si in zip(self.model.parameters, self.MAP, self.sigma_intervals([-sigma_interval, sigma_interval]).T)}

    def sigma_intervals(self, sigmas=[-2, -1, 1, 2]):
        def quantile(s):
            q = 50 * (1+scipy.special.erf(s/np.sqrt(2)))
            p = self.samples.reduce(np.percentile, q=q, dim=['walker', 'chain'])
            p.coords['sigma'] = s
            return p
        return xr.concat([quantile(s) for s in sigmas], dim='sigma')

    @property
    def _names(self):
        return [p.name for p in self.model.parameters]

    def _serialization_ds(self):
        ds = copy(self.dataset)
        ds.attrs['model'] = yaml.dump(self.model)
        ds.attrs['strategy'] = yaml.dump(self.strategy)
        ds.attrs['_source_class'] = self._source_class
        ds.data.attrs = pack_attrs(ds.data)
        autocorr_to_sentinal(ds.samples)
        autocorr_to_sentinal(ds.lnprobs)
        if 'flat' in ds:
            ds.rename({'flat': 'point'}, inplace=True)
            ds['point'].values = np.arange(len(ds.point))
        return ds

    @property
    def _source_class(self):
        return "holopy.inference.{}".format(self.__class__.__name__)

    def _save(self, filename):
        ds = self._serialization_ds()

        ds.to_netcdf(filename, engine='h5netcdf')

    @classmethod
    def _load(cls, ds):
        if isinstance (ds, str):
            ds = xr.open_dataset(ds, engine='h5netcdf')
        ds.data.attrs = unpack_attrs(ds.data.attrs)
        model = yaml.load(ds.attrs.pop('model'))
        strategy = yaml.load(ds.attrs.pop('strategy'))
        r = cls(dataset=ds, model=model, strategy=strategy)
        autocorr_from_sentinal(r.samples)
        autocorr_from_sentinal(r.lnprobs)
        return r

    def best_fit(self):
        shape, spacing, start, coords = yaml.load(self.dataset.data.original_dims)
        schema = detector_grid(shape, spacing, extra_dims = coords)
        schema['x'] = schema['x'] + start[0]
        schema['y'] = schema['y'] + start[1]
        schema = copy_metadata(self.dataset.data, schema, do_coords = False) 
        return self.model._forward(self.values(), schema)

    def output_scatterer(self):
        return self.model.scatterer.make_from(self.values())

def get_stage_names(inf):
    d = OrderedDict([(k, k) for k in h5py.File(inf).keys()])
    del d['end_result']
    return d

class TemperedSamplingResult(SamplingResult):
    def __init__(self, end_result, stage_results, strategy):
        self.end_result = end_result
        self.stage_results = stage_results
        self.strategy = strategy


    @property
    def model(self):
        return self.end_result.model

    @property
    def dataset(self):
        return self.end_result.dataset

    def _save(self, filename):
        # make up a dummy xarray so that we have somewhere to store the strategy
        s = xr.Dataset({}, attrs={'strategy': yaml.dump(self.strategy), '_source_class': self._source_class})
        s.to_netcdf(filename, engine='h5netcdf')

        def write(ds, group, mode='a'):
            ds.to_netcdf(filename, engine='h5netcdf', group=group, mode=mode)

        write(self.end_result._serialization_ds(), 'end_result')

        for i, sr in enumerate(self.stage_results):
            write(sr._serialization_ds(), 'stage_results[{}]'.format(i))


    @classmethod
    def _load(cls, inf):
        with xr.open_dataset(inf, engine='h5netcdf') as top:
            strategy = yaml.load(top.attrs['strategy'])

        with xr.open_dataset(inf, engine='h5netcdf', group='end_result') as end:
            # we have to force the load, since the lazy load does not get
            # called before the context manager closes the file
            end.load()
            end_result = SamplingResult._load(end)
        stages = []
        for stage in get_stage_names(inf):
            with xr.open_dataset(inf, engine='h5netcdf', group=stage) as sds:
                sds.load()
                stages.append(SamplingResult._load(sds))

        return TemperedSamplingResult(end_result, stages, strategy)

def autocorr_to_sentinal(d):
    if 'autocorr' in d.attrs and d.attrs['autocorr'] == None:
        d.attrs['autocorr'] = -1
    return d

def autocorr_from_sentinal(d):
    if 'autocorr' in d.attrs and d.attrs['autocorr'] == -1:
        d.attrs['autocorr'] = None
    return d

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
    def __init__(self, value, plus, minus=None, n_sigma=1, kind='MAP'):
        self.value = value
        self.plus = plus
        self.minus = minus
        self.n_sigma = n_sigma

    def _repr_latex_(self):
        from IPython.display import Math
        confidence=""
        if self.n_sigma != 1:
            confidence=" (\mathrm{{{}\ sigma}})".format(self.n_sigma)
        display_precision = int(round(np.log10(self.value/(min(self.plus, self.minus))) + .6))
        value_fmt = "{{:.{}g}}".format(max(display_precision, 2))
        value = value_fmt.format(self.value)
        return "${value}^{{+{s.plus:.2g}}}_{{-{s.minus:.2g}}}{confidence}$".format(s=self, confidence=confidence, value=value)
