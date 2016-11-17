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
import yaml

import xarray as xr
import numpy as np
import scipy.special

from holopy.core.holopy_object import HoloPyObject

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
        return self.samples[np.unravel_index(self.lnprobs.argmax(), self.lnprobs.shape)]

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
        ds.attrs['_source_class'] = "holopy.inference.{}".format(self.__class__.__name__)
        autocorr_to_sentinal(ds.samples)
        autocorr_to_sentinal(ds.lnprobs)
        return ds

    def save(self, filename):
        self._serialization_ds().to_netcdf(filename, engine='h5netcdf')

    @classmethod
    def _load(cls, ds):
        model = yaml.load(ds.attrs.pop('model'))
        strategy = yaml.load(ds.attrs.pop('strategy'))
        r = cls(dataset=ds, model=model, strategy=strategy)
        autocorr_from_sentinal(r.samples)
        autocorr_from_sentinal(r.lnprobs)
        return r


class TemperedSamplingResult(SamplingResult):
    def __init__(self, end_result, stage_results, model, strategy):
        self.dataset = end_result
        self.stage_results = stage_results
        self.model = model
        self.strategy = strategy

    def _serialization_ds(self):
        ds = super()._serialization_ds()
        for i, r in enumerate(self.stage_results):
            for d in _res_ds_contents:
                ds["stage_{}_{}".format(i, d)] = getattr(r, d)

        return ds

    @classmethod
    def _load(cls, ds):
        model = yaml.load(ds.attrs.pop('model'))
        strategy = yaml.load(ds.attrs.pop('strategy'))
        i = 0
        stages = []
        while hasattr(ds, "stage_{}_samples".format(i)):
            c = {}
            for d in _res_ds_contents:
                c[d] = getattr(ds, "stage_{}_{}".format(i, c))
                stages.append(SamplingResult(xr.DataSet(c, model, strategy)))
            i += 1
        result = SamplingResult(ds, model, strategy)
        return TemperedSamplingResult(result, stages, model, strategy)


def autocorr_to_sentinal(d):
    if d.attrs['autocorr'] == None:
        d.attrs['autocorr'] = -1

def autocorr_from_sentinal(d):
    if d.attrs['autocorr'] == -1:
        d.attrs['autocorr'] = None


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
