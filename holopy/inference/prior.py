# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, and Ryan McGorty, Anna Wang
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


from __future__ import division

from holopy.fitting.parameter import Parameter
from holopy.fitting.errors import ParameterSpecificationError

import numpy as np
from numpy import random



class Prior(Parameter):
    pass


class Uniform(Prior):
    def __init__(self, lower_bound, upper_bound, name=None):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.name = name
        self._lnprob = np.log(1/(self.upper_bound - self.lower_bound))

    def lnprob(self, p):
        if p < self.lower_bound or p > self.upper_bound:
            return -np.inf
        # For a uniform prior, the value is the same every time, so we precompute it
        return self._lnprob
        # Turns out scipy.stats is noticably slower than doing it ourselves
        #return stats.uniform.logpdf(p, self.lower_bound, self.upper_bound)

    @property
    def interval(self):
        return self.upper_bound - self.lower_bound

    @property
    def guess(self):
        return (self.upper_bound - self.lower_bound)/2

    def sample(self, size=None):
        return random.uniform(self.lower_bound, self.upper_bound, size)


class Gaussian(Prior):
    def __init__(self, mu, sd, name=None):
        self.mu = mu
        self.sd = sd
        if sd <= 0:
            raise ParameterSpecificationError("Specified sd of {} is not greater than 0".format(sd))
        self.sdsq = sd**2
        self.name=name
        self._lnprob_normalization = -np.log(self.sd * np.sqrt(2*np.pi))

    def lnprob(self, p):
        return self._lnprob_normalization - (p-self.mu)**2/(2*self.sdsq)
        # Turns out scipy.stats is noticably slower than doing it ourselves
        #return stats.norm.logpdf(p, self.mu, self.sd)

    def prob(self, p):
        return stats.norm.pdf(p, self.mu, self.sd)

    @property
    def guess(self):
        return self.mu

    def sample(self, size=None):
        return random.normal(self.mu, self.sd, size=size)


class BoundedGaussian(Gaussian):
    # Note: this is not normalized
    def __init__(self, mu, sd, lower_bound=-np.inf, upper_bound=np.inf, name=None):
        if mu < lower_bound or mu > upper_bound:
            raise OutOfBoundsError(mu, lower_bound, upper_bound)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        super(BoundedGaussian, self).__init__(mu, sd, name)

    def lnprob(self, p):
        if np.any(p < self.lower_bound) or np.any(p > self.upper_bound):
            return -np.inf
        else:
            return super(BoundedGaussian, self).lnprob(p)


    def sample(self, size=None):
        val = super(BoundedGaussian, self).sample(size)
        out = True
        while np.any(out):
            out = np.where(np.logical_and(val < self.lower_bound, val > self.upper_bound))
            val[out] = super(BoundedGaussian, self).sample(len(out[0]))

        return val


def updated(prior, v, extra_uncertainty=0):
    """
    Update a prior from a posterior

    Parameters
    ----------
    v : UncertainValue
        The new value, usually from an mcmc result
    extra_uncertainty : float
        provide a floor for uncertainty (sd) of the new parameter
    """
    sd = max(v.plus, v.minus, extra_uncertainty)
    if hasattr(prior, 'lower_bound'):
        return BoundedGaussian(v.value, sd,
                               getattr(prior, 'lower_bound', -np.inf),
                               getattr(prior, 'upper_bound', np.inf),
                               name=prior.name)
    else:
        return Gaussian(v.value, sd, prior.name)
