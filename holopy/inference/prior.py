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



from holopy.core.metadata import get_extents, get_spacing
from holopy.core.process import center_find
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

def make_center_priors(im, z_range_extents=5, xy_uncertainty_pixels=1, z_range_units=None):
    """
    Make sensible default priors for the center of a sphere in a hologram

    Parameters
    ----------
    im : xarray
         The image you wish to make priors for
    z_range_extents : float (optional)
         What range to extend a uniform prior for z over, measured in multiples
         of the total extent of the image. The default is 5 times the extent of
         the image, a large range, but since tempering is quite good at refining
         this, it is safer to just choose a large range to be sure to include
         the correct value.
    xy_uncertainty_pixels: float (optional)
         The number of pixels of uncertainty to assume for the centerfinder.
         The default is 1 pixel, and this is probably correct for most images.
    z_range_units : float
         Specify the range of the z prior in your data units. If this is provided,
         z_range_extents is ignored.
    """
    if z_range_units is not None:
        z_range = z_range_units
    else:
        extents = get_extents(im)
        extent = max(extents['x'], extents['y'])
        z_range = 0, extent * z_range_extents

    spacing = get_spacing(im)
    center = center_find(im) * spacing + [im.x[0], im.y[0]]

    xy_sd = xy_uncertainty_pixels * spacing
    return [Gaussian(c, s) for c, s in zip(center, xy_sd)] + [Uniform(*z_range)]
