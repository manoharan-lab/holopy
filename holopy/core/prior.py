# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, and Ryan McGorty, Anna Wang, Solomon Barkley
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

from copy import copy
import operator

import numpy as np
from numpy import random
from numbers import Number, Real
from scipy import stats

from holopy.core.metadata import get_extents, get_spacing
from holopy.core.utils import ensure_listlike
from holopy.core.process import center_find
from holopy.core.holopy_object import HoloPyObject
from holopy.scattering.errors import ParameterSpecificationError

EPS = 1e-6


def _reciprocal(val):
    return 1 / val


class Prior(HoloPyObject):
    """Base class for Bayesian priors in holopy.
    
       Prior subclasses should define at least the following methods:
       
           - guess
           - sample
           - prob
           - lnprob
    """

    def __init__(self):
        raise NotImplementedError("Use subclass with a defined probability"
                                  "distribution method prob and/or lnprob.")

    def __add__(self, value):
        if isinstance(value, (Number, Prior)):
            if value == 0:
                return self
            else:
                return TransformedPrior(operator.add, [self, value])
        elif isinstance(value, np.ndarray):
            return np.array([self + val for val in value])
        else:
            raise TypeError(
                "Cannot add prior to objects of type {}".format(type(value)))

    def __mul__(self, value):
        if isinstance(value, (Real, Prior)):
            if value == 0:
                raise TypeError("Cannot multiply a prior by 0")
            elif value == 1:
                return self
            else:
                return TransformedPrior(operator.mul, [self, value])
        elif isinstance(value, np.ndarray):
            return np.array([self * val for val in value])
        else:
            badtype = type(value)
            raise TypeError(
                "Cannot multiply prior by objects of type {}".format(badtype))

    def __radd__(self, value):
        return self + value

    def __sub__(self, value):
        return self + (-value)

    def __rsub__(self, value):
        return -self + value

    def __rmul__(self, value):
        return self * value

    def __truediv__(self, value):
        return self * (1/value)

    def __rtruediv__(self, value):
        return value * TransformedPrior(_reciprocal, self)

    def __neg__(self):
        return self * -1

    def __pow__(self, value):
        return TransformedPrior(operator.pow, [self, value])

    def __rpow__(self, value):
        return TransformedPrior(operator.pow, [value, self])

    def __array_ufunc__(self, ufunc, method, *args, name=None, **kwargs):
        if method == "__call__" and len(kwargs) == 0:
            return TransformedPrior(ufunc, args, name)
        else:
            raise TypeError('Could not apply numpy ufunc to Prior object. '
                            'Use TransformedPrior.')

    def scale(self, physical):
        return physical / self.scale_factor

    def unscale(self, scaled):
        return scaled * self.scale_factor

    def renamed(self, name):
        like_me = copy(self)
        like_me.name = name
        return like_me


class Uniform(Prior):
    def __init__(self, lower_bound, upper_bound, guess=None, name=None):
        """
        Uniform prior.

        Parameters
        ----------
        lower_bouund, upper_bound : float
        guess : float or None, optional
            The value to take as an initial guess from the prior. If
            guess is None, defaults to the midpoint of the prior for
            proper priors.
        name : string or None, optional
            The name of the parameter.
        """
        if lower_bound >= upper_bound:
            raise ParameterSpecificationError(
                    "Lower bound {} is not less than upper bound {}".format(
                    lower_bound, upper_bound))
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.name = name

        if np.isfinite(self.interval):
            self._lnprob = np.log(1/self.interval)
        else:
            self._lnprob = -1/EPS  # don't want -inf to add likelihood

        if guess is None:
            if np.isfinite(lower_bound) and np.isfinite(upper_bound):
                self.guess = (upper_bound + lower_bound) / 2
            elif np.isfinite(lower_bound):
                self.guess = lower_bound
            elif np.isfinite(upper_bound):
                self.guess = upper_bound
            else:
                self.guess = 0
        elif guess < lower_bound or guess > upper_bound:
            raise ParameterSpecificationError(
                    "Guess {} is not within bounds {} and {}.".format(
                    guess, lower_bound, upper_bound))
        else:
            self.guess = guess

        if abs(self.guess) > 1e-12:
            self.scale_factor = abs(self.guess)
        elif np.isfinite(self.interval):
            self.scale_factor = self.interval/10.
        else:
            self.scale_factor = 1.

    def lnprob(self, p):
        if p < self.lower_bound or p > self.upper_bound:
            return -np.inf
        # For a uniform prior, the value is always the same, so precompute it
        return self._lnprob
        # Turns out scipy.stats is noticably slower than doing it ourselves
        # return stats.uniform.logpdf(p, self.lower_bound, self.upper_bound)

    def prob(self, p):
        if p < self.lower_bound or p > self.upper_bound:
            return 0
        return 1/self.interval

    @property
    def interval(self):
        return(self.upper_bound - self.lower_bound)

    def sample(self, size=None):
        return random.uniform(self.lower_bound, self.upper_bound, size)


class Gaussian(Prior):
    def __init__(self, mu, sd, name=None):
        """
        Gaussian prior.

        Parameters
        ----------
        mu, sd : float
            The mean and standard deviation of the Gaussian.
        name : string or None, optional
            The name of the parameter.
        """
        self.mu = mu
        self.sd = sd
        if sd <= 0:
            raise ParameterSpecificationError(
                    "Specified sd of {} is not greater than 0".format(sd))
        self.name = name
        self._lnprob_normalization = -np.log(self.sd * np.sqrt(2*np.pi))

        if abs(self.guess) > 1e-12:
            self.scale_factor = abs(self.guess)
        else:  # values near 0
            self.scale_factor = self.sd

    @property
    def variance(self):
        return self.sd**2

    def lnprob(self, p):
        return self._lnprob_normalization - (p-self.mu)**2/(2*self.variance)
        # Turns out scipy.stats is noticably slower than doing it ourselves
        # return stats.norm.logpdf(p, self.mu, self.sd)

    def prob(self, p):
        return stats.norm.pdf(p, self.mu, self.sd)

    @property
    def guess(self):
        return self.mu

    def sample(self, size=None):
        return random.normal(self.mu, self.sd, size=size)


class BoundedGaussian(Gaussian):
    # Note: this is not normalized
    def __init__(self, mu, sd, lower_bound=-np.inf, upper_bound=np.inf,
                 name=None):
        """Gaussian prior restricted to an interval.

        Note that the `prob` and `lnprob` methods return a value proportional
        to the probability only, and not the actual probability.

        Parameters
        ----------
        mu, sd : float
            The mean and standard deviation of the Gaussian.
        lower_bouund, upper_bound : float, optional
            Defaults to +- infinity.
        name : string or None, optional
            The name of the parameter.
        """

        if mu < lower_bound or mu > upper_bound or lower_bound == upper_bound:
            raise ParameterSpecificationError(
                "Lower bound {} must be less than mean {}. Upper bound {} must"
                " be greater than mean.")

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        super().__init__(mu, sd, name)

    def lnprob(self, p):
        """Note that this does not return the actual log-probability, but
        a value proportional to it.
        """
        if p < self.lower_bound or p > self.upper_bound:
            return -np.inf
        else:
            return super().lnprob(p)

    def prob(self, p):
        """Note that this does not return the actual probability, but
        a value proportional to it.
        """
        if p < self.lower_bound or p > self.upper_bound:
            return 0
        else:
            return super().prob(p)

    def sample(self, size=None):
        val = super().sample(size)
        out = True
        while np.any(out):
            out = np.logical_or(val < self.lower_bound, val > self.upper_bound)
            out = np.where(out)
            val[out] = super().sample(len(out[0]))
        return val


class TransformedPrior(Prior):
    def __init__(self, transformation, base_prior, name=None):
        """Composite prior composed of one or more base priors transformed by
        a function. Note there are no `prob` and `lnprob` methods since those
        just depend on the probabilities of the underlying base priors.

        Parameters
        ----------
        transformation : func
            Function to apply to base prior to get transformed value
        base_prior : Prior object or listlike containing Priors
            Values to be passed into transformation function
        name : string or None, optional
            The name of the parameter.
        """
        self.base_prior = tuple(ensure_listlike(base_prior))
        if callable(transformation):
            self.transformation = transformation
        else:
            msg = 'transformation must be function of one or more base priors'
            raise TypeError(msg)
        self.name = name

    def prob(self, p):
        msg = "Cannot calculate probability. Use base priors."
        raise NotImplementedError(msg)

    def lnprob(self, p):
        msg = "Cannot calculate probability. Use base priors."
        raise NotImplementedError(msg)

    def sample(self, size=None):
        if size is None:
            repeat = lambda x: x
        else:
            repeat = lambda x: np.repeat(x, size)
        raw_samples = [bp.sample(size) if isinstance(bp, Prior) else repeat(bp)
                       for bp in self.base_prior]
        if size is None:
            return self.transformation(*raw_samples)
        else:
            return np.array([self.transformation(*sample_set)
                             for sample_set in zip(*raw_samples)])
        return samples

    @property
    def guess(self):
        guess_vals = [bp.guess if isinstance(bp, Prior) else bp
                      for bp in self.base_prior]
        return self.transformation(*guess_vals)

    @property
    def map_keys(self):
        if len(self.base_prior) == 1:
            return [('',) + self.base_prior]
        else:
            return enumerate(self.base_prior)


class ComplexPrior(TransformedPrior):
    """
    A complex free parameter

    ComplexPrior has a real and imaginary part which can (potentially)
    vary separately.

    Parameters
    ----------
    real, imag : float or :class:`Prior`
        The real and imaginary parts of this parameter.  Assign floats to fix
        that portion or parameters to allow it to vary.  The parameters must be
        purely real.  You should omit names for the parameters;
        ComplexPrior will name them
    name : string
        Short descriptive name of the ComplexPrior.  Do not provide this if
        using a ParameterizedScatterer, a name will be assigned based its
        position within the scatterer.
    """
    def __init__(self, real, imag, name=None):
        '''
        real and imag may be scalars or Priors. If Priors, they must be
        pure real.
        '''
        self.transformation = complex
        self.base_prior = [real, imag]
        self.name = name

    @property
    def real(self):
        return self.base_prior[0]

    @property
    def imag(self):
        return self.base_prior[1]

    def lnprob(self, p):
        try:
            realprob = self.real.lnprob(np.real(p))
        except AttributeError:
            realprob = 0
        try:
            imagprob = self.imag.lnprob(np.imag(p))
        except AttributeError:
            imagprob = 0
        return realprob + imagprob

    def prob(self, p):
        return np.exp(self.lnprob(p))

    @property
    def map_keys(self):
        return ((key, getattr(self, key)) for key in ['real', 'imag'])


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
        return BoundedGaussian(v.guess, sd,
                               getattr(prior, 'lower_bound', -np.inf),
                               getattr(prior, 'upper_bound', np.inf),
                               name=prior.name)
    else:
        return Gaussian(v.guess, sd, prior.name)


def generate_guess(parameters, nguess=1, scaling=1, seed=None):
    def scaled_sample(prior):
        raw_sample = prior.sample(size=nguess)
        scaled_guess = prior.guess + scaling * (raw_sample - prior.guess)
        return scaled_guess

    if seed is not None:
        np.random.seed(seed)
    return np.vstack([scaled_sample(p) for p in parameters]).T


def make_center_priors(im, z_range_extents=5, xy_uncertainty_pixels=1,
                       z_range_units=None):
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
         Specify the range of the z prior in your data units. If provided,
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
