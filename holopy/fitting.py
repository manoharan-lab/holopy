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
"""Fit models of scattering to data

Make precision measurements of a scattering system by fitting a model
of it to data

The fitting module is used to:

1. Define Scattering Model -> :class:`~holopy.fitting.model.Model` object
2. Fit model to data -> :class:`.FitResult` object
3. Fit model to timeseries -> list of :class:`.FitResult` objects

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Jerome Fung <jerome.fung@post.harvard.edu>
.. moduleauthor:: Rebecca W. Perry <rperry@seas.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>

The fitting module is deprecated and has been rolled into inference as of
HoloPy 3.3. This file maintains namespaces and warns users to switch.

"""
import warnings
import numpy as np

from holopy.core.metadata import get_values
from holopy.core.utils import ensure_listlike
from holopy.scattering import calc_holo
from holopy.scattering.errors import ParameterSpecificationError, MissingParameter
from holopy.inference.prior import Uniform, ComplexPrior, Prior
from holopy.inference.model import AlphaModel, ExactModel, LimitOverlaps
from holopy.inference.nmpfit import NmpfitStrategy
from holopy.inference.result import UncertainValue, FitResult as RealFitResult

from holopy.core.math import chisq, rsq
from holopy.core.metadata import make_subset_data

def fit_warning(correct_obj, obselete_obj):
    msg = ("HoloPy's inference API has changed. "
    "Use a {} object instead of {}.".format(correct_obj, obselete_obj))
    warnings.warn(msg, UserWarning)
    pass

class Parameter():
    def __new__(self,guess=None, limit=None, name=None, **kwargs):
        fit_warning('hp.inference.prior', 'Parameter')
        if len(ensure_listlike(limit)) == 2:
            if limit[0] == limit[1]:
                return Parameter(guess, limit[0])
            out = Uniform(limit[0], limit[1], guess, name)
        elif guess is None and limit is not None:
            return limit
        elif guess == limit and limit is not None:
            return guess
        elif limit is None and guess is not None:
            out = Uniform(-np.inf, np.inf, guess, name)
        else:
            raise ParameterSpecificationError(
                "Can't interpret Parameter with limit {} and guess {}".format(
                limit, guess))
        setattr(out, 'limit', limit)
        setattr(out, 'kwargs',kwargs)
        return out

class ComplexParameter():
    def __new__(self, real, imag, name=None):
        fit_warning('hp.inference.prior.ComplexPrior', 'ComplexParameter')
        if isinstance(real, Prior) or isinstance(imag, Prior):
            return ComplexPrior(real, imag, name)
        else:
            return real + 1.0j * imag

class ParameterizedObject():
    def __new__(self, obj):
        fit_warning('Scatterer', 'ParameterizedObject')
        setattr(obj, 'make_from', obj.from_parameters)
        return obj

class Parametrization():
    def __new__(self, scatclass, pars):
        fit_warning('Scatterer', 'Parametrization')
        return ParameterizedObject(scatclass(*pars))


class limit_overlaps():
    # defined as a class, not a function for compatibility with pre-release 3.3
    def __new__(self, fraction=.1):
        fit_warning('inference.model.LimitOverlaps', 'limit_overlaps')
        return LimitOverlaps(fraction)

class Model():
    def __new__(self, scatterer, calc_func, medium_index=None,
                 illum_wavelen=None, illum_polarization=None, theory='auto',
                 alpha=None, constraints=[]):
        if calc_func is calc_holo:
            fit_warning('hp.inference.AlphaModel', 'Model')
            if alpha is None:
                alpha = 1.0
            model = AlphaModel(scatterer, alpha, None, medium_index,
                        illum_wavelen, illum_polarization, theory, constraints)
        elif alpha is None:
            fit_warning('hp.inference.ExactModel','Model')
            model = ExactModel(scatterer, calc_func, None, medium_index,
                        illum_wavelen, illum_polarization, theory, constraints)
        else:
            raise ValueError("Cannot interpret alpha for non-hologram scattering \
                            calculations")

        def residual(pars, data):
            return model._residuals(pars, data, 1/np.sqrt(2)).flatten()
        setattr(model, '_calc', model.forward)
        setattr(model, 'residual', residual)
        def get_alpha(pars):
            try:
                return model._get_parameter('alpha', pars)
            except MissingParameter:
                return 1.0
        setattr(model, 'get_alpha', get_alpha)
        return model

class Nmpfit():
    def __new__(self, **kwargs):
        fit_warning('hp.inference.NmpfitStrategy', 'Nmpfit minimizer')
        return NmpfitStrategy(**kwargs)

def fit(model, data, minimizer=None, random_subset=None):
    fit_warning('hp.inference.NmpfitStrategy', 'fit')
    if minimizer is None:
        minimizer = NmpfitStrategy()
    if random_subset is not None:
        minimizer.npixels = int(random_subset*len(data.x)*len(data.y))
    return minimizer.optimize(model, data)

class FitResult():
    def __new__(self, parameters, scatterer, fitchisq, fitrsq, converged, time, model,
                 minimizer, minimization_details):
        fit_warning('hp.inference.result.FitResult', 'FitResult')
        intervals = [UncertainValue(fitted_pars[par.name], diff, name=par.name)
                     for diff, par in zip(minimization_details.perror, parameters)]
        fit = RealFitResult(None, model, minimizer, intervals, None, minimization_details)
        setattr(fit.mpfitdetails, 'converged', converged)
        setattr(fit, 'converged', converged)
        setattr(fit, 'chisq', fitchisq)
        setattr(fit, 'rsq', fitrsq)
        setattr(fit, 'niter', fit.mpfitdetails.niter)
        def fitted_holo(schema):
            fit.data = schema
            return fit.best_fit()
        setattr(fit, 'fitted_holo', fitted_holo)
        return fit
