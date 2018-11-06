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

"""
import warnings
import numpy as np

from holopy.core.metadata import get_values
from holopy.core.utils import ensure_listlike
from holopy.scattering import calc_holo
from holopy.scattering.errors import ParameterSpecificationError
from holopy.inference.prior import Uniform
from holopy.inference.model import AlphaModel, ExactModel

from holopy.core.math import chisq, rsq
from holopy.core.metadata import make_subset_data
from holopy.inference.prior import ComplexPrior as ComplexParameter
from holopy.inference.model import LimitOverlaps as limit_overlaps
from holopy.inference.nmpfit import NmpfitStrategy as Nmpfit

def fit_warning(correct_obj):
        msg = "HoloPy's fitting API is deprecated. \
        Use a {} object instead.".format(correct_obj)
        warnings.warn(msg, UserWarning)
        pass

def Parameter(guess=None, limit=None, name=None, **kwargs):
    fit_warning('hp.inference.prior')
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

def Model(scatterer, calc_func, medium_index=None,
                 illum_wavelen=None, illum_polarization=None, theory='auto',
                 alpha=None, constraints=[]):
    from holopy.inference.model import AlphaModel, ExactModel
    if calc_func is calc_holo:
        fit_warning('hp.inference.AlphaModel')
        if alpha is None:
            alpha = 1.0
        model = AlphaModel(scatterer, None, alpha, medium_index, illum_wavelen,
                        illum_polarization, theory, constraints)
    elif alpha is None:
        fit_warning('hp.inference.ExactModel')
        model = ExactModel(scatterer, calc_func, None, medium_index,
                        illum_wavelen, illum_polarization, theory, constraints)
    else:
        raise ValueError("Cannot interpret alpha for non-hologram scattering \
                            calculations")
    def residual(pars, data):
        return get_values(model.forward(pars, data) - data).flatten()
    setattr(model, '_calc', model.forward)
    setattr(model, 'residual', residual)
    return model

def fit(model, data, minimizer=None, random_subset=None):
    from holopy.inference import NmpfitStrategy
    fit_warning('hp.inference.prior')
    if minimizer is None:
        minimizer = NmpfitStrategy()
    if random_subset is not None:
        minimizer.random_subset = random_subset
    return minimizer.fit(model, data)
