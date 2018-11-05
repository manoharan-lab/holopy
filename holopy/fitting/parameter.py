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
"""
Classes for describing free parameters in fitting models.

This API is deprecated. Use holopy.inference instead.

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Jerome Fung <jfung@physics.harvard.edu>
"""

import numpy as np

from holopy.core.utils import ensure_listlike
from holopy.scattering.errors import ParameterSpecificationError
from holopy.fitting.errors import fit_warning

def Parameter(guess=None, limit=None, name=None, **kwargs):
    fit_warning('hp.inference.prior')
    from holopy.inference.prior import Uniform, Fixed
    if len(ensure_listlike(limit)) == 2:
        if limit[0] == limit[1]:
            return Parameter(guess, limit[0])
        out = Uniform(limit[0], limit[1], guess, name)
    elif guess is None:
        out = Fixed(limit, name)
    elif guess == limit:
        out = Fixed(guess, name)
    elif limit is None:
        out = Uniform(-np.inf, np.inf, guess, name)
    else:
        raise ParameterSpecificationError(
                "Can't interpret Parameter with limit {} and guess {}".format(
                limit, guess))
    setattr(out, 'limit', limit)
    setattr(out, 'kwargs',kwargs)
    return out

def ComplexParameter(real, imag, name=None):
    fit_warning('hp.inference.prior.ComplexPrior')
    from holopy.inference.prior import ComplexPrior
    return ComplexPrior(real, imag, name)

