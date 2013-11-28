# Copyright 2011-2013, Vinothan N. Manoharan, Thomas G. Dimiduk,
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
Classes for describing free parameters in fitting models

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Jerome Fung <jfung@physics.harvard.edu>
"""
from __future__ import division

import numpy as np
from ..core.holopy_object import HoloPyObject
from .errors import GuessOutOfBoundsError, ParameterSpecificationError

class Parameter(HoloPyObject):
    """
    Describe a free parameter in a fitting model

    Parameters
    ----------
    guess : (optional) float
        Your inital guess for this parameter
    limit : (optional) float or (float, float)
        Describe how the minimizer should allow a parameter to vary.  A single
        value here fixes the parameter to that value, a pair limits the
        parameter to  vary between (min, max)
    name : (optional) string
        A short descriptive name of the parameter.  Do not provide this if using
        the parameter in a :class:`.ParameterizedScatterer`, it will assign a
        name based on the Parameter's position within the scatterer
    **kwargs : varies
        Additional information made available to the minimizer.  This can be
        used for things like step sizes.
    """
    def __init__(self, guess = None, limit = None, name = None, **kwargs):
        self.name = name
        self.guess = guess
        self.limit = limit
        self.kwargs = kwargs

        if self.fixed:
            if guess is not None and guess != limit:
                raise GuessOutOfBoundsError(self)
            self.guess = limit
        else:
            if limit is not None and guess is not None:
                if guess > limit[1] or guess < limit[0]:
                    raise GuessOutOfBoundsError(self)

            if guess is not None:
                if abs(guess) > 1e-12:
                    self.scale_factor = abs(guess)
                else: # values near 0
                    if limit is not None:
                        self.scale_factor = (limit[1] - limit[0])/10.
                    else:
                        self.scale_factor = 1. # guess if necessary
            elif limit is not None:
                # TODO: bug, this will fail for negative limits, but that is
                # sufficently uncommon that I am not going to worry about it for
                # now.
                self.scale_factor = np.sqrt(limit[0]*limit[1])
            else:
                raise ParameterSpecificationError("In order to specify a parameter "
                                                    "you must provide at least an "
                                                    "initial guess or limit")
    @property
    def fixed(self):
        if self.limit is not None:
            try:
                self.limit[1]
            except TypeError:
                return True
        return False

    def scale(self, physical):
        """
        Scales parameters to approximately unity

        Parameters
        ----------
        physical: np.array(dtype=float)

        Returns
        -------
        scaled: np.array(dtype=float)
        """
        return physical / self.scale_factor

    def unscale(self, scaled):
        """
        Inverts scale's transformation

        Parameters
        ----------
        scaled: np.array(dtype=float)

        Returns
        -------
        physical: np.array(dtype=float)
        """
        return scaled * self.scale_factor

class ComplexParameter(Parameter):
    """
    A complex free parameter

    ComplexParameters have a real and imaginary part which can (potentially)
    vary separately.

    Parameters
    ----------
    real, imag : float or :class:`Parameter`
        The real and imaginary parts of this parameter.  Assign floats to fix
        that portion or parameters to allow it to vary.  The parameters must be
        purely real.  You should omit name's for the parameters;
        ComplexParameter will name them 
    name : string
        Short descriptive name of the ComplexParameter.  Do not provide this if
        using a ParameterizedScatterer, a name will be assigned based its
        position within the scatterer.
    """
    def __init__(self, real, imag, name = None):
        '''
        real and imag may be scalars or Parameters. If Parameters, they must be
        pure real.
        '''
        if not isinstance(real, Parameter):
            real = Parameter(real, real)
        self.real = real
        if not isinstance(imag, Parameter):
            imag = Parameter(imag, imag)
        self.imag = imag
        self.name = name

    @property
    def guess(self):
        return self.real.guess + self.imag.guess*1.0j

# provide a shortcut name for Parameter since users will have to type it a lot
par = Parameter
