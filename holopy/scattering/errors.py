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
Exceptions used in scatterpy module.  These are separated out from the
other exceptions in other parts of HoloPy to keep things modular.

.. moduleauthor :: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""
from __future__ import division

import warnings
import exceptions

from ..core.errors import Error

class InvalidScatterer(Exception):
    def __init__(self, scatterer, message):
        self.scatterer = scatterer
        super(InvalidScatterer, self).__init__(message)

class OverlapWarning(exceptions.UserWarning):
    def __init__(self, scatterer, overlaps):
        self.scatterer = scatterer
        self.overlaps = overlaps
    def __str__(self):
        return "{0} has overlaps between spheres: {1}".format(repr(self.scatterer),
                                                              self.overlaps)
warnings.simplefilter('always', OverlapWarning)

class ScattererDefinitionError(Error):
    def __init__(self, message, scatterer):
        self.scatterer = scatterer
        super(ScattererDefinitionError, self).__init__(message)
    def __str__(self):
        return ("Error defining scatterer object of type " +
                self.scatterer.__class__.__name__ +
                ".\n" + self.message)

class TheoryNotCompatibleError(Error):
    def __init__(self, theory, scatterer, message = None):
        self.theory = theory
        self.scatterer = scatterer
        super(TheoryNotCompatibleError, self).__init__(message)
    def __str__(self):
        if self.message is None:
            return ("The implementation of the " +
                    self.theory.__class__.__name__ +
                    " scattering theory doesn't know how to handle " +
                    "scatterers of type " +
                    self.scatterer.__class__.__name__)
        else:
            return ("The implementation of the " +
                    self.theory.__class__.__name__ +
                    " scattering theory doesn't know how to handle " +
                    "scatterers of type " +
                    self.scatterer.__class__.__name__ +
                    " because: " + self.message)

class UnrealizableScatterer(Error):
    def __init__(self, theory, scatterer, message):
        self.theory = theory
        self.scatterer = scatterer
        super(UnrealizableScatterer, self).__init__(message)
    def __str__(self):
        return ("Cannot compute scattering with "+ self.theory.__class__.__name__
                + " scattering theory for a scatterer of type " +
                self.scatterer.__class__.__name__ + " because: " + self.message)

class ModelInputError(Error):
    pass

class NoCenter(Error):
    pass

class NoPolarization(Error):
    pass

class MultisphereFieldNaN(UnrealizableScatterer):
    def __str__(self):
        return "Fields computed with Multisphere are NaN, this probably "
        "represents a failure of the code to converge, check your scatterer."


class MultisphereExpansionNaN(Exception):
    def __str__(self):
        return ("Internal expansion for Multisphere coefficients contains "
                "NaN.  This probably means your scatterer is unphysical.")

class ConvergenceFailureMultisphere(Exception):
    def __str__(self):
        return ("Multisphere calculations failed to converge, this probably means "
                "your scatterer is unphysical, or possibly just huge")
