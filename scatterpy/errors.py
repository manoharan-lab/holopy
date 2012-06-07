# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca
# W. Perry, Jerome Fung, and Ryan McGorty
#
# This file is part of Holopy.
#
# Holopy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Holopy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Holopy.  If not, see <http://www.gnu.org/licenses/>.
"""
Exceptions used in scatterpy module.  These are separated out from the
other exceptions in other parts of holopy to keep things modular.

.. moduleauthor :: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""
from __future__ import division

# Defined as a seperate function to make it easier to redirect or quiet warnings
# in the future
def warning(msg, context=None):
    print("WARNING:{0}".format(msg))

class InvalidScatterer(Exception):
    pass

class InvalidScattererSphereOverlap(InvalidScatterer):
    def __init__(self, scatterer, overlaps):
        self.scatterer = scatterer
        self.overlaps = overlaps
    def __str__(self):
        return "{0} has overlaps between spheres: {1}".format(repr(self.scatterer),
                                                              self.overlaps)

class ScattererDefinitionError(Exception):
    def __init__(self, message, scatterer):
        self.message = message
        self.scatterer = scatterer
    def __str__(self):
        return ("Error defining scatterer object of type " + 
                self.scatterer.__class__.__name__ +
                ".\n" + self.message)

class TheoryNotCompatibleError(Exception):
    def __init__(self, theory, scatterer):
        self.theory = theory
        self.scatterer = scatterer
    def __str__(self):
        return ("The implementation of the " +
                self.theory.__class__.__name__ + 
                " scattering theory doesn't know how to handle " +
                "scatterers of type " + 
                self.scatterer.__class__.__name__)

class UnrealizableScatterer(Exception):
    def __init__(self, theory, scatterer, message):
        self.theory = theory
        self.scatterer = scatterer
        self.message = message
    def __str__(self):
        return ("Cannot compute scattering with "+ self.theory.__class__.__name__
                + " scattering theory for a scatterer of type " +
                self.scatterer.__class__.__name__ + " because: " + self.message)
