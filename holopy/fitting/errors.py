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

from ..core.errors import Error

class OutOfBounds(Error):
    def __str__(self):
        return "Image access out of bounds: " + self.message

class ParameterDefinitionError(Error):
    def __str__(self):
        return "Input error: " + self.message

class ParameterSpecificationError(Error):
    pass

class ModelDefinitionError(Error):
    pass
    
class GuessOutOfBoundsError(ParameterSpecificationError):
    def __init__(self, parameter):
        self.par = parameter
    def __str__(self):
        if self.par.fixed:
            return "guess {s.guess} does not match fixed value {s.limit}".format(s=self.par)
        return "guess {s.guess} is not within bounds {s.limit}".format(s=self.par)
    
class MinimizerConvergenceFailed(Error):
    def __init__(self, result, details):
        self.result = result
        self.details = details

class InvalidMinimizer(Error):
    pass
