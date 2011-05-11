# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca W. Perry,
# Jerome Fung, and Ryan McGorty
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
Error classes used in holopy

.. moduleauthor :: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
"""

class NotImplemented(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return "Capability not implemented: " + self.message

class LoadError(Exception):
    def __init__(self, filename, message):
        self.message = message
        self.filename = filename
    def __str__(self):
        return "Error loading file " + self.filename + ": " + self.message
    

class NoFilesFound(Exception):
    def __init__(self, pattern, location):
        self.pattern = pattern
        self.location = location
    def __str__(self):
        return ("No files matching {0} found in {1}.  This could be an error "
                "with you not following our naming conventions.  Fits expect "
                "filenames of the form image%%%%.tif".format(self.pattern,
                                                             self.location))
  
class OpticsError(Exception):
    def __init__(self, message = None):
        self.message = message
    def __str__(self):
	return "Optics instance not specified! " + self.message

class ImageError(Exception):
    def __init__(self, message = None):
        self.message = message
    def __str__(self):
        return "Faulty image: " + self.message

class OutOfBounds(Exception):
    def __init__(self, message = None):
        self.message = message
    def __str__(self):
        return "Image access out of bounds: " + self.message

class ModelInputError(Exception):
    def __init__(self, message = None):
        self.message = message
    def __str__(self):
        return "Input error: " + self.message

class ParameterDefinitionError(Exception):
    def __init__(self, message = None):
        self.message = message
    def __str__(self):
        return "Input error: " + self.message
