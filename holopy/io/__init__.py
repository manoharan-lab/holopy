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
Module for reading and writing holograms, input decks, and yaml files.
This includes some image loading code that may be useful for more than
just holograms.

.. moduleauthor:: Thomas G. Dimiduk <tdiimduk@physics.harvard.edu>
"""
from __future__ import division

from yaml_io import save, load_yaml
from image_io import load, save_image
import image_io
import yaml_io

