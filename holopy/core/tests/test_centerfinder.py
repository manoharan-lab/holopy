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
from __future__ import division
import numpy as np
from numpy.testing import assert_allclose
from ..process.centerfinder import center_find
from .common import get_example_data

gold_location = np.array([ 48.62964885,  50.22328811])

def test_FoundLocation():
    #load a hologram
    holo = get_example_data('image0001.yaml')
    
    #find the center of it
    location = center_find(holo, threshold=.25)
	
    #check to make sure it matches the gold
    assert_allclose(location, gold_location)
