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
'''
Routines for image processing.  Useful for pre-processing raw
holograms prior to extracting final data or post-processing
reconstructions.

.. moduleauthor:: Ryan McGorty <mcgorty@fas.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Tom G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>
'''


from .img_proc import normalize, detrend, zero_filter, subimage, add_noise, fft, ifft
from .centerfinder import center_find, centered_subimage, hough, image_gradient
from .math import cartesian_distance, rotate_points, rotation_matrix,simulate_noise
from .utilities import _ensure_array, ensure_listlike, _ensure_pair, mkdir_p
from .utilities import dict_without, is_none, updated,  get_values
from .utilities import copy_metadata, flat, from_flat, make_subset_data
from .math import chisq, rsq
