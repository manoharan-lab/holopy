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
from __future__ import division

import os
import numpy as np

from ...core import Optics

from ...core.tests.common import assert_allclose, assert_obj_close, verify

wavelen = 658e-9
ypolarization = [0., 1.0] # y-polarized
xpolarization = [1.0, 0.] # x-polarized
divergence = 0
pixel_scale = [.1151e-6, .1151e-6]
index = 1.33

yoptics = Optics(wavelen=wavelen, index=index,
                 pixel_scale=pixel_scale,
                 polarization=ypolarization,
                 divergence=divergence)
    
xoptics = Optics(wavelen=wavelen, index=index,
                 pixel_scale=pixel_scale,
                 polarization=xpolarization,
                 divergence=divergence)

optics=yoptics


