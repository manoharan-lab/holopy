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

import numpy as np
from ...core import ImageTarget, VolumeTarget, Optics, Grid
from ...scattering.theory import Mie
from ...scattering.scatterer import Sphere
from .. import propagate
from ...core.tests.common import assert_obj_close

class test_propagation():
    def __init__(self):
        self.optics = Optics(.66, 1.33)
        im_target = ImageTarget(pixel_size = .1, shape = 100, optics = self.optics)
        sphere = Sphere(n = 1.59, center = (5, 5, 5))
        self.holo = Mie.calc_holo(sphere, im_target)

    def test_propagate_volume(self):
        vol_target = VolumeTarget(positions = Grid(shape = (50, 50, 50),
                                                   spacing = .75),
                                  optics = self.optics, center = (5, 5, 5))
        
        vol = propagate(self.holo, target = vol_target)
                    
    
        
    def test_d_vs_target(self):
        d = np.arange(5, 10, 1)
        vol = VolumeTarget(positions = Grid(
                shape = np.append(self.holo.shape, len(d)),
                spacing = np.append(self.holo.positions.spacing, 1)),
                           center = (5, 5, 7))
        

        r1 = propagate(self.holo, d)
        r2 = propagate(self.holo, vol)
        assert_obj_close(r1, r2)
