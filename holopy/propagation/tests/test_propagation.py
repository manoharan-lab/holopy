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
from ...core import ImageSchema, VolumeSchema, Optics
from ...scattering.theory import Mie
from ...scattering.scatterer import Sphere
from .. import propagate
from ...core.tests.common import assert_obj_close, verify

class test_propagation():
    def __init__(self):
        self.optics = Optics(.66, 1.33, (1,0))
        self.im_schema = ImageSchema(spacing = .1, shape = 100,
                                     optics = self.optics)
        self.sphere = Sphere(n = 1.59, r = .5, center = (5, 5, 5))
        self.holo = Mie.calc_holo(self.sphere, self.im_schema)

    def test_propagate_volume(self):
        vol_schema = VolumeSchema(shape = (40, 40, 25), spacing = .2,
                                  optics = self.optics)
        vol_schema.center = (5, 5, 7.5)

        vol = propagate(self.holo, vol_schema)
        verify(vol, 'propagate_into_volume')


    def test_d_vs_schema(self):
        d = np.arange(5, 10, 1)
        vol = VolumeSchema(shape = np.append(self.holo.shape, len(d)),
                spacing = np.append(self.holo.positions.spacing, 1))
        vol.center = (5, 5, 7.5)

        r1 = propagate(self.holo, d)
        r2 = propagate(self.holo, vol)
        assert_obj_close(r1, r2, context = 'propagated_volume')

    def test_propagate_e_field(self):
        e = Mie.calc_field(self.sphere, self.im_schema)
        prop_e = propagate(e, 10)
        verify(prop_e, 'propagate_e_field')
