# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, and Ryan McGorty, Anna Wang, Solomon Barkley
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

import numpy as np
from holopy.core import detector_grid, update_metadata
from holopy.scattering.scatterer import Sphere
from holopy.scattering.theory.scatteringtheory import ScatteringTheory

wavelen = 658e-9
ypolarization = [0., 1.0]  # y-polarized
xpolarization = [1.0, 0.]  # x-polarized
pixel_scale = [.1151e-6, .1151e-6]
index = 1.33

xschema = update_metadata(detector_grid(shape=128, spacing=pixel_scale),
                          illum_wavelen=wavelen, medium_index=index,
                          illum_polarization=xpolarization)
yschema = update_metadata(xschema, illum_polarization=ypolarization)

xschema_lens = update_metadata(detector_grid(shape=32, spacing=pixel_scale),
                               illum_wavelen=wavelen, medium_index=index,
                               illum_polarization=xpolarization)

scaling_alpha = .6
radius = .85e-6
n_particle_real = 1.59
n_particle_imag = 1e-4
n = n_particle_real + n_particle_imag * 1.0j
x = .576e-05
y = .576e-05
z = 15e-6

sphere = Sphere(n=n_particle_real + n_particle_imag*1j, r=radius,
                center=(x, y, z))

class MockTheory(ScatteringTheory):
    """Minimally-functional daughter of ScatteringTheory for fast tests."""
    def __init__(*args, **kwargs):
        pass  # an init is necessary for the repr

    def can_handle(self, scatterer):
        return isinstance(scatterer, Sphere)

    def raw_fields(self, positions, *args, **kwargs):
        return np.ones(positions.shape, dtype='complex128')


class MockScatteringMatrixBasedTheory(ScatteringTheory):
    """Minimally-functional daughter of ScatteringTheory which
    uses the scattering matrix pathway, for fast tests.
    Smells like a Rayleigh scatterer, just for fun. But it's not"""
    def __init__(*args, **kwargs):
        pass  # an init is necessary for the repr

    def can_handle(self, scatterer):
        return isinstance(scatterer, Sphere)

    def raw_scat_matrs(self, scatterer, positions, *args, **kwargs):
        strength = scatterer.n * scatterer.r
        scattering_matrix = np.array(
            [np.eye(2) for _ in range(positions.shape[1])])
        return strength * scattering_matrix.astype('complex128')
