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


from holopy.core import detector_grid, update_metadata
from holopy.scattering.scatterer import Sphere

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
