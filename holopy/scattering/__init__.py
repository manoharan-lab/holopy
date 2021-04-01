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
'''Scattering calculations

The scattering package provides objects and methods to define
scatterer geometries, and theories to compute scattering from
specified geometries.  Scattering depends on holopy.core, and certain
scattering theories may require external scattering codes.

The HoloPy scattering module is used to:

1. Describe geometry as a :mod:`~holopy.scattering.scatterer` object
2. Define the result you want as a xarray.DataArray xarray.DataArray
3. Calculate scattering quantities with an
   :mod:`~holopy.scattering.theory` appropriate for your
   :mod:`~holopy.scattering.scatterer`

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Jerome Fung <jerome.fung@post.harvard.edu>
.. moduleauthor:: Ryan McGorty <mcgorty@fas.harvard.edu>
.. moduleauthor:: Rebecca W. Perry <rperry@seas.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>

'''

from holopy.scattering import scatterer, theory
from holopy.scattering.scatterer import (Scatterer, Scatterers, Sphere,
    LayeredSphere, Spheres, RigidCluster, Ellipsoid, Capsule, Cylinder,
    Bisphere, Spheroid, JanusSphere_Uniform, JanusSphere_Tapered)
from holopy.scattering.interface import (calc_holo, calc_field,
    calc_intensity, calc_cross_sections, calc_scat_matrix)
from holopy.scattering.theory import (
    Mie, MieLens, AberratedMieLens, Multisphere, DDA, Tmatrix)
