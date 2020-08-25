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

'''
Modules for defining different types of scatterers, including
scattering primitives such as Spheres, and more complex objects such
as Clusters.

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Thomas G. Dimiduk <tdimidusk@physics.harvard.edu>
'''


from holopy.scattering.scatterer.scatterer import Scatterer, Indicators
from holopy.scattering.scatterer.sphere import Sphere, LayeredSphere
from holopy.scattering.scatterer.composite import Scatterers
from holopy.scattering.scatterer.spherecluster import Spheres, RigidCluster
from holopy.scattering.scatterer.janus import (JanusSphere_Uniform,
                                               JanusSphere_Tapered)
from holopy.scattering.scatterer.spheroid import Spheroid
from holopy.scattering.scatterer.ellipsoid import Ellipsoid
from holopy.scattering.scatterer.capsule import Capsule
from holopy.scattering.scatterer.cylinder import Cylinder
from holopy.scattering.scatterer.bisphere import Bisphere
from holopy.scattering.scatterer.csg import (Union, Difference, Intersection,
                                             CsgScatterer)
