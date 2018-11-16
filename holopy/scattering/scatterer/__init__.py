# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
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

'''
Modules for defining different types of scatterers, including
scattering primitives such as Spheres, and more complex objects such
as Clusters.

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Thomas G. Dimiduk <tdimidusk@physics.harvard.edu>
'''


from .scatterer import (Scatterer, Indicators, _expand_parameters,
                         _interpret_parameters)
from .sphere import Sphere, LayeredSphere
from .composite import Scatterers
from .spherecluster import Spheres, RigidCluster
from .janus import JanusSphere_Uniform, JanusSphere_Tapered
from .spheroid import Spheroid
from .ellipsoid import Ellipsoid
from .capsule import Capsule
from .cylinder import Cylinder
from .bisphere import Bisphere
from .csg import Union, Difference, Intersection, CsgScatterer
