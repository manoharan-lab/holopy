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
"""
Base class for scattering theories.  Implements python-based
calc_intensity and calc_holo, based on subclass's calc_field
.. moduleauthor:: Jerome Fung <jerome.fung@post.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Brian Leahy <bleahy@g.harvard.edu>
"""
import numpy as np

from holopy.core.holopy_object import HoloPyObject
try:
    from holopy.scattering.theory.mie_f import mieangfuncs
except ImportError:
    pass


class ScatteringTheory(HoloPyObject):
    """
    Defines common interface for all scattering theories.

    Subclasses must implement:
    * _can_handle
    * _raw_fields or _raw_scat_matrs or both.
    * (optionally) _raw_cross_sections,

    Notes
    -----
    A subclasses that do the work of computing scattering should do it
    by implementing _raw_fields and/or _raw_scat_matrs and (optionally)
    _raw_cross_sections. _raw_cross_sections is needed only for
    calc_cross_sections. Either of _raw_fields or _raw_scat_matrs will
    give you calc_holo, calc_field, and calc_intensity. Obviously
    calc_scat_matrix will only work if you implement _raw_cross_sections.
    So the simplest thing is to just implement _raw_scat_matrs. You only
    need to do _raw_fields there is a way to compute it more efficently
    and you care about that speed, or if it is easier and you don't care
    about matrices.
    """
    desired_coordinate_system = 'spherical'

    def _can_handle(self, scatterer):
        raise NotImplementedError

    def _raw_scat_matrs(self, scatterer, pos, medium_wavevec, medium_index):
        raise NotImplementedError

    def _raw_cross_sections(self, *args, **kwargs):
        raise NotImplementedError

    def _raw_fields(self, pos, scatterer, medium_wavevec, medium_index,
                    illum_polarization):
        scat_matr = self._raw_scat_matrs(
            scatterer, pos, medium_wavevec=medium_wavevec,
            medium_index=medium_index)

        fields = np.zeros_like(pos.T, dtype=np.array(scat_matr).dtype)
        for i, point in enumerate(pos.T):
            kr, theta, phi = point
            escat_sph = mieangfuncs.calc_scat_field(
                kr, phi, scat_matr[i], illum_polarization.values[:2])
            fields[i] = mieangfuncs.fieldstocart(escat_sph, theta, phi)
        return fields.T

