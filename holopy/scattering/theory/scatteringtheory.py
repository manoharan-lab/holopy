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
    * can_handle
    * raw_fields or raw_scat_matrs or both.
    * (optionally) raw_cross_sections,

    Notes
    -----
    Subclasses should implement the following methods to create a
    scatteringtheory that works for the following user-facing methods:
    * `calc_holo`:              `raw_fields` or `raw_scat_matrs`
    * `calc_intensity`:         `raw_fields` or `raw_scat_matrs`
    * `calc_field`:             `raw_fields` or `raw_scat_matrs`
    * `calc_scat_matrix`:       `raw_scat_matrs`
    * `calc_cross_sections`:    `raw_cross_sections`

    By default, ScatteringTheories computer `raw_fields` from the
    `raw_scat_matrs`; over-ride the `raw_fields` method to compute the
    fields in a different way.
    """
    desired_coordinate_system = 'spherical'
    parameter_names = tuple()

    def __init__(self):
        # holopy's yaml functionality inspects the code, so we need an
        # init, even though it is empty.
        pass

    def can_handle(self, scatterer):
        """Given a scatterer, returns a bool"""
        raise NotImplementedError

    def raw_scat_matrs(self, scatterer, pos, medium_wavevec, medium_index):
        """Given a (3, N) array `pos` etc, returns an (N, 2, 2) array"""
        raise NotImplementedError

    def raw_cross_sections(
            self, scatterer, medium_wavevec, medium_index, illum_polarization):
        """Returns cross-sections, as an array [cscat, cabs, cext, asym]"""
        raise NotImplementedError

    def raw_fields(self, pos, scatterer, medium_wavevec, medium_index,
                    illum_polarization):
        """Given a (3, N) array `pos`, etc, returns a (3, N) array"""
        scat_matr = self.raw_scat_matrs(
            scatterer, pos, medium_wavevec=medium_wavevec,
            medium_index=medium_index)

        fields = np.zeros_like(pos.T, dtype=np.array(scat_matr).dtype)
        for i, point in enumerate(pos.T):
            kr, theta, phi = point
            escat_sph = mieangfuncs.calc_scat_field(
                kr, phi, scat_matr[i], illum_polarization.values[:2])
            fields[i] = mieangfuncs.fieldstocart(escat_sph, theta, phi)
        return fields.T

    @property
    def parameters(self):
        return {k: getattr(self, k) for k in self.parameter_names}

    def from_parameters(self, parameters):
        """Creates a ScatteringTheory like the current one, but with different
        parameters. Used for fitting

        Parameters
        ----------
        dict
            keys should be valid `self.parameter_names` fields, values
            should be the corresponding kwargs

        Returns
        -------
        ScatteringTheory instance, of the same class as `self`
        """
        kwargs = self._dict
        kwargs.update(parameters)
        return self.__class__(**kwargs)
