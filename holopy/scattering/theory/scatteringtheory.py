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
"""
Base class for scattering theories.  Implements python-based
calc_intensity and calc_holo, based on subclass's calc_field

.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
"""

import numpy as np
from warnings import warn
from ...core.marray import make_vector_schema
from holopy.core.holopy_object import HoloPyObject
from ..scatterer import Scatterers, Sphere
from ..errors import TheoryNotCompatibleError, MissingParameter

class ScatteringTheory(HoloPyObject):
    """
    Defines common interface for all scattering theories.

    Notes
    -----
    A subclasses that do the work of computing scattering should do it by
    implementing a _calc_field(self, scatterer, schema) function that returns a
    VectorGrid electric field.
    """

    def _calc_field(self, scatterer, schema):
        """
        Calculate fields.  Implemented in derived classes only.

        Parameters
        ----------
        scatterer : :mod:`.scatterer` object
            (possibly composite) scatterer for which to compute scattering

        Returns
        -------
        e_field : :mod:`.VectorGrid`
            scattered electric field
        """
        raise NotImplemented

    def calc_scat_matrix(cls_self, scatterer, schema):
        """
        Compute scattering matricies for scatterer

        Parameters
        ----------
        scatterer : :mod:`holopy.scattering.scatterer` object
            (possibly composite) scatterer for which to compute scattering

        Returns
        -------
        scat_matr : :mod:`.Marray`
            Scattering matricies at specified positions

        Notes
        -----
        calc_* functions can be called on either a theory class or a theory
        object.  If called on a theory class, they use a default theory object
        which is correct for the vast majority of situations.  You only need to
        instantiate a theory object if it has adjustable parameters and you want
        to use non-default values.
        """

        return cls_self._calc_scat_matrix(scatterer, schema)

    def _finalize_fields(self, z, fields, schema):
        # expects fields as an Nx3 ndarray
        phase = np.exp(-1j*np.pi*2*z / schema.optics.med_wavelen)
        schema = make_vector_schema(schema)
        result = schema.interpret_1d(fields)
        return result * phase


# Subclass of scattering theory, overrides functions that depend on array
# ordering and handles the tranposes for sending values to/from fortran
class FortranTheory(ScatteringTheory):
    def _calc_field(self, scatterer, schema):
        optics=schema.optics
        def get_field(s):
            if isinstance(scatterer,Sphere) and scatterer.center is None:
                raise MissingParameter("center location")

            positions = schema.positions.kr_theta_phi(s.center, optics.wavevec)
            field = np.vstack(self._raw_fields(positions.T, s, optics)).T
            phase = np.exp(-1j*np.pi*2*s.center[2] / optics.med_wavelen)
            if self._scatterer_overlaps_schema(scatterer, schema):
                inner = scatterer.contains(schema.positions.xyz())
                field[inner] = np.vstack(
                    self._raw_internal_fields(positions[inner].T, s,
                                              optics)).T
            field *= phase
            return make_vector_schema(schema).interpret_1d(field)


        # See if we can handle the scatterer in one step
        if self._can_handle(scatterer):
            field = get_field(scatterer)
        elif isinstance(scatterer, Scatterers):
        # if it is a composite, try superposition
            scatterers = scatterer.get_component_list()
            field = get_field(scatterers[0])
            for s in scatterers[1:]:
                field += get_field(s)
        else:
            raise TheoryNotCompatibleError(self, scatterer)

        return field
        # TODO: fix internal field and re-enable this
#        return self._set_internal_fields(field, scatterer)


    def _scatterer_overlaps_schema(self, scatterer, schema):
        if hasattr(schema, 'center'):
            center_to_center = scatterer.center - schema.center
            unit_vector = center_to_center - abs(center_to_center).sum()
            return schema.contains(scatterer.center - unit_vector)
        else:
            return scatterer.contains(schema.positions.xyz()).any()

    def _set_internal_fields(self, fields, scatterer):
        center_to_center = scatterer.center - fields.center
        unit_vector = center_to_center - abs(center_to_center).sum()
        if fields.contains(scatterer.center - unit_vector):
            warn("Fields inside your Sphere(s) set to 0 because {0} Theory "
                 " does not yet support calculating internal fields".format(
                     self.__class__.__name__))

            origin = fields.origin
            extent = fields.extent
            shape = fields.shape
            def points(i):
                return enumerate(np.linspace(origin[i], origin[0]+extent[i], shape[i]))
            # TODO: may be missing hitting a point or two because of
            # integer truncation, see about fixing that


            # TODO: vectorize or otherwise speed this up
            if len(shape) == 2:
                for i, x in points(0):
                    for j, y in points(1):
                        if scatterer.contains((x, y, fields.center[2])):
                            fields[i, j] = 0

            else:
                for i, x in points(0):
                    for j, y in points(1):
                        for k, z in points(2):
                            if scatterer.contains((x, y, z)):
                                fields[i, j, k] = 0
        return fields

class InvalidElectricFieldComputation(Exception):
    def __init__(self, reason):
        self.reason = reason
    def __str__(self):
        return "Invalid Electric Computation: " + self.reason
