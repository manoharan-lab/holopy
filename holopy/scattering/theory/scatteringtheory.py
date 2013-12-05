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
from ...core.marray import Image, VectorGrid, VectorSchema, dict_without, make_vector_schema
from ...core import Optics
from ...core.holopy_object import HoloPyObject
from ..binding_method import binding, finish_binding
from ..scatterer import Sphere, Scatterers
from ..errors import NoCenter, NoPolarization, TheoryNotCompatibleError

class ScatteringTheory(HoloPyObject):
    """
    Defines common interface for all scattering theories.

    Notes
    -----
    A subclasses that do the work of computing scattering should do it by
    implementing a _calc_field(self, scatterer, schema) function that returns a
    VectorGrid electric field.

    ScatteringTheory uses pseudo classmethods which when called on a
    ScatteringTheory class are in fact called on a default instantiation
    (no parameters given to the constructor).  If you manually instantiate a
    ScatteringTheory Object then it's calc_* methods refer to itself.
    """

    def __init__(self):
        # If the user instantiates a theory, we need to replace the classmethods
        # that instantiate an object with normal methods that reference the
        # theory object
        finish_binding(self)

    @classmethod
    @binding
    def calc_field(cls_self, scatterer, schema, scaling = 1.0):
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


        Notes
        -----
        calc_* functions can be called on either a theory class or a theory
        object.  If called on a theory class, they use a default theory object
        which is correct for the vast majority of situations.  You only need to
        instantiate a theory object if it has adjustable parameters and you want
        to use non-default values.
        """
        fields = cls_self._calc_field(scatterer, schema) * scaling
        fields.get_metadata_from(schema)
        return fields


    @classmethod
    @binding
    def calc_intensity(cls_self, scatterer, schema, scaling = 1.0):
        """
        Calculate intensity at focal plane (z=0)

        Parameters
        ----------
        scatterer : :mod:`.scatterer` object
            (possibly composite) scatterer for which to compute scattering

        Returns
        -------
        inten : :mod:`.Image`
            scattered intensity

        Notes
        -----
        calc_* functions can be called on either a theory class or a theory
        object.  If called on a theory class, they use a default theory object
        which is correct for the vast majority of situations.  You only need to
        instantiate a theory object if it has adjustable parameters and you want
        to use non-default values.

        Total scattered intensity only takes into account the x- and
        y-components of the E-field.  The z-component is ignored
        because the detector's pixels should be sensitive to the z
        component of the Poynting vector, E x B, and the z component
        of E x B cannot depend on Ez.
        """
        field = cls_self.calc_field(scatterer, schema = schema, scaling = scaling)
        normal = np.array([0, 0, 1])
        normal = normal.reshape(_field_scalar_shape(field))
        return (abs(field*(1-normal))**2).sum(-1)


    @classmethod
    @binding
    def calc_holo(cls_self, scatterer, schema, scaling=1.0):
        """
        Calculate hologram formed by interference between scattered
        fields and a reference wave

        Parameters
        ----------
        scatterer : :mod:`.scatterer` object
            (possibly composite) scatterer for which to compute scattering
        scaling : scaling value (alpha) for intensity of reference wave

        Returns
        -------
        holo : :class:`.Image` object
            Calculated hologram from the given distribution of spheres

        Notes
        -----
        calc_* functions can be called on either a theory class or a theory
        object.  If called on a theory class, they use a default theory object
        which is correct for the vast majority of situations.  You only need to
        instantiate a theory object if it has adjustable parameters and you want
        to use non-default values.
        """

        if isinstance(scatterer, Sphere) and scatterer.center == None:
            raise NoCenter("Center is required for hologram calculation of a sphere")
        else:
            pass

        if schema.optics.polarization.shape == (2,):
            pass
        else:
            raise NoPolarization("Polarization is required for hologram calculation")

        scat = cls_self.calc_field(scatterer, schema = schema, scaling = scaling)
        return scattered_field_to_hologram(scat, schema.optics)

    @classmethod
    @binding
    def calc_cross_sections(cls_self, scatterer, optics):
        """
        Calculate scattering, absorption, and extinction
        cross sections, and asymmetry parameter <cos \theta>.
        To be implemented by derived classes.

        Parameters
        ----------
        scatterer : :mod:`.scatterer` object
            (possibly composite) scatterer for which to compute scattering

        Returns
        -------
        cross_sections : array (4)
            Dimensional scattering, absorption, and extinction
            cross sections, and <cos theta>

        Notes
        -----
        calc_* functions can be called on either a theory class or a theory
        object.  If called on a theory class, they use a default theory object
        which is correct for the vast majority of situations.  You only need to
        instantiate a theory object if it has adjustable parameters and you want
        to use non-default values.
        """
        return cls_self._calc_cross_sections(scatterer, optics)

    @classmethod
    @binding
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
        def get_field(s):
            positions = schema.positions.kr_theta_phi(s.center, schema.optics)
            field = np.vstack(self._raw_fields(positions.T, s, schema.optics)).T
            phase = np.exp(-1j*np.pi*2*s.center[2] / schema.optics.med_wavelen)
            if self._scatterer_overlaps_schema(scatterer, schema):
                inner = scatterer.contains(schema.positions.xyz())
                field[inner] = np.vstack(
                    self._raw_internal_fields(positions[inner].T, s,
                                              schema.optics)).T
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

def _field_scalar_shape(e):
    # this is a clever hack with list arithmetic to get [1, 3] or [1,
    # 1, 3] as needed
    return [1]*(e.ndim-1) + [3]

# this is pulled out separate from the calc_holo method because occasionally you
# want to turn prepared  e_fields into holograms directly
def scattered_field_to_hologram(scat, ref, detector_normal = (0, 0, 1)):
    """
    Calculate a hologram from an E-field

    Parameters
    ----------
    scat : :class:`.VectorGrid`
        The scattered (object) field
    ref : :class:`.VectorGrid` or :class:`.Optics`
        The reference field, it can also be inferred from polarization of an
        Optics object
    detector_normal : (float, float, float)
        Vector normal to the detector the hologram should be measured at
        (defaults to z hat, a detector in the x, y plane)
    """
    shape = _field_scalar_shape(scat)
    if isinstance(ref, Optics):
        # add the z component to polarization and adjust the shape so that it is
        # broadcast correctly
        ref = VectorGrid(np.append(ref.polarization, 0).reshape(shape))
    detector_normal = np.array(detector_normal).reshape(shape)

    holo = Image(((abs(scat)**2 + abs(ref)**2 + 2* np.real(scat*ref)) *
                  (1 - detector_normal)).sum(axis=-1),
                 **dict_without(scat._dict, ['dtype', 'components']))

    return holo

class InvalidElectricFieldComputation(Exception):
    def __init__(self, reason):
        self.reason = reason
    def __str__(self):
        return "Invalid Electric Computation: " + self.reason
