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
"""
Base class for scattering theories.  Implements python-based
calc_intensity and calc_holo, based on subclass's calc_field

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
"""

from holopy.core.holopy_object import SerializableMetaclass
from holopy.core import Optics, Schema, Image, VectorGrid, interpret_args
from holopy.core.metadata import vector
from holopy.core.helpers import dict_without, is_none
from holopy.scattering import Mie, Multisphere, Sphere, Spheres
from holopy.scattering.theory import dda
from holopy.scattering.errors import AutoTheoryFailed, MissingParameter

import numpy as np

def check_schema(schema):
    if schema.optics.wavelen is None:
        raise MissingParameter("wavelength")
    if schema.optics.index is None:
        raise MissingParameter("medium refractive index")
    if is_none(schema.optics.polarization):
        raise MissingParameter("polarization")
    return schema

def interpret_theory(scatterer,theory='auto'):
    if isinstance(theory, str) and theory == 'auto':
        theory = determine_theory(scatterer)
    if isinstance(theory, SerializableMetaclass):
        theory = theory()
    return theory


def determine_theory(scatterer):
    if isinstance(scatterer, Sphere):
        return Mie()
    elif isinstance(scatterer, Spheres):
        return Multisphere()
    elif isinstance(scatterer, dda.scatterers_handled):
        return dda.DDA()
    else:
        raise AutoTheoryFailed(scatterer)

def calc_intensity(schema, scatterer, medium_index=None, wavelen=None, polarization=None, theory='auto', optics=None):
    """
    Calculate intensity at a location or set of locations

    Parameters
    ----------
    scatterer : :class:`.scatterer` object
        (possibly composite) scatterer for which to compute scattering
    medium_index : float or complex
        Refractive index of the medium in which the scatter is imbedded
    locations : np.ndarray or :class:`.locations` object
        The locations to compute intensity at
    wavelen : float or ndarray(float)
        Wavelength of illumination light. If wavelen is an array result
        will add a dimension and have all wavelengths
    optics : :class:`.Optics` object (optional)
        Object describing the optical train of illumination before the object
        and after the object until detection
    theory : :class:`.theory` object (optional)
        Scattering theory object to use for the calculation. This is optional
        if there is a clear choice of theory for your scatterer. If there is not
        a clear choice, calc_intensity will error out and ask you to specify a theory
    Returns
    -------
    inten : :class:`.Image`
        scattered intensity
    """
    field = calc_field(schema, scatterer, medium_index, wavelen, polarization, theory, optics)
    return (abs(field*(1-schema.normals))**2).sum(dim=vector)


def calc_holo(schema, scatterer, medium_index=None, wavelen=None, polarization=None, theory='auto', optics=None, scaling=1.0):
    """
    Calculate hologram formed by interference between scattered
    fields and a reference wave

    Parameters
    ----------
    scatterer : :class:`.scatterer` object
        (possibly composite) scatterer for which to compute scattering
     medium_index : float or complex
        Refractive index of the medium in which the scatter is imbedded
    locations : np.ndarray or :class:`.locations` object
        The locations to compute hologram at
    wavelen : float or ndarray(float)
        Wavelength of illumination light. If wavelen is an array result
        will add a dimension and have all wavelengths
    optics : :class:`.Optics` object (optional)
        Object describing the optical train of illumination before the object
        and after the object until detection
    theory : :class:`.theory` object (optional)
        Scattering theory object to use for the calculation. This is optional
        if there is a clear choice of theory for your scatterer. If there is not
        a clear choice, calc_intensity will error out and ask you to specify a theory
    scaling : scaling value (alpha) for intensity of reference wave

    Returns
    -------
    holo : :class:`.Image` object
        Calculated hologram from the given distribution of spheres
    """
    schema = check_schema(interpret_args(schema, medium_index, wavelen, polarization, optics))
    theory = interpret_theory(scatterer,theory)
    scat = theory._calc_field(scatterer, schema)
    return scattered_field_to_hologram(scat*scaling, schema.optics.polarization_field, schema.normals)

def calc_cross_sections(scatterer, medium_index=None, wavelen=None, polarization=None, theory='auto', optics=None):
    """
    Calculate scattering, absorption, and extinction
    cross sections, and asymmetry parameter <cos \theta>.

    Parameters
    ----------
    scatterer : :class:`.scatterer` object
        (possibly composite) scatterer for which to compute scattering
    medium_index : float or complex
        Refractive index of the medium in which the scatter is imbedded
    wavelen : float or ndarray(float)
        Wavelength of illumination light. If wavelen is an array result
        will add a dimension and have all wavelengths
    optics : :class:`.Optics` object (optional)
        Object describing the optical train of illumination before the object
        and after the object until detection
    theory : :class:`.theory` object (optional)
        Scattering theory object to use for the calculation. This is optional
        if there is a clear choice of theory for your scatterer. If there is not
        a clear choice, calc_intensity will error out and ask you to specify a theory
 
    Returns
    -------
    cross_sections : array (4)
        Dimensional scattering, absorption, and extinction
        cross sections, and <cos theta>
    """
    schema = check_schema(interpret_args(index=medium_index, wavelen=wavelen, polarization=polarization, optics=optics))
    theory = interpret_theory(scatterer,theory)
    return theory._calc_cross_sections(scatterer, schema.optics)

def calc_scat_matrix(schema, scatterer, medium_index=None, wavelen=None, theory='auto', optics=None):
    """
    Compute farfield scattering matricies for scatterer

    Parameters
    ----------
    scatterer : :class:`holopy.scattering.scatterer` object
        (possibly composite) scatterer for which to compute scattering
    medium_index : float or complex
        Refractive index of the medium in which the scatter is imbedded
    wavelen : float or ndarray(float)
        Wavelength of illumination light. If wavelen is an array result
        will add a dimension and have all wavelengths
    optics : :class:`.Optics` object (optional)
        Object describing the optical train of illumination before the object
        and after the object until detection
    theory : :class:`.theory` object (optional)
        Scattering theory object to use for the calculation. This is optional
        if there is a clear choice of theory for your scatterer. If there is not
        a clear choice, calc_intensity will error out and ask you to specify a theory

    Returns
    -------
    scat_matr : :class:`.Marray`
        Scattering matricies at specified positions

    """
    schema = interpret_args(schema, medium_index, wavelen, optics=optics)
    if schema.optics.wavelen is None:
        raise MissingParameter("wavelength")
    if schema.optics.index is None:
        raise MissingParameter("medium refractive index")

    theory = interpret_theory(scatterer,theory)
    return theory.calc_scat_matrix(scatterer, schema)

def calc_field(schema, scatterer, medium_index=None, wavelen=None, polarization=None, theory='auto', optics=None):
    """
    Calculate hologram formed by interference between scattered
    fields and a reference wave

    Parameters
    ----------
    scatterer : :class:`.scatterer` object
        (possibly composite) scatterer for which to compute scattering
     medium_index : float or complex
        Refractive index of the medium in which the scatter is imbedded
    locations : np.ndarray or :class:`.locations` object
        The locations to compute hologram at
    wavelen : float or ndarray(float)
        Wavelength of illumination light. If wavelen is an array result
        will add a dimension and have all wavelengths
    optics : :class:`.Optics` object (optional)
        Object describing the optical train of illumination before the object
        and after the object until detection
    theory : :class:`.theory` object (optional)
        Scattering theory object to use for the calculation. This is optional
        if there is a clear choice of theory for your scatterer. If there is not
        a clear choice, calc_intensity will error out and ask you to specify a theory
    scaling : scaling value (alpha) for intensity of reference wave

    Returns
    -------
    e_field : :class:`.Vector` object
        Calculated hologram from the given distribution of spheres
    """
    schema = check_schema(interpret_args(schema, medium_index, wavelen, polarization, optics))
    theory = interpret_theory(scatterer,theory)
    return theory._calc_field(scatterer, schema)

# this is pulled out separate from the calc_holo method because occasionally you
# want to turn prepared  e_fields into holograms directly
def scattered_field_to_hologram(scat, ref, detector_normal = None):
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
    if detector_normal is None:
        detector_normal = (0, 0, 1)

    holo = (np.abs(scat+ref)**2 * (1 - detector_normal)).sum(dim=vector)

    return holo

def _field_scalar_shape(e):
    # this is a clever hack with list arithmetic to get [1, 3] or [1,
    # 1, 3] as needed
    return [1]*(e.ndim-1) + [3]


