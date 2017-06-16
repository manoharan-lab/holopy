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

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
"""

from ..core.holopy_object import SerializableMetaclass
from ..core.metadata import vector, update_metadata, to_vector, copy_metadata, from_flat, detector_points
from ..core.utils import dict_without, is_none
from .scatterer import Sphere, Spheres
from .errors import AutoTheoryFailed, MissingParameter

try:
    from .theory import Mie, Multisphere
    from .theory.dda import DDA
except:
    pass

import numpy as np
from warnings import warn

def check_schema(schema, pol = True):
    if schema.illum_wavelen is None:
        raise MissingParameter("wavelength")
    if schema.medium_index is None:
        raise MissingParameter("medium refractive index")
    if pol is not False and is_none(schema.illum_polarization):
        raise MissingParameter("polarization")
    return schema

def prep_schema(schema, medium_index, illum_wavelen, illum_polarization):
    return check_schema(update_metadata(schema, medium_index, illum_wavelen, illum_polarization), illum_polarization)

def interpret_theory(scatterer,theory='auto'):
    if isinstance(theory, str) and theory == 'auto':
        theory = determine_theory(scatterer.guess())
    if isinstance(theory, SerializableMetaclass):
        theory = theory()
    return theory

def finalize(schema, result):
    if not hasattr(schema, 'flat'):
        result = from_flat(result)
    return copy_metadata(schema, result, do_coords=False)

def determine_theory(scatterer):
    if isinstance(scatterer, Sphere):
        return Mie()
    elif isinstance(scatterer, Spheres):
        if all([np.isscalar(scat.r) for i,scat in enumerate(scatterer.scatterers)]):
            return Multisphere()
        else:
            warn("HoloPy's multisphere theory can't handle coated spheres. Using Mie theory.")
            return Mie()
    elif DDA()._can_handle(scatterer):
        return DDA()
    else:
        raise AutoTheoryFailed(scatterer)

def calc_intensity(schema, scatterer, medium_index=None, illum_wavelen=None, illum_polarization=None, theory='auto'):
    """
    Calculate intensity at a location or set of locations

    Parameters
    ----------
    scatterer : :class:`.scatterer` object
        (possibly composite) scatterer for which to compute scattering
    medium_index : float or complex
        Refractive index of the medium in which the scatter is imbedded
    illum_wavelen : float or ndarray(float)
        Wavelength of illumination light. If illum_wavelen is an array result
        will add a dimension and have all wavelengths
    theory : :class:`.theory` object (optional)
        Scattering theory object to use for the calculation. This is optional
        if there is a clear choice of theory for your scatterer. If there is not
        a clear choice, calc_intensity will error out and ask you to specify a theory
    Returns
    -------
    inten : :class:`.Image`
        scattered intensity
    """
    field = calc_field(schema, scatterer, medium_index=medium_index, illum_wavelen=illum_wavelen, illum_polarization=illum_polarization, theory=theory)
    return finalize(schema, (abs(field*(1-schema.normals))**2).sum(dim=vector))


def calc_holo(schema, scatterer, medium_index=None, illum_wavelen=None, illum_polarization=None, theory='auto', scaling=1.0):
    """
    Calculate hologram formed by interference between scattered
    fields and a reference wave

    Parameters
    ----------
    scatterer : :class:`.scatterer` object
        (possibly composite) scatterer for which to compute scattering
    medium_index : float or complex
        Refractive index of the medium in which the scatter is imbedded
    illum_wavelen : float or ndarray(float)
        Wavelength of illumination light. If illum_wavelen is an array result
        will add a dimension and have all wavelengths
    theory : :class:`.theory` object (optional)
        Scattering theory object to use for the calculation. This is optional
        if there is a clear choice of theory for your scatterer. If there is not
        a clear choice, calc_intensity will error out and ask you to specify a theory
    scaling : scaling value (alpha) for amplitude of reference wave

    Returns
    -------
    holo : :class:`.Image` object
        Calculated hologram from the given distribution of spheres
    """
    if hasattr(scaling, 'guess'):
        scaling = scaling.guess

    theory = interpret_theory(scatterer,theory)
    uschema = prep_schema(schema, medium_index, illum_wavelen, illum_polarization)
    scat = theory._calc_field(scatterer.guess(), uschema)
    holo = scattered_field_to_hologram(scat*scaling, uschema.illum_polarization, uschema.normals)
    return finalize(uschema, holo)

def calc_cross_sections(scatterer, medium_index=None, illum_wavelen=None, illum_polarization=None, theory='auto'):
    """
    Calculate scattering, absorption, and extinction
    cross sections, and asymmetry parameter <cos \theta>.

    Parameters
    ----------
    scatterer : :class:`.scatterer` object
        (possibly composite) scatterer for which to compute scattering
    medium_index : float or complex
        Refractive index of the medium in which the scatter is imbedded
    illum_wavelen : float or ndarray(float)
        Wavelength of illumination light. If illum_wavelen is an array result
        will add a dimension and have all wavelengths
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
    theory = interpret_theory(scatterer,theory)
    return theory._calc_cross_sections(scatterer=scatterer.guess(), medium_wavevec=2*np.pi/(illum_wavelen/medium_index), medium_index=medium_index, illum_polarization=to_vector(illum_polarization))

def calc_scat_matrix(schema, scatterer, medium_index=None, illum_wavelen=None, theory='auto'):
    """
    Compute farfield scattering matricies for scatterer

    Parameters
    ----------
    scatterer : :class:`holopy.scattering.scatterer` object
        (possibly composite) scatterer for which to compute scattering
    medium_index : float or complex
        Refractive index of the medium in which the scatter is imbedded
    illum_wavelen : float or ndarray(float)
        Wavelength of illumination light. If illum_wavelen is an array result
        will add a dimension and have all wavelengths
    theory : :class:`.theory` object (optional)
        Scattering theory object to use for the calculation. This is optional
        if there is a clear choice of theory for your scatterer. If there is not
        a clear choice, calc_intensity will error out and ask you to specify a theory

    Returns
    -------
    scat_matr : :class:`.Marray`
        Scattering matricies at specified positions

    """
    theory = interpret_theory(scatterer,theory)
    uschema=prep_schema(schema, medium_index=medium_index, illum_wavelen=illum_wavelen, illum_polarization = False)
    return finalize(uschema, theory._calc_scat_matrix(scatterer.guess(), uschema))

def calc_field(schema, scatterer, medium_index=None, illum_wavelen=None, illum_polarization=None, theory='auto'):
    """
    Calculate hologram formed by interference between scattered
    fields and a reference wave

    Parameters
    ----------
    scatterer : :class:`.scatterer` object
        (possibly composite) scatterer for which to compute scattering
    medium_index : float or complex
        Refractive index of the medium in which the scatter is imbedded
    illum_wavelen : float or ndarray(float)
        Wavelength of illumination light. If illum_wavelen is an array result
        will add a dimension and have all wavelengths
    theory : :class:`.theory` object (optional)
        Scattering theory object to use for the calculation. This is optional
        if there is a clear choice of theory for your scatterer. If there is not
        a clear choice, calc_intensity will error out and ask you to specify a theory

    Returns
    -------
    e_field : :class:`.Vector` object
        Calculated hologram from the given distribution of spheres
    """
    theory = interpret_theory(scatterer,theory)
    uschema = prep_schema(schema, medium_index=medium_index, illum_wavelen=illum_wavelen, illum_polarization=illum_polarization)
    return finalize(uschema, theory._calc_field(scatterer.guess(), uschema))

# this is pulled out separate from the calc_holo method because occasionally you
# want to turn prepared  e_fields into holograms directly
def scattered_field_to_hologram(scat, ref, normals):
    """
    Calculate a hologram from an E-field

    Parameters
    ----------
    scat : :class:`.VectorGrid`
        The scattered (object) field
    ref : xarray[vector]]
        The reference field
    detector_normal : (float, float, float)
        Vector normal to the detector the hologram should be measured at
        (defaults to z hat, a detector in the x, y plane)
    """
    holo = (np.abs(scat+ref)**2 * (1 - normals)).sum(dim=vector)

    return holo

def _field_scalar_shape(e):
    # this is a clever hack with list arithmetic to get [1, 3] or [1,
    # 1, 3] as needed
    return [1]*(e.ndim-1) + [3]
