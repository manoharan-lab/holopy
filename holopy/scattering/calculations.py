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

from warnings import warn

import xarray as xr
import numpy as np

from holopy.core.holopy_object import SerializableMetaclass
from holopy.core.metadata import (
    vector, illumination, update_metadata, to_vector, copy_metadata, from_flat)
from holopy.core.utils import dict_without, ensure_array
from holopy.scattering.scatterer import (
    Sphere, Spheres, Spheroid, Cylinder, _expand_parameters,
    _interpret_parameters)
from holopy.scattering.errors import AutoTheoryFailed, MissingParameter
from holopy.scattering.theory import Mie, Multisphere
from holopy.scattering.theory import Tmatrix
from holopy.scattering.theory.dda import DDA


# Used in scattering.tests.test_2_color,
# and here in calc_holo, calc_scat_matrix, calc_field
def prep_schema(detector, medium_index, illum_wavelen, illum_polarization):
    detector = update_metadata(
        detector, medium_index, illum_wavelen, illum_polarization)

    if detector.illum_wavelen is None:
        raise MissingParameter("wavelength")
    if detector.medium_index is None:
        raise MissingParameter("medium refractive index")
    if illum_polarization is not False and detector.illum_polarization is None:
        raise MissingParameter("polarization")

    illum_wavelen = ensure_array(detector.illum_wavelen)
    illum_polarization = detector.illum_polarization

    if len(illum_wavelen) > 1 or ensure_array(illum_polarization).ndim == 2:
        #  multiple illuminations to calculate
        if illumination in illum_polarization.dims:
            if isinstance(illum_wavelen, xr.DataArray):
                pass
            else:
                if len(illum_wavelen) == 1:
                    illum_wavelen = illum_wavelen.repeat(
                        len(illum_polarization.illumination))
                illum_wavelen = xr.DataArray(
                    illum_wavelen, dims=illumination,
                    coords={illumination: illum_polarization.illumination})
        else:
            #  need to interpret illumination from detector.illum_wavelen
            if not isinstance(illum_wavelen, xr.DataArray):
                illum_wavelen = xr.DataArray(
                    illum_wavelen, dims=illumination,
                    coords={illumination: illum_wavelen})
            illum_polarization = xr.broadcast(
                illum_polarization, illum_wavelen, exclude=[vector])[0]

        if illumination in detector.dims:
            detector = detector.sel(
                illumination=detector.illumination[0], drop=True)
        detector = update_metadata(
            detector, illum_wavelen=illum_wavelen,
            illum_polarization=illum_polarization)

    return detector


# Used here in calc_holo, calc_cross_section, calc_scat_matrix, calc_field
def interpret_theory(scatterer, theory='auto'):
    if isinstance(theory, str) and theory == 'auto':
        theory = determine_default_theory_for(scatterer.guess)
    if isinstance(theory, SerializableMetaclass):
        theory = theory()
    return theory


# Used here in calc_intensity, calc_holo, calc_scat_matrix, calc_field
def finalize(detector, result):
    if not hasattr(detector, 'flat'):
        result = from_flat(result)
    return copy_metadata(detector, result, do_coords=False)


# Used in inference.model, but commented out
# Used in scattering.tests.tests_calculations, which just tests this function
# Used here in interpret_theory

# Some comments on why `determine_default_theory_for` exists, rather than each
# Scatterer class knowing what a good default theory is.
# The problem is that the theories (Mie etc) import Sphere to see if
# the theory can handle the scatterer, in the _can_handle method and
# others. Worse, since the DDA theory calls an external DDA library
# with specially-defined DDA objects, the DDA theory has a switch statement
# for basically every holopy scatterer. So right now the scatterers can't
# have a default theory and/or valid theory attr, as this causes a dependency
# loop.
def determine_default_theory_for(scatterer):
    if isinstance(scatterer, Sphere):
        theory = Mie()
    elif isinstance(scatterer, Spheres):
        if all([np.isscalar(scat.r) for scat in scatterer.scatterers]):
            theory = Multisphere()
        else:
            warn("HoloPy's multisphere theory can't handle coated spheres." +
                 "Using Mie theory.")
            theory = Mie()
    elif isinstance(scatterer, Spheroid) or isinstance(scatterer, Cylinder):
        theory = Tmatrix()
    elif DDA()._can_handle(scatterer):
        theory = DDA()
    else:
        raise AutoTheoryFailed(scatterer)
    return theory


def calc_intensity(detector, scatterer, medium_index=None, illum_wavelen=None,
                   illum_polarization=None, theory='auto'):
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
        Scattering theory object to use for the calculation. This is
        optional if there is a clear choice of theory for your scatterer.
        If there is not a clear choice, calc_intensity will error out and
        ask you to specify a theory
    Returns
    -------
    inten : xarray.DataArray
        scattered intensity
    """
    field = calc_field(detector, scatterer, medium_index=medium_index,
                       illum_wavelen=illum_wavelen,
                       illum_polarization=illum_polarization, theory=theory)
    return finalize(detector, (abs(field*(1-detector.normals))**2).sum(dim=vector))


def calc_holo(detector, scatterer, medium_index=None, illum_wavelen=None,
              illum_polarization=None, theory='auto', scaling=1.0):
    """
    Calculate hologram formed by interference between scattered
    fields and a reference wave

    Parameters
    ----------
    detector : xarray object
        The detector points and calculation metadata used to calculate
        the hologram.
    scatterer : :class:`.scatterer` object
        (possibly composite) scatterer for which to compute scattering
    medium_index : float or complex
        Refractive index of the medium in which the scatter is imbedded
    illum_wavelen : float or ndarray(float)
        Wavelength of illumination light. If illum_wavelen is an array result
        will add a dimension and have all wavelengths
    theory : :class:`.theory` object (optional)
        Scattering theory object to use for the calculation. This is
        optional if there is a clear choice of theory for your scatterer.
        If there is not a clear choice, `calc_holo` will error out and
        ask you to specify a theory
    scaling : scaling value (alpha) for amplitude of reference wave

    Returns
    -------
    holo : xarray.DataArray
        Calculated hologram from the given distribution of spheres
    """
    theory = interpret_theory(scatterer, theory)  # 427 ns
    uschema = prep_schema(
        detector, medium_index, illum_wavelen, illum_polarization)  # 2.2 ms

    scattered_field = theory._calculate_scattered_field(
        scatterer.guess, uschema)
    reference_field = uschema.illum_polarization
    holo = scattered_field_to_hologram(
        scattered_field * scaling, reference_field, uschema.normals)
    return finalize(uschema, holo)  # 563 us


def calc_cross_sections(scatterer, medium_index=None, illum_wavelen=None,
                        illum_polarization=None, theory='auto'):
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
        Scattering theory object to use for the calculation. This is
        optional if there is a clear choice of theory for your scatterer.
        If there is not a clear choice, `calc_cross_sections` will error
        out and ask you to specify a theory

    Returns
    -------
    cross_sections : array (4)
        Dimensional scattering, absorption, and extinction
        cross sections, and <cos theta>
    """
    theory = interpret_theory(scatterer, theory)
    cross_section = theory._calc_cross_sections(
        scatterer=scatterer.guess,
        medium_wavevec=2*np.pi/(illum_wavelen/medium_index),
        medium_index=medium_index,
        illum_polarization=to_vector(illum_polarization))
    return cross_section


def calc_scat_matrix(detector, scatterer, medium_index=None, illum_wavelen=None,
                     theory='auto'):
    """
    Compute farfield scattering matrices for scatterer

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
        Scattering theory object to use for the calculation. This is
        optional if there is a clear choice of theory for your scatterer.
        If there is not a clear choice, `calc_scat_matrix` will error out
        and ask you to specify a theory

    Returns
    -------
    scat_matr : :class:`.Marray`
        Scattering matrices at specified positions

    """
    theory = interpret_theory(scatterer, theory)
    uschema = prep_schema(
        detector, medium_index=medium_index, illum_wavelen=illum_wavelen,
        illum_polarization=False)
    result = theory._calc_scat_matrix(scatterer.guess, uschema)
    return finalize(uschema, result)


def calc_field(detector, scatterer, medium_index=None, illum_wavelen=None,
               illum_polarization=None, theory='auto'):
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
        Scattering theory object to use for the calculation. This is
        optional if there is a clear choice of theory for your scatterer.
        If there is not a clear choice, `calc_field` will error out and
        ask you to specify a theory

    Returns
    -------
    e_field : :class:`.Vector` object
        Calculated hologram from the given distribution of spheres
    """
    theory = interpret_theory(scatterer, theory)
    uschema = prep_schema(
        detector, medium_index=medium_index, illum_wavelen=illum_wavelen,
        illum_polarization=illum_polarization)
    result = theory._calculate_scattered_field(scatterer.guess, uschema)
    return finalize(uschema, result)


# this is pulled out separate from the calc_holo method because
# occasionally you want to turn prepared  e_fields into holograms directly
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

