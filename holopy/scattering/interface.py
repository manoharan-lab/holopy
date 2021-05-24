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
import yaml

from holopy.core.holopy_object import SerializableMetaclass
from holopy.core.metadata import (
    vector, illumination, update_metadata, to_vector, copy_metadata, from_flat,
    dict_to_array)
from holopy.core.utils import dict_without, ensure_array
from holopy.scattering.scatterer import Sphere, Spheres, Spheroid, Cylinder
from holopy.scattering.errors import (
    AutoTheoryFailed, MissingParameter, InvalidScatterer)
from holopy.scattering.theory import Mie, Multisphere
from holopy.scattering.imageformation import ImageFormation
from holopy.scattering.theory import Tmatrix
from holopy.scattering.theory.dda import DDA
from holopy.core.mapping import Mapper, read_map


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


def interpret_theory(scatterer, theory='auto'):
    if isinstance(theory, str) and theory == 'auto':
        theory = determine_default_theory_for(scatterer)
    if isinstance(theory, SerializableMetaclass):
        theory = theory()
    return theory


def validate_scatterer(scatterer):
    mapper = Mapper()
    scatterer_map = mapper.convert_to_map(scatterer.parameters)
    guesses = [par.guess for par in mapper.parameters]
    return scatterer.from_parameters(read_map(scatterer_map, guesses))


def finalize(detector, result):
    if not hasattr(detector, 'flat'):
        result = from_flat(result)
    return copy_metadata(detector, result, do_coords=False)


# Some comments on why `determine_default_theory_for` exists, rather than each
# Scatterer class knowing what a good default theory is.
# The problem is that the theories (Mie etc) import Sphere to see if
# the theory can handle the scatterer, in the can_handle method and
# others. Worse, since the DDA theory calls an external DDA library
# with specially-defined DDA objects, the DDA theory has a switch statement
# for basically every holopy scatterer. So right now the scatterers can't
# have a default theory and/or valid theory attr, as this causes a dependency
# loop.
def determine_default_theory_for(scatterer):
    if isinstance(scatterer, Sphere):
        theory = Mie()
    elif isinstance(scatterer, Spheres):
        theory = _choose_mie_vs_multisphere(scatterer)
    elif isinstance(scatterer, Spheroid) or isinstance(scatterer, Cylinder):
        theory = Tmatrix()
    elif DDA.can_handle(scatterer):
        theory = DDA()
    else:
        raise AutoTheoryFailed(scatterer)
    return theory


def calc_intensity(detector, scatterer, medium_index=None, illum_wavelen=None,
                   illum_polarization=None, theory='auto'):
    """
    Calculate intensity from the scattered field at a set of locations

    Parameters
    ----------
    detector : xarray object
        The detector points and calculation metadata used to calculate
        the intensity.
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
    intensity = (np.abs(field.sel(vector=['x', 'y']))**2).sum(dim=vector)
    return finalize(detector, intensity)


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
    scatterer = validate_scatterer(scatterer)
    uschema = prep_schema(
        detector, medium_index, illum_wavelen, illum_polarization)
    scaling = dict_to_array(detector, scaling)
    theory = interpret_theory(scatterer, theory)
    imageformer = ImageFormation(theory)
    scattered_field = imageformer.calculate_scattered_field(scatterer, uschema)
    reference_field = uschema.illum_polarization
    holo = scattered_field_to_hologram(
        scattered_field * scaling, reference_field)
    return finalize(uschema, holo)


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
    scatterer = validate_scatterer(scatterer)
    theory = interpret_theory(scatterer, theory)
    imageformer = ImageFormation(theory)
    cross_section = imageformer.calculate_cross_sections(
        scatterer=scatterer,
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
    detector : xarray object
        The detector points and calculation metadata used to calculate
        the scattering matrices.
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
    scatterer = validate_scatterer(scatterer)
    uschema = prep_schema(
        detector, medium_index=medium_index, illum_wavelen=illum_wavelen,
        illum_polarization=False)
    theory = interpret_theory(scatterer, theory)
    imageformer = ImageFormation(theory)
    result = imageformer.calculate_scattering_matrix(scatterer, uschema)
    return finalize(uschema, result)


def calc_field(detector, scatterer, medium_index=None, illum_wavelen=None,
               illum_polarization=None, theory='auto'):
    """
    Calculate the scattered fields from a scatterer illuminated by
    a reference wave.

    Parameters
    ----------
    detector : xarray object
        The detector points and calculation metadata used to calculate
        the scattered fields.
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
    scatterer = validate_scatterer(scatterer)
    uschema = prep_schema(
        detector, medium_index=medium_index, illum_wavelen=illum_wavelen,
        illum_polarization=illum_polarization)
    theory = interpret_theory(scatterer, theory)
    imageformer = ImageFormation(theory)
    result = imageformer.calculate_scattered_field(scatterer, uschema)
    return finalize(uschema, result)


# this is pulled out separate from the calc_holo method because
# occasionally you want to turn prepared  e_fields into holograms directly
def scattered_field_to_hologram(scat, ref):
    """
    Calculate a hologram from an E-field

    Parameters
    ----------
    scat : :class:`.VectorGrid`
        The scattered (object) field
    ref : xarray[vector]]
        The reference field
    """
    total_field = scat + ref
    holo = (np.abs(total_field.sel(vector=['x', 'y']))**2).sum(dim=vector)
    return holo


def _choose_mie_vs_multisphere(spheres):
    center_or_radius_not_set = [
        getattr(s, k) is None
        for s in spheres.scatterers for k in ['center', 'r']]
    if len(spheres.scatterers) == 1:
        theory = Mie()
    elif any(center_or_radius_not_set):
        msg = ("Sphere centers and radii must be set for scattering " +
               "calculations with more than one sphere.")
        raise InvalidScatterer(spheres, msg)
    elif any([not np.isscalar(sphere.r) for sphere in spheres.scatterers]):
        warn("HoloPy's multisphere theory can't handle coated spheres." +
             "Using Mie theory.")
        theory = Mie()
    else:
        # We choose the theory that is most accurate, which is
        # Multisphere if the spheres are close enough, else Mie
        # superposition.
        # What is close enough? From Jerome's paper [1], the relative
        # effects from multiple scattering are on the order of
        #       error_mie ~ Q_ext *(ka)^2 / kR
        # where a is the sphere radius, R the characteristic separation,
        # and k the wavevector of the light.
        # For large spheres, Q_ext -> 2, so we can write this as
        #       error_mie ~ kR * (a / R)^2
        #
        # The Multisphere theory uses spherical harmonic translation
        # theorems to expand out each particle's scattered field and
        # self-consistently solve for multiple-sphere scattering.
        # For computational reasons Multisphere does not go past a fixed
        # order (=70, which happens at kR ~ 100), but the actual number
        # of terms needed scales as kR. Presumably at kR ~ 100 the error
        # from multisphere is still small, say ~0.1.
        # So for large R, the error in Multisphere is approximately
        #       error_multisphere = 0.1 * kR / 100
        # The error for Multisphere is smaller than the Mie
        # superposition error when
        #       0.1 * kR / 100 < kR * (a / R)^2,       or
        #       R < sqrt(1000) * a
        # Since this is just an order-of-magnitude calculation, we take
        # sqrt(1000) ~ 30.
        # Note that the Mie error could still be large, if ka >> R / a
        #
        # [1] Fung, Jerome, et al. "Imaging multiple colloidal particles
        # by fitting electromagnetic scattering solutions to digital
        # holograms." Journal of Quantitative Spectroscopy and Radiative
        # Transfer 113.18 (2012): 2482-2489.
        max_radius = max([sphere.r for sphere in spheres.scatterers])
        centers = np.array([sphere.center for sphere in spheres.scatterers])
        dx = centers.reshape(1, -1, 3) - centers.reshape(-1, 1, 3)
        max_separation = np.linalg.norm(dx, axis=2).max()
        close_enough = max_separation <= 30 * max_radius

        theory = Multisphere() if close_enough else Mie()
    return theory
