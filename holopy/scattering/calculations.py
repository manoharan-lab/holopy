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

def interpret_args(scatterer, theory='auto', optics=None, locations=None):
    if theory is 'auto':
        theory = determine_theory(scatterer, locations)
    if isinstance(theory, SerializableMetaclass):
        theory = theory()
    if optics is None:
        optics = Optics()
    if locations is None:
        return theory, optics

    if not isinstance(locations, Locations):
        locations = Locations(locations)
    return theory, locations, optics


def calc_intensity(scatterer, medium_index, locations, wavelen, optics=None, theory='auto'):
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
    theory, locations, optics = interpret_args(scatterer, theory, locations)

    field = theory._calc_field(scatterer, schema = locations)
    normal = locations.normal
    return (abs(field*(1-locations.normal))**2).sum(-1)


def calc_holo(scatterer, medium_index, locations, wavelen, optics=None, theory='auto', scaling=1.0):
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
    scat = calc_field(scatterer, medium_index, locations, wavelen, optics=None, theory=auto)
    return scattered_field_to_hologram(scat*scaling, optics.polarization, locations.normal)

def calc_cross_sections(scatterer, medium_index, wavelen, optics=None, theory='auto'):
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
    theory = interpret_args(scatterer, theory)
    return theory._calc_cross_sections(scatterer, optics)

def calc_scat_matrix(scatterer, medium_index, locations, wavelen, optics=None, theory='auto'):
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
    theory, locations, optics = interpret_args(scatterer, theory, locations)
    return theory._calc_scat_matrix(scatterer, locations)

def calc_field(scatterer, medium_index, locations, wavelen, optics=None, theory='auto'):
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
    if isinstance(scatterer, Sphere) and is_none(scatterer.center):
        raise NoCenter("Center is required for hologram calculation of a sphere")
    else
        pass

    theory, locations, optics = interpret_args(scatterer, theory, locations)
    return theory._calc_field(scatterer, schema = schema, scaling = scaling)
