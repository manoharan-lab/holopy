# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca
# W. Perry, Jerome Fung, and Ryan McGorty
#
# This file is part of Holopy.
#
# Holopy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Holopy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Holopy.  If not, see <http://www.gnu.org/licenses/>.
"""
Base class for scattering theories.  Implements python-based
calc_intensity and calc_holo, based on subclass's calc_field

.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

import numpy as np
from holopy import Optics
from holopy.hologram import Hologram
from holopy.utility.helpers import _ensure_pair

class NotImplementedError(Exception):
    def __init__(self, method, theory, message=None):
        self.message = message
        self.method = method
        self.theory = theory
    def __str__(self):
        return ("Method " + self.method + " not implemented in theory " + 
                self.theory + ". " + self.message)

class ScatteringTheory(object):
    """
    Base class for scattering theories

    Attributes
    ----------
    imshape : float or tuple (optional)
        Size of grid to calculate scattered fields or
        intensities. This is the shape of the image that calc_field or
        calc_intensity will return
    phis : array 
        Specifies azimuthal scattering angles to calculate (incident
        direction is z)
    thetas : array 
        Specifies polar scattering angles to calculate
    optics : :class:`holopy.optics.Optics` object
        specifies optical train

    Notes
    -----
    If phis and thetas are both 1-D vectors, the calc_ functions
    should return an array where result(i,j) = result(phi(i),
    theta(j))
    """

    def __init__(self, imshape=(256, 256), thetas=None, phis=None,
                 optics=None): 
        self.imshape = _ensure_pair(imshape)
        self.thetas = thetas
        self.phis = phis
        if isinstance(optics, dict):
            optics = Optics(**optics)
        elif optics is None:
            self.optics = Optics()
        else:
            self.optics = optics

    def calc_field(self, scatterer):
        """
        Calculate fields.  Implemented in derived classes only.

        Parameters
        ----------
        scatterer : :mod:`scatterpy.scatterer` object
            scatterer or list of scatterers to compute field for

        Returns
        -------
        xfield, yfield, zfield : complex arrays with shape `imshape`
            x, y, z components of scattered fields

        Raises
        ------
        NotImplemented : if calc_field is undefined in the derived class 
        """

        raise NotImplementedError(self.calc_field().__name__,
                             self.__class__.__name__) 

    # TODO: is this function still needed?  The new ElectricField class makes it
    # essentially trivial -tgd 2011-08-12
    def superpose(self, scatterers):
        """
        Superpose fields from different scatterers, taking into
        account phase differences.

        Parameters
        ----------
        scatterers : list of :mod:`scatterpy.scatterer` objects
            list of scatterers to compute field for

        Notes
        -----
        For multiple particles, this code superposes the fields
        calculated from each particle (using calc_field()). The
        scattering calculation for each individual particle assumes
        that the incident field phase angle is 0 at each particle's
        center.  So when we superpose the fields, we need to correct
        for the phase differences between particles.  We choose the
        convention that the incident field phase angle will be 0 at
        z=0.  This makes it possible to interfere the total scattered
        field with the incident field to compute the hologram (in
        calc_holo())

        Short summary: the total scattered field is computed such that
        the phase angle of the incident field is 0 at z=0
        """

        field = self.calc_field(scatterers[0])
        for s in scatterers[1:]:
            field += self.calc_field(s)

        return field
        
    def calc_intensity(self, scatterer, 
                       xfield=None, yfield=None, zfield=None): 
        """
        Calculate intensity at focal plane (z=0)

        Parameters
        ----------
        scatterer : :mod:`scatterpy.scatterer` object
            scatterer or list of scatterers to compute field for
        xfield, yfield, zfield : array (optional)
            Components of scattered field

        Returns
        -------
        inten : array(imshape, imshape)
            scattered intensity

        Notes
        -----
        Total scattered intensity only takes into account the x- and
        y-components of the E-field.  The z-component is ignored
        because the detector's pixels should be sensitive to the z
        component of the Poynting vector, E x B, and the z component
        of E x B cannot depend on Ez.

        You can specify the fields to avoid the cost of calculating
        them twice during calc_holo()
        """
        field = self.calc_field(scatterer)
        return interfere_at_detector(field, field)

    def calc_holo(self, scatterer, alpha=1.0):
        """
        Calculate hologram formed by interference between scattered
        fields and a reference wave
        
        Parameters
        ----------
        scatterer : :mod:`scatterpy.scatterer` object
            scatterer or list of scatterers to compute field for
        alpha : scaling value for intensity of reference wave

        Returns
        -------
        holo : :class:`holopy.hologram.Hologram` object
            Calculated hologram from the given distribution of spheres
        """

        scat = self.calc_field(scatterer)
        ref = ElectricField(self.optics.polarization[0],
                            self.optics.polarization[1], 0, 0,
                            self.optics.med_wavelen)

        return Hologram(interfere_at_detector(scat * alpha, ref),
                        optics=self.optics)

    def _spherical_grid(self, x, y, z):
        """
        Parameters
        ----------
        x, y, z : real
            Center of the spherical coordinate system

        Returns
        -------
        theta, phi: 1-D array
            Angles
        r : 2-D array
            Distances (normalized by wavevector)
        """
        px, py = self.optics.pixel
        xdim, ydim = self.imshape
        x = np.mgrid[0:xdim]*px - x
        y = np.mgrid[0:ydim]*py - y
        theta = np.arctan2(np.sqrt(x**2+y**2), z)
        phi = np.arctan2(y, x)
        r = np.sqrt(x.reshape(x.size,1)**2 + y.reshape(1,y.size)**2 + z**2)
        return theta, phi, r*self.optics.wavevec
        
#TODO: Should this be a method of the Electric field class? - tgd 2011-08-15
def interfere_at_detector(e1, e2):
    """
    Compute the intensity as detected by a plane sensor normal to z from the
    interference of e1 and e2

    Parameters
    ----------
    e1, e2: :class:`scatterpy.theory.scatteringtheory.ElectricField`
        The two electric fields to superimpose

    Returns
    i: :class:`numpy.ndarray`
        2d array of detected intensity
    """
    # This function assumes the detector is normal to z and planar, these
    # assumptions could be relaxed by adding more parameters if necessary

    # normally we would have
    # interference = conj(xfield)*phase + conj(phase)*xfield, 
    # but we choose phase angle = 0 at z=0, so phase = 1
    # which gives 2*real(xfield)

    return (abs(e1.x_comp**2) + abs(e1.y_comp**2) + abs(e2.x_comp**2) +
            abs(e2.y_comp**2) + 2 * np.real(e1.x_comp*e2.x_comp) +
            2 * np.real(e1.y_comp*e2.y_comp))


    
class InvalidElectricFieldComputation(Exception):
    def __init__(self, reason):
        self.reason = reason
    def __str__(self):
        "Invalid Electric Computation: " + self.reason
    
class ElectricField(object):
    """
    Representation of electric fields.  Correctly handles phase referencing for
    superposition

    Attributes
    ----------
    x_field, y_field, z_field : complex :class:`numpy.ndarray`
        Fields in each cartesian firection
    z_ref: float (distance)
        Z position of 0 phase (phase reference)
    wavelen: float (distance)
        wavelength of the light this field represents, this should be the
        wavelength in medium if a non unity index medium is present
    """

    def __init__(self, x_field, y_field, z_field, z_ref, wavelen):
        self.x_comp = x_field
        self.y_comp = y_field
        self.z_comp = z_field
        self.z_ref = z_ref
        self.wavelen = wavelen

    def __add__(self, other):
        """
        Superpositon of electric fields.  Phase shifts as necessary to give all
        fields the same reference phase.
        """
        
        if self.wavelen != other.wavelen:
            raise InvalidElectricFieldComputation(
                "Superposition of fields with different wavelengths is not " +
                "implemented")
        
        
        def phase_shift(z):
            # - sign is needed to adjust phase to 0, since our z is opposite the
            # propagation direction we have to backpropagate fields to get them
            # to the origin.  
            return np.exp(-1j*np.pi*2*z/self.wavelen)
        
        new_x = (self.x_comp * phase_shift(self.z_ref) +
                 other.x_comp * phase_shift(other.z_ref))
        new_y = (self.y_comp * phase_shift(self.z_ref) +
                 other.y_comp * phase_shift(other.z_ref))
        new_z = (self.z_comp * phase_shift(self.z_ref) +
                 other.z_comp * phase_shift(other.z_ref))

        # We phase shifted all of the fields to reference against 0, so the
        # field returned has z_ref=0
        return ElectricField(new_x, new_y, new_z, 0.0, self.wavelen)

    def __rmul__(self, other):
        return self.__mult__(self, other)
    def __mul__(self, other):
        if not np.isscalar(other):
            raise InvalidElectricFieldComputation(
                "multiplication by nonscalar values not yet implemented")

        return ElectricField(self.x_comp * other, self.y_comp * other,
                             self.z_comp * other, self.z_ref, self.wavelen)
