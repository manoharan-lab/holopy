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
import holopy as hp
from holopy.hologram import Hologram
from holopy.utility.helpers import _ensure_pair
from scatterpy.io import SerializeByConstructor
from scatterpy.errors import InvalidSelection

class ScatteringTheory(SerializeByConstructor):
    """
    Base class for scattering theories
    
    Attributes
    ----------
    imshape : float or tuple (optional)
        Size of grid to calculate scattered fields or
        intensities. This is the shape of the image that calc_field or
        calc_intensity will return
    phi : array 
        Specifies azimuthal scattering angles to calculate (incident
        direction is z)
    theta : array 
        Specifies polar scattering angles to calculate
    optics : :class:`holopy.optics.Optics` object
        specifies optical train

    Notes
    -----
    If phi and theta are both 1-D vectors, the calc_ functions
    should return an array where result(i,j) = result(phi(i),
    theta(j))
    """

    def __init__(self, optics, imshape=(256,256), theta=None, phi=None): 
        self.imshape = _ensure_pair(imshape)
        self.theta = theta
        self.phi = phi
        if isinstance(optics, dict):
            self.optics = hp.Optics(**optics)
        else:
            self.optics = optics

    def calc_field(self, scatterer, selection=None):
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
        selection : array of integers (optional)
            a mask with 1's in the locations of pixels where you
            want to calculate the field, defaults to all pixels

        Raises
        ------
        IncompleteTheory : if calc_field is undefined in the derived class 
        """

        raise NotImplementedError

    # TODO: is this function still needed?  The new ElectricField
    # class makes it essentially trivial -tgd 2011-08-12
    def superpose(self, scatterers, selection=None):
        """
        Superpose fields from different scatterers, taking into
        account phase differences.

        Parameters
        ----------
        scatterers : list of :mod:`scatterpy.scatterer` objects
            list of scatterers to compute field for
        selection : array on integers (optional)
            a mask with 1's in the locations of pixels where you
            want to calculate the field, defaults to all pixels

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

        field = ElectricField(np.zeros(self.imshape), 
                              np.zeros(self.imshape),
                              np.zeros(self.imshape), 0, 
                              self.optics.med_wavelen)
        for s in scatterers:
            field += self.calc_field(s, selection)

        return field
        
    def calc_intensity(self, scatterer, selection=None, 
                       xfield=None, yfield=None, zfield=None): 
        """
        Calculate intensity at focal plane (z=0)

        Parameters
        ----------
        scatterer : :mod:`scatterpy.scatterer` object
            scatterer or list of scatterers to compute field for
        selection : array on integers (optional)
            a mask with 1's in the locations of pixels where you
            want to calculate the field, defaults to all pixels
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
        field = self.calc_field(scatterer, selection)
        return abs(field.x_comp)**2 + abs(field.y_comp)**2

    def calc_holo(self, scatterer, alpha=1.0, selection=None):
        """
        Calculate hologram formed by interference between scattered
        fields and a reference wave
        
        Parameters
        ----------
        scatterer : :mod:`scatterpy.scatterer` object
            scatterer or list of scatterers to compute field for
        alpha : scaling value for intensity of reference wave
        selection : array of integers (optional)
            a mask with 1's in the locations of pixels where you
            want to calculate the field, defaults to all pixels

        Returns
        -------
        holo : :class:`holopy.hologram.Hologram` object
            Calculated hologram from the given distribution of spheres
        """
        scat = self.calc_field(scatterer, selection)
        ref = ElectricField(self.optics.polarization[0],
                            self.optics.polarization[1], 0, 0,
                            self.optics.med_wavelen)

        return Hologram(interfere_at_detector(scat * alpha, ref),
                        optics=self.optics)

    def calc_cross_sections(self, scatterer):
        """
        Calculate scattering, absorption, and extinction 
        cross sections, and asymmetry parameter <cos \theta>. 
        To be implemented by derived classes.

        Parameters
        ----------
        scatterer : :mod:`scatterpy.scatterer` object
            scatterer or list of scatterers to compute for

        Returns
        -------
        cross_sections : array (4)
            Dimensional scattering, absorption, and extinction 
            cross sections, and <cos \theta>
        """
        raise NotImplementedError

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
        xg, yg = np.ogrid[0:xdim, 0:ydim]
        x = xg*px - x
        y = yg*py - y
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(np.sqrt(x**2 + y**2), z)
        phi = np.arctan2(y, x)
        # get phi between 0 and 2pi
        phi = phi + 2*np.pi * (phi < 0)
        return np.dstack((r*self.optics.wavevec, theta, phi))

    def _list_of_sph_coords(self, center, selection=None):
        points = self._spherical_grid(*center)
        if selection is not None:
            points = points[selection]
            if not selection.any():
                raise InvalidSelection("No pixels selected, can't compute fields")

        else:
            points = points.reshape((self.imshape[0]*self.imshape[1], 3))

        return points
    
        
    def _interpret_fields(self, fields, z, selection = None):
        if selection is not None:
            new_fields = []
            for i, field in enumerate(fields):
                new_fields.append(np.zeros(self.imshape, dtype=field.dtype))
                new_fields[i][selection] = field
            fields = new_fields
        else:
            if hasattr(fields, 'shape') and fields.shape[1] == 3:
                fields = fields.T
            fields = [f.reshape(self.imshape) for f in fields]
        return ElectricField(*fields, z_ref = z, wavelen = self.optics.med_wavelen)           
        

# Subclass of scattering theory, overrides functions that depend on array
# ordering and handles the tranposes for sending data to/from fortran
class FortranTheory(ScatteringTheory):
    def _list_of_sph_coords(self, center, selection=None):
        return super(FortranTheory, self)._list_of_sph_coords(center,
                                                              selection).T

    
    
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


    return (abs(e1.x_comp)**2 + abs(e1.y_comp)**2 +
            abs(e2.x_comp)**2 + abs(e2.y_comp)**2 +
            2 * np.real(e1.x_comp*e2.x_comp) +
            2 * np.real(e1.y_comp*e2.y_comp))

    
class InvalidElectricFieldComputation(Exception):
    def __init__(self, reason):
        self.reason = reason
    def __str__(self):
        return "Invalid Electric Computation: " + self.reason
    
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
        # Store the original z in case we later want to access it, but we will
        # now shift everything to store at z=0
        self.orig_z = z_ref
        # shift all electric fields to reference to z=0 (usually this is a
        # detector).  This simplifies all computations involving electric fields
        phase =  np.exp(-1j*np.pi*2*z_ref/wavelen)
        self.x_comp = x_field * phase
        self.y_comp = y_field * phase
        self.z_comp = z_field * phase
        self.wavelen = wavelen

    def __add__(self, other):
        """
        Superpositon of electric fields.  
        """

        if self.wavelen != other.wavelen:
            raise InvalidElectricFieldComputation(
                "Superposition of fields with different wavelengths is not " +
                "implemented")
        
        new_x = self.x_comp + other.x_comp
        new_y = self.y_comp + other.y_comp
        new_z = self.z_comp + other.z_comp

        # We are already z=0 referenced, so z_ref=0 for the new ElectricField
        return ElectricField(new_x, new_y, new_z, 0.0, self.wavelen)

    def __rmul__(self, other):
        return self.__mul__(other)
    def __mul__(self, other):
        if not np.isscalar(other):
            raise InvalidElectricFieldComputation(
                "multiplication by nonscalar values not yet implemented")

        return ElectricField(self.x_comp * other, self.y_comp * other,
                             self.z_comp * other, 0.0, self.wavelen)

    def _array(self):
        return np.dstack((self.x_comp, self.y_comp, self.z_comp))
