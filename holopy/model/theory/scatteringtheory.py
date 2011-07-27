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

class ScatteringTheory():
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
        scatterer : :mod:`holopy.model.scatterer` object
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

    def superpose(self, scatterers):
        """
        Superpose fields from different scatterers, taking into
        account phase differences.

        Parameters
        ----------
        scatterers : list of :mod:`holopy.model.scatterer` objects
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
        xfield_tot = np.zeros(self.imshape, dtype='complex128')
        yfield_tot = np.zeros(self.imshape, dtype='complex128')
        zfield_tot = np.zeros(self.imshape, dtype='complex128')

        for s in scatterers:
            xfield, yfield, zfield  = self.calc_field(s)
            # see Notes section above for how phase is computed.
            # The - sign in front of the phase is necessary to get the
            # holograms to come out right!  I think this is because in
            # our convention, k points in the -z direction. 
            phase_dif = (np.exp(-1j*np.pi*2*(s.z)/self.optics.med_wavelen))
            xfield_tot += xfield*phase_dif
            yfield_tot += yfield*phase_dif
            zfield_tot += zfield*phase_dif
        return xfield_tot, yfield_tot, zfield_tot

    def calc_intensity(self, scatterer, 
                       xfield=None, yfield=None, zfield=None): 
        """
        Calculate intensity at focal plane (z=0)

        Parameters
        ----------
        scatterer : :mod:`holopy.model.scatterer` object
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

        if (xfield is None) and (yfield is None):
            xfield, yfield, zfield = self.calc_field(scatterer)
        return (abs(xfield**2) + abs(yfield**2))

    def calc_holo(self, scatterer, alpha=1.0):
        """
        Calculate hologram formed by interference between scattered
        fields and a reference wave
        
        Parameters
        ----------
        scatterer : :mod:`holopy.model.scatterer` object
            scatterer or list of scatterers to compute field for
        alpha : scaling value for intensity of reference wave

        Returns
        -------
        holo : :class:`holopy.hologram.Hologram` object
            Calculated hologram from the given distribution of spheres
        """

        xfield, yfield, zfield = self.calc_field(scatterer)
        total_scat_inten = self.calc_intensity(scatterer,
                                               xfield=xfield,
                                               yfield=yfield,
                                               zfield=zfield) 
        # normally we would have
        # interference = conj(xfield)*phase + conj(phase)*xfield, 
        # but we choose phase angle = 0 at z=0, so phase = 1
        # which gives 2*real(xfield)
        interference = 2*np.real(xfield * self.optics.polarization[0] +
                                 yfield * self.optics.polarization[1])
        holo = (1. + total_scat_inten*(alpha**2) + 
                interference*alpha)     # holo should be purely real

        return Hologram(holo, optics = self.optics)

