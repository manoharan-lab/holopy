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
'''
Calculates holograms of moving spheres using Fortran implementation of 
the Lorenz-Mie solution.  Currently, only works for MovingSphere 
scatterers.

TODO: a general MovingScatterer class?

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
'''
import mie_f.mieangfuncs as mieangfuncs
import mie_f.miescatlib as miescatlib
import numpy as np

from holopy.hologram import Hologram
from scatterpy.errors import TheoryNotCompatibleError, UnrealizableScatterer
from scatterpy.scatterer import Sphere
from scatterpy.scatterer.movingsphere import MovingSphere
from scatterpy.theory.scatteringtheory import ScatteringTheory, ElectricField
from scatterpy.theory.scatteringtheory import interfere_at_detector
#from scatterpy.theory.scatteringtheory import Mie

class MieSmear(ScatteringTheory):
    """
    Class that contains methods and parameters for calculating
    scattering using Mie theory.

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

    # don't need to define __init__() because we'll use the base class
    # constructor

    def calc_field(self, scatterer, scat_coeffs = None):
        """
        Calculate fields for single moving spheres.

        Parameters
        ----------
        scatterer : :mod:`scatterpy.scatterer` object
            scatterer or list of scatterers to compute field for

        Returns
        -------
        field : :class:`scatterpy.theory.scatteringtheory.ElectricField`with shape `imshape`
            scattered electric field

        Notes
        -----
        For multiple particles, this code superposes the fields
        calculated from each particle (using calc_mie_fields()). 
        """

        scatterer.validate()
                        
        def sphere_field(s, coeffs):
            # mieangfuncs.f90 works with everything dimensionless.
            e_x, e_y, e_z = mieangfuncs.mie_fields_sph(self._spherical_grid(s.x,
                                                                            s.y,
                                                                            s.z),
                                                       coeffs,
                                                       self.optics.polarization)

            return ElectricField(e_x, e_y, e_z, s.z, self.optics.med_wavelen)
        
        if isinstance(scatterer, Sphere):
            if scat_coeffs == None:
                scat_coeffs = self._scat_coeffs()
            return sphere_field(scatterer, scat_coeffs)
        else: raise TheoryNotCompatibleError(self, scatterer)

    def calc_holo(self, scatterer, alpha=1.0):
        '''
        Superpose hologram intensities from each of the constituent spheres.
        Divide by number of smeared spheres.
        '''
        scat_coeffs = self._scat_coeffs(scatterer.scatterers[0])
        holo = Hologram(np.zeros(self.imshape), optics = self.optics)
        for sph in scatterer.scatterers: # spheres are identical
            scat = self.calc_field(sph, scat_coeffs)
            ref = ElectricField(self.optics.polarization[0],
                                self.optics.polarization[1], 0, 0,
                                self.optics.med_wavelen)
            holo = holo + Hologram(interfere_at_detector(scat * alpha, ref),
                                   optics = self.optics)
        return holo/scatterer.scatterers.__len__()


    def _scat_coeffs(self, s):
        x_p = self.optics.wavevec * s.r
        m_p = s.n / self.optics.index

        # Check that the scatterer is in a range we can compute for
        if x_p < 0:
            raise UnrealizableScatterer(self, s, "radius is negative")
        if x_p > 1e3:
            raise UnrealizableScatterer(self, s, "radius too large, field "+
                                        "calculation would take forever")
        
        # Calculate maximum order lmax of Mie series expansion.
        lmax = miescatlib.nstop(x_p)
        return  miescatlib.scatcoeffs(x_p, m_p, lmax)

