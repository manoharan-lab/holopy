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
Calculates holograms of spheres using Fortran implementation of Mie
theory. Uses superposition to calculate scattering from multiple
spheres. Uses full radial dependence of spherical Hankel functions for
scattered field.

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
'''
import numpy as np
import mie_f.mieangfuncs as mieangfuncs
import mie_f.miescatlib as miescatlib
from mie_f.multilayer_sphere_lib import scatcoeffs_multi

from scatterpy.errors import TheoryNotCompatibleError, UnrealizableScatterer
from scatterpy.scatterer import Sphere, CoatedSphere, Composite
from scatterpy.theory.scatteringtheory import ScatteringTheory, ElectricField
from holopy.utility.helpers import _ensure_array

class Mie(ScatteringTheory):
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

    def calc_field(self, scatterer, selection=None):
        """
        Calculate fields for single or multiple spheres

        Parameters
        ----------
        scatterer : :mod:`scatterpy.scatterer` object
            scatterer or list of scatterers to compute field for
        selection : array of integers (optional)
            a mask with 1's in the locations of pixels where you
            want to calculate the field, defaults to all pixels
        Returns
        -------
        field : :class:`scatterpy.theory.scatteringtheory.ElectricField`with shape `imshape`
            scattered electric field

        Notes
        -----
        For multiple particles, this code superposes the fields
        calculated from each particle (using calc_mie_fields()). 
        """
        if selection == None:
            selection = np.ones(self.imshape,dtype='int')
        def sphere_field(s, selection=None):
            scat_coeffs = self._scat_coeffs(s)

            # mieangfuncs.f90 works with everything dimensionless.
            e_x, e_y, e_z = mieangfuncs.mie_fields_sph(self._spherical_grid(s.x,
                                                                            s.y,
                                                                            s.z),
                                                       scat_coeffs,
                                                       selection,
                                                       self.optics.polarization)

            return ElectricField(e_x, e_y, e_z, s.z, self.optics.med_wavelen)
        
        if isinstance(scatterer, (Sphere, CoatedSphere)):
            return sphere_field(scatterer, selection)
        elif isinstance(scatterer, Composite):
            spheres = scatterer.get_component_list()
            # compatibility check: verify that the cluster only contains
            # spheres 
            for s in spheres:
                if not isinstance(s, (Sphere, CoatedSphere)):
                    raise TheoryNotCompatibleError(self, s)
            # if it passes, superpose the fields
            scatterer.validate()
    
            return self.superpose(spheres, selection)
        else: raise TheoryNotCompatibleError(self, scatterer)

# Disable this function for now since the fast fortran subroutine does not
# handle spherical coordinates yet.  This will cause fallback to the method
# which uses the spherical coordinates calc_fields version.  
#
# TODO: make a spherical coordinates version of the singleholo fortran function
# and switch to using it
#    def calc_holo(self, scatterer, alpha=1.0):
#        """
#        Calculate hologram formed by interference between scattered
#        fields and a reference wave
#        
#        Parameters
#        ----------
#        scatterer : :mod:`scatterpy.scatterer` object
#            scatterer or list of scatterers to compute field for
#        alpha : scaling value for intensity of reference wave
#
#        Returns
#        -------
#        holo : :class:`holopy.hologram.Hologram` object
#            Calculated hologram from the given distribution of spheres
#
#        Notes
#        -----
#        For a single particle, this code uses a fast Fortran
#        subroutine to calculate the hologram.  Otherwise it uses the
#        Fortran subroutine for calculating the fields from each
#        particle, then superposes them using numpy.
#        """
#
#        if isinstance(scatterer, Sphere):
#            scat_coeffs = self._scat_coeffs(scatterer)
#
#            theta, phi, kr = self._spherical_grid(scatterer.x, scatterer.y, scatterer.z)
#
#            # TODO: convert to calling in spherical coordinates
#            gridx, gridy = self._grid()
#
#            holo = singleholo(gridx, gridy,
#                              scatterer.center * self.optics.wavevec,
#                              scat_coeffs, alpha, self.optics.polarization)
#            
#        else:   # call base class calc_holo
#            holo = ScatteringTheory.calc_holo(self, scatterer, 
#                                              alpha=alpha)
#
#        return Hologram(holo, optics = self.optics)
#
#        
#    # TODO: remove this function once self.calc_holo no longer needs it
#    def _grid(self):
#        px, py = self.optics.pixel
#        xdim, ydim = self.imshape
#        return (self.optics.wavevec*np.mgrid[0:xdim]*px,
#                self.optics.wavevec*np.mgrid[0:ydim]*py)



    def _scat_coeffs(self, s):
        x_arr = self.optics.wavevec * _ensure_array(s.r)
        m_arr = _ensure_array(s.n) / self.optics.index
    
        # Check that the scatterer is in a range we can compute for
        if x_arr.min() < 0:
            raise UnrealizableScatterer(self, s, "radius is negative")
        if x_arr.max() > 1e3:
            raise UnrealizableScatterer(self, s, "radius too large, field "+
                                        "calculation would take forever")
        
        if len(x_arr) == 1 and len(m_arr) == 1:
            # Could just use scatcoeffs_multi here, but jerome is in favor of
            # keeping the simpler single layer code here 
            lmax = miescatlib.nstop(x_arr[0])
            return  miescatlib.scatcoeffs(x_arr[0], m_arr[0], lmax)
        else:
            return scatcoeffs_multi(m_arr, x_arr)
