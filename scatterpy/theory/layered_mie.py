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
Calculates scattering from multilayered spheres using algorithm of Yang, App.
Opt. 1993.

TODO: think about generalizing this to do clusters of coated particles.

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
'''
import mie_f.mieangfuncs as mieangfuncs
import numpy as np
from mie_f.mieangfuncs import mie_fields_sph
from mie_f.multilayer_sphere_lib import scatcoeffs_multi
import scatterpy
from holopy.hologram import Hologram
from scatterpy.errors import TheoryNotCompatibleError, UnrealizableScatterer
from scatterpy.scatterer import CoatedSphere
from scatterpy.theory.scatteringtheory import ScatteringTheory, ElectricField

class LayeredMie(ScatteringTheory):
    """
    Class that contains methods and parameters for calculating
    scattering using concentrically-coated extension to the Lorentz-Mie
    solution.

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

    def calc_field(self, scatterer):
        """
        Calculate fields for single coated spheres.  So far we've only
        defined a 2-layered scatterer.

        Parameters
        ----------
        scatterer : :mod:`scatterpy.scatterer` object
            scatterer or list of scatterers to compute field for

        Returns
        -------
        field : :class:`scatterpy.theory.scatteringtheory.ElectricField` with
        shape `imshape`
            scattered electric field

        """

        scatterer.validate()
        
        if isinstance(scatterer, CoatedSphere):
            scat_coeffs = self._scat_coeffs(scatterer)
            e_x, e_y, e_z = mie_fields_sph(self._spherical_grid(scatterer.x,
                                                                scatterer.y,
                                                                scatterer.z),
                                           scat_coeffs, 
                                           self.optics.polarization)
            return ElectricField(e_x, e_y, e_z, scatterer.z, 
                                 self.optics.med_wavelen)
        else: raise TheoryNotCompatibleError(self, scatterer)


    def _scat_coeffs(self, s):
        x_arr = self.optics.wavevec * np.array([s.r1, s.r2])
        m_arr = np.array([s.n1, s.n2]) / self.optics.index

        # Check that the scatterer is in a range we can compute for
        if x_arr.min() < 0:
            raise UnrealizableScatterer(self, s, "radius is negative")
        if x_arr.max() > 1e3:
            raise UnrealizableScatterer(self, s, "radius too large, field "+
                                        "calculation would take forever")
        
        return scatcoeffs_multi(m_arr, x_arr)

