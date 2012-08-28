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
from __future__ import division
import numpy as np
from ...core.helpers import _ensure_array
from ...core.data import VectorData
from ..errors import TheoryNotCompatibleError, UnrealizableScatterer
from ..scatterer import Sphere, CoatedSphere, Scatterers
from .scatteringtheory import FortranTheory
from .mie_f import mieangfuncs, miescatlib
from .mie_f.multilayer_sphere_lib import scatcoeffs_multi


class Mie(FortranTheory):
    """
    Compute scattering using the Lorenz-Mie solution.

    This theory calculates exact scattering for single spheres and approximate
    results for groups of spheres.  It does not account for multiple scattering,
    hence the approximation in the case of multiple spheres.  Neglecting
    multiple scattering is a good approximation if the particles are
    sufficiently seperated.
    
    This model can also calculate the exact scattered field from a 
    spherically symmetric particle with an arbitrary number of layers
    with differing refractive indices, using Yang's recursive
    algorithm ([Yang2003]_).
    """

    # don't need to define __init__() because we'll use the base class
    # constructor

    def _calc_scat_matrix(self, scatterer, target):
        if isinstance(scatterer, (Sphere, CoatedSphere)):
            scat_coeffs = self._scat_coeffs(scatterer, target.optics)

            if scatterer.center is None:
                scat_matrs = [mieangfuncs.asm_mie_far(scat_coeffs, theta) for
                              theta in target.positions_theta()]
                return np.array(scat_matrs)
            else:
                raise TheoryNotCompatibleError(self, scatterer)
        else:
            raise TheoryNotCompatibleError(self, scatterer)

    def _calc_field(self, scatterer, target):
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
        if isinstance(scatterer, (Sphere, CoatedSphere)):
            scat_coeffs = self._scat_coeffs(scatterer, target.optics)
            
            # mieangfuncs.f90 works with everything dimensionless.
            # tranpose to get things in fortran format
            # TODO: move this transposing to a wrapper
            fields = mieangfuncs.mie_fields(target.positions_kr_theta_phi(
                    origin = scatterer.center).T, scat_coeffs,
                    target.optics.polarization)
            phase = np.exp(-1j*np.pi*2*scatterer.z / target.optics.med_wavelen)
            result = target.from_1d(VectorData(np.vstack(fields).T))
            return result * phase
        elif isinstance(scatterer, Scatterers):
            spheres = scatterer.get_component_list()
            
            field = self._calc_field(spheres[0], target)
            for sphere in spheres[1:]:
                field += self._calc_field(sphere, target)
            return field
        else:
            raise TheoryNotCompatibleError(self, scatterer)
         
    def _calc_cross_sections(self, scatterer, optics):
        """
        Calculate scattering, absorption, and extinction cross 
        sections, and asymmetry parameter for spherically
        symmetric scatterers.

        Parameters
        ----------
        scatterer : :mod:`scatterpy.scatterer` object
            spherically symmetric scatterer to compute for
            (Calculation would need to be implemented in a radically
            different way, via numerical quadrature, for sphere clusters)

        Returns
        -------
        cross_sections : array (4)
            Dimensional scattering, absorption, and extinction 
            cross sections, and <cos \theta> 
        """
        if isinstance(scatterer, Scatterers):
            raise UnrealizableScatterer(self, scatterer, 
                                        "Use Multisphere to calculate " + 
                                        "radiometric quantities")
        albl = self._scat_coeffs(scatterer, optics)
       
        cscat, cext, cback = miescatlib.cross_sections(albl[0], albl[1]) * \
            (2. * np.pi / optics.wavevec**2)

        cabs = cext - cscat # conservation of energy
        
        asym = 4. * np.pi / (optics.wavevec**2 * cscat) * \
            miescatlib.asymmetry_parameter(albl[0], albl[1])

        return np.array([cscat, cabs, cext, asym])
        
    def _scat_coeffs(self, s, optics):
        x_arr = optics.wavevec * _ensure_array(s.r)
        m_arr = _ensure_array(s.n) / optics.index
    
        # Check that the scatterer is in a range we can compute for
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
        
