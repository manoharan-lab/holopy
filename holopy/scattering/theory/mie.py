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
'''
Calculates holograms of spheres using Fortran implementation of Mie
theory. Uses superposition to calculate scattering from multiple
spheres. Uses full radial dependence of spherical Hankel functions for
scattered field.

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Jerome Fung <jerome.fung@post.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
'''

import numpy as np
from holopy.core.utils import ensure_array
from holopy.core.errors import DependencyMissing
from holopy.scattering.errors import TheoryNotCompatibleError, InvalidScatterer
from holopy.scattering.scatterer import Sphere, Spheres
from holopy.scattering.theory.scatteringtheory import ScatteringTheory
try:
    from holopy.scattering.theory.mie_f import (mieangfuncs, miescatlib,
                                                scatcoeffs_multi)
    _COMPILED_FORTRAN = True
except ImportError:
    _COMPILED_FORTRAN = False


class Mie(ScatteringTheory):
    """
    Compute scattering using the Lorenz-Mie solution.

    This theory calculates exact scattering for single spheres and approximate
    results for groups of spheres.  It does not account for multiple scattering,
    hence the approximation in the case of multiple spheres.  Neglecting
    multiple scattering is a good approximation if the particles are
    sufficiently separated.

    This model can also calculate the exact scattered field from a
    spherically symmetric particle with an arbitrary number of layers
    with differing refractive indices, using Yang's recursive
    algorithm ([Yang2003]_).

    By default, calculates radial component of scattered electric fields,
    which is nonradiative.

    Currently, in calculating the Lorenz-Mie scattering coefficients,
    the maximum size parameter x = ka is limited to 1000.
    """
    def __init__(self, compute_escat_radial=True, full_radial_dependence=True,
                 eps1=1e-2, eps2=1e-16):
        """
        Parameters
        ----------
        compute_escat_radial : bool
            determines if radial components will be calculated
        full_radial dependence : bool
            determines if the full spherical Hankel function will be used,
            or if it will be approximated to be in the far field.
        """
        self.compute_escat_radial = compute_escat_radial
        self.full_radial_dependence = full_radial_dependence
        self.eps1 = eps1
        self.eps2 = eps2
        if not _COMPILED_FORTRAN:
            raise DependencyMissing("Mie theory", "This is probably "
                                    "due to a problem with compiling Fortran "
                                    "code, as it should be built with the rest"
                                    " of HoloPy through f2py.")
        super().__init__()

    def can_handle(self, scatterer):
        return isinstance(scatterer, Sphere)

    def raw_scat_matrs(self, scatterer, pos, medium_wavevec, medium_index):
        '''
        Returns far-field amplitude scattering matrices (with theta and phi
        dependence only) -- assume spherical wave asymptotic r dependence
        '''
        if self.can_handle(scatterer):
            scat_coeffs = self._scat_coeffs(
                scatterer, medium_wavevec, medium_index)

            # In the mie solution the amplitude scattering matrix is
            # independent of phi
            return [mieangfuncs.asm_mie_far(scat_coeffs, theta)
                    for r, theta, phi in pos.T]
        else:
            raise TheoryNotCompatibleError(self, scatterer)

    def raw_fields(
            self, positions, scatterer, medium_wavevec, medium_index,
            illum_polarization):
        scat_coeffs = self._scat_coeffs(scatterer, medium_wavevec, medium_index)
        fields = mieangfuncs.mie_fields(
            positions, scat_coeffs, illum_polarization.values[:2],
            self.compute_escat_radial, self.full_radial_dependence)
        return fields

    def _raw_internal_fields(
            self, positions, scatterer, medium_wavevec, medium_index,
            illum_polarization):
        scat_coeffs = self._scat_coeffs(scatterer, medium_wavevec, medium_index)
        # TODO BUG: this isn't right for layered spheres (and will
        # probably crash)
        fields = mieangfuncs.mie_internal_fields(
            positions, scatterer.n, scat_coeffs, illum_polarization)
        return fields

    def raw_cross_sections(
            self, scatterer, medium_wavevec, medium_index, illum_polarization):
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

        Notes
        -----
        The radiation pressure cross section C_pr is given by
        C_pr = C_ext - <cos \theta> C_sca.

        The radiation pressure force on a sphere is

        F = (n_med I_0 C_pr) / c

        where I_0 is the incident intensity.  See van de Hulst, p. 14.
        """
        if isinstance(scatterer, Spheres):
            msg = "Use Multisphere to calculate radiometric quantities"
            raise InvalidScatterer(scatterer, msg)
        albl = self._scat_coeffs(scatterer, medium_wavevec, medium_index)

        cscat, cext, cback = miescatlib.cross_sections(albl[0], albl[1]) * \
            (2. * np.pi / medium_wavevec**2)

        cabs = cext - cscat # conservation of energy

        asym = 4. * np.pi / (medium_wavevec**2 * cscat) * \
            miescatlib.asymmetry_parameter(albl[0], albl[1])

        return np.array([cscat, cabs, cext, asym])

    def _scat_coeffs(self, s, medium_wavevec, medium_index):
        '''
        Calculate Mie scattering coefficients.

        Parameters
        ----------
        s : :mod:`scatterer.Sphere` object
        medium_wavevec : float
            Wave vector in the medium, k = 2 * pi * n_med / lambda_0
        medium_index : float
            Medium refractive index

        Returns
        -------
        ndarray (2, n), complex
           Lorenz-Mie scattering coefficients a_n and b_n

        Notes
        -----
        See Bohren & Huffman for mathematical description.

        '''
        if (ensure_array(s.r) == 0).any():
            raise InvalidScatterer(s, "Radius is zero")
        x_arr = ensure_array(medium_wavevec * ensure_array(s.r))
        m_arr = ensure_array(ensure_array(s.n) / medium_index)

        # Check that the scatterer is in a range we can compute for
        if x_arr.max() > 1e3:
            msg =  "radius too large, field calculation would take forever"
            raise InvalidScatterer(s, msg)

        if len(x_arr) == 1 and len(m_arr) == 1:
            # Could just use scatcoeffs_multi here, but jerome is in favor of
            # keeping the simpler single layer code here
            lmax = miescatlib.nstop(x_arr[0])
            return  miescatlib.scatcoeffs(m_arr[0], x_arr[0], lmax, self.eps1,
                                          self.eps2)
        else:
            return scatcoeffs_multi(m_arr, x_arr, self.eps1, self.eps2)

    def _scat_coeffs_internal(self, s, medium_wavevec, medium_index):
        '''
        Calculate expansion coefficients for Lorenz-Mie electric field
        inside a sphere.
        '''
        x_arr = medium_wavevec * ensure_array(s.r)
        m_arr = ensure_array(s.n) / medium_index

        # Check that the scatterer is in a range we can compute for
        if x_arr.max() > 1e3:
            msg = "radius too large, field calculation would take forever"
            raise InvalidScatterer(s, msg)

        if len(x_arr) == 1 and len(m_arr) == 1:
            # Could just use scatcoeffs_multi here, but jerome is in favor of
            # keeping the simpler single layer code here
            lmax = miescatlib.nstop(x_arr[0])
            return  miescatlib.internal_coeffs(m_arr[0], x_arr[0], lmax)

