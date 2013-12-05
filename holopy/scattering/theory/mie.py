# Copyright 2011-2013, Vinothan N. Manoharan, Thomas G. Dimiduk,
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
from ..errors import TheoryNotCompatibleError, UnrealizableScatterer
from ..scatterer import Sphere, Scatterers
from .scatteringtheory import FortranTheory
from .mie_f import mieangfuncs, miescatlib
from .mie_f.multilayer_sphere_lib import scatcoeffs_multi
import copy


class Mie(FortranTheory):
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
    """

    # don't need to define __init__() because we'll use the base class
    # constructor

    def __init__(self, compute_escat_radial = True):
        self.compute_escat_radial = compute_escat_radial
        # call base class constructor
        super(Mie, self).__init__()

    def _can_handle(self, scatterer):
        return isinstance(scatterer, Sphere)

    def _calc_scat_matrix(self, scatterer, schema):
        if isinstance(scatterer, Sphere):
            scat_coeffs = self._scat_coeffs(scatterer, schema.optics)

            # TODO: actually use (rather than ignore) the phi
            scat_matrs = [mieangfuncs.asm_mie_far(scat_coeffs, theta) for
                          theta, phi in schema.positions_theta_phi()]
            return np.array(scat_matrs)
        else:
            raise TheoryNotCompatibleError(self, scatterer)

    def _raw_fields(self, positions, scatterer, optics):
        scat_coeffs = self._scat_coeffs(scatterer, optics)
        return mieangfuncs.mie_fields(positions, scat_coeffs, optics.polarization,
                                      self.compute_escat_radial)

    def _raw_internal_fields(self, positions, scatterer, optics):
        scat_coeffs = self._scat_coeffs(scatterer, optics)
        # TODO BUG: this isn't right for layered spheres (and will
        # probably crash)
        return mieangfuncs.mie_internal_fields(positions, scatterer.n,
                                               scat_coeffs, optics.polarization)


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

        Notes
        -----
        The radiation pressure cross section C_pr is given by
        C_pr = C_ext - <cos \theta> C_sca.

        The radiation pressure force on a sphere is

        F = (n_med I_0 C_pr) / c

        where I_0 is the incident intensity.  See van de Hulst, p. 14.
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
            return  miescatlib.scatcoeffs(m_arr[0], x_arr[0], lmax)
        else:
            return scatcoeffs_multi(m_arr, x_arr)


    def _scat_coeffs_internal(self, s, optics):
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
            return  miescatlib.internal_coeffs(m_arr[0], x_arr[0], lmax)
        # else:
#             return scatcoeffs_multi(m_arr, x_arr)
