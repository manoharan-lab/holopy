# Copyright 2011-2013, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, and Ryan McGorty, Anna Wang, Solomon Barkley
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
Compute holograms using Mishchenko's T-matrix method for axisymmetric scatterers.  Currently uses

.. moduleauthor:: Anna Wang <annawang@seas.harvard.edu>
.. moduleauthor:: Ron Alexander <ralex0@users.noreply.github.com>
"""
import copy

import numpy as np

from holopy.scattering.scatterer import Sphere, Spheroid, Cylinder
from holopy.scattering.errors import TheoryNotCompatibleError, TmatrixFailure
from holopy.core.errors import DependencyMissing
from holopy.scattering.theory.scatteringtheory import ScatteringTheory
try:
    from holopy.scattering.theory.tmatrix_f.S import ampld
    COMPILED_TMATRIX_FORTRAN = True
except ModuleNotFoundError:
    COMPILED_TMATRIX_FORTRAN = False
try:
    from holopy.scattering.theory.mie_f import mieangfuncs
    _NO_MIEANGFUNCS = False
except ImportError:
    _NO_MIEANGFUNCS = True

class Tmatrix(ScatteringTheory):
    """
    Computes scattering using the axisymmetric T-matrix solution
    by Mishchenko with extended precision.

    It can calculate scattering from axisymmetric scatterers such as
    cylinders and spheroids. Calculations for particles that are very
    large or have high aspect ratios may not converge.

    Notes
    -----
    Does not handle near fields.  This introduces ~5% error at 10 microns.

    """
    def __init__(self):
        if not COMPILED_TMATRIX_FORTRAN:
            raise DependencyMissing("T-matrix theory", "This is probably "
                                    "due to a problem with compiling Fortran "
                                    "code, as it should be built with the rest"
                                    " of HoloPy through f2py.")
        super().__init__()

    def can_handle(self, scatterer):
        return isinstance(scatterer, Sphere) or isinstance(scatterer, Cylinder) \
            or isinstance(scatterer, Spheroid)

    # FIXME why is S (scatterer, pos, ...) but fields are (pos, scatterer, ...)?
    def raw_scat_matrs(self, scatterer, pos, medium_wavevec, medium_index):
        args = self._parse_args(scatterer, pos, medium_wavevec, medium_index)
        s = self._run_tmat(args)
        return s

    def _parse_args(self, scatterer, pos, medium_wavevec, medium_index):
        """Parses inputs into form usable by tmatrix_f. The definitions of
        the aruguments can be found in "Scattering, Absorbtion, and Emission of
        Light by Small Particles" by Mishchenko, Travis and Lacis in Chapter 5.

        The incident polarization is set to (1, 0)
        """
        angles = pos.T[:, 1:] * 180/np.pi

        med_wavelen = 2*np.pi/medium_wavevec
        if isinstance(scatterer, Sphere):
            rxy = scatterer.r
            rz = scatterer.r
            iscyl = False
            scatterer = copy.copy(scatterer)
            scatterer.rotation = (0,0,0)
        elif isinstance(scatterer, Spheroid):
            rxy = scatterer.r[0]
            rz = scatterer.r[1]
            iscyl = False
        elif isinstance(scatterer, Cylinder):
            rxy = scatterer.d/2
            rz = scatterer.h/2
            iscyl = True
        else:
            raise TheoryNotCompatibleError(self, scatterer)

        axi = (3/2)**iscyl*(rz*rxy**2)**(1/3.)
        rat = 1
        lam = med_wavelen
        mrr = scatterer.n.real/medium_index
        mri = scatterer.n.imag/medium_index
        eps = rxy/rz
        NP = -1 - int(iscyl)
        ndgs = 5
        alpha = scatterer.rotation[2] * 180 / np.pi
        beta = scatterer.rotation[1] * 180 / np.pi

        # FIXME: Why does the incident polarization have to be set to  (1, 0)?
        thet0 = 0
        thet = angles[:, 0]
        phi0 = 0
        phi = angles[:, 1]
        nang = angles.shape[0]

        args = [axi, rat, lam, mrr, mri, eps, NP, ndgs, alpha, beta,
                thet0, thet, phi0, phi, nang]

        return args

    def _run_tmat(self, args):
        med_wavelen = args[2]
        nang = args[-1]
        s11, s12, s21, s22 = ampld(*args)
        for s in [s11, s12, s21, s22]:
            s *= (-2j*np.pi/med_wavelen)
        scat_matr = np.array([[s11, s12], [s21, s22]]).transpose()
        return scat_matr

    def raw_fields(self, pos, scatterer, medium_wavevec, medium_index,
                    illum_polarization):
        if not (np.array(illum_polarization)[:2] == np.array([1,0])).all():
            msg = ("Our implementation of Tmatrix scattering can only " +
                   "handle [1,0] polarization. Adjust your reference " +
                   "frame accordingly.")
            raise ValueError(msg)

        scat_matr = self.raw_scat_matrs(scatterer, pos,
                    medium_wavevec=medium_wavevec, medium_index=medium_index)
        fields = np.zeros_like(pos.T, dtype = scat_matr.dtype)

        if _NO_MIEANGFUNCS:
            warnings.warn("Problem with holopy.scattering.theory.mie_f.mieang"
                          "funcs. This is probably due to a problem compiling"
                          "Fortran code. Returning scattering matrices only,"
                          "not fields. Subsequent calculations will fail.")
            return scat_matr

        for i, point in enumerate(pos.T):
            kr, theta, phi = point
            # TODO: figure out why postfactor is needed -- it is not used in dda.py
            postfactor = np.array([[np.cos(phi),np.sin(phi)],
                                   [-np.sin(phi),np.cos(phi)]])
            escat_sph = mieangfuncs.calc_scat_field(kr, phi,
                                    np.dot(scat_matr[i],postfactor), [1,0])
            fields[i] = mieangfuncs.fieldstocart(escat_sph, theta, phi)
        return fields.T
