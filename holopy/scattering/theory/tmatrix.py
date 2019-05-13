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
"""
Compute holograms using Mishchenko's T-matrix method for axisymmetric scatterers.  Currently uses

.. moduleauthor:: Anna Wang <annawang@seas.harvard.edu>
"""
import subprocess
import tempfile
import os
import shutil
import copy
import warnings

import numpy as np

from holopy.scattering.scatterer import Sphere, Spheroid, Cylinder
from holopy.scattering.errors import TheoryNotCompatibleError, TmatrixFailure
from holopy.core.errors import DependencyMissing
from holopy.scattering.theory.scatteringtheory import ScatteringTheory
try:
    from holopy.scattering.theory.mie_f import mieangfuncs
    _NO_MIEANGFUNCS = False
except:
    _NO_MIEANGFUNCS = True

class Tmatrix(ScatteringTheory):
    """
    Computes scattering using the axisymmetric T-matrix solution 
    by Mishchenko with extended precision.

    It can calculate scattering from axisymmetric scatterers such as 
    cylinders and spheroids. Calculations for particles that are very 
    large or have high aspect ratios may not converge.


    Attributes
    ----------
    delete : bool (optional)
        If true (default), delete the temporary directory where we store the
        input and output file for the fortran executable

    Notes
    -----
    Does not handle near fields.  This introduces ~5% error at 10 microns.

    """
    def __init__(self, delete=True):
        self.delete = delete
        path, _ = os.path.split(os.path.abspath(__file__))
        self.tmatrix_executable = os.path.join(path, 'tmatrix_f', 'S')
        if os.name == 'nt':
            self.tmatrix_executable += '.exe'
        if not os.path.isfile(self.tmatrix_executable):
            raise DependencyMissing('Tmatrix', "Tmatrix code should compile "
            "with the rest of HoloPy. Check that you can compile fortran code "
            "from a makefile.")

        super().__init__()

    def _can_handle(self, scatterer):
        return isinstance(scatterer, Sphere) or isinstance(scatterer, Cylinder) \
            or isinstance(scatterer, Spheroid)

    def _run_tmat(self, temp_dir):
        # must give full path to executable even when specifying cwd keyword.
        # we'll run the executable from its location in the package tree
        subprocess.check_call(self.tmatrix_executable, cwd=temp_dir)
        # can replace the above with subprocess run in python 3.5 and higher
        return

    def _raw_scat_matrs(self, scatterer, pos, medium_wavevec, medium_index):
        temp_dir = tempfile.mkdtemp()

        angles = pos.T[:, 1:] * 180/np.pi
        outf = open(os.path.join(temp_dir, 'tmatrix_tmp.inp'), 'wb')

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
            # cleanup and raise error
            outf.close()
            shutil.rmtree(temp_dir)
            raise TheoryNotCompatibleError(self, scatterer)

        # write the info into the scattering angles file in the following order:
        outf.write((str((3/2)**iscyl*(rz*rxy**2)**(1/3.))+'\n').encode('utf-8'))
        outf.write((str(med_wavelen)+'\n').encode('utf-8'))
        outf.write((str(scatterer.n.real/medium_index)+'\n').encode('utf-8'))
        outf.write((str(scatterer.n.imag/medium_index)+'\n').encode('utf-8'))
        outf.write((str(rxy/rz)+'\n').encode('utf-8'))
        outf.write((str(scatterer.rotation[2]*180/np.pi)+'\n').encode('utf-8'))
        outf.write((str(scatterer.rotation[1]*180/np.pi)+'\n').encode('utf-8'))
        outf.write((str(-1 - iscyl)+'\n').encode('utf-8'))
        outf.write((str(angles.shape[0])+'\n').encode('utf-8'))

        # Now write all the angles
        np.savetxt(outf, angles)
        outf.close()

        self._run_tmat(temp_dir)
        try:
            tmat_result = np.loadtxt(os.path.join(temp_dir, 'tmatrix_tmp.out'))
        except (FileNotFoundError, OSError):
            #No output file
            raise TmatrixFailure(os.path.join(temp_dir, 'log'))
        if len(tmat_result)==0:
            #Output file is empty
            raise TmatrixFailure(os.path.join(temp_dir, 'log'))

        # columns in result are
        # s11.r s11.i s12.r s12.i s21.r s21.i s22.r s22.i
        # should be
        # s11 s12
        # s21 s22

        # Combine the real and imaginary components from the file into complex
        # numbers. Then scale by -ki due to Mishchenko's conventions in eq 5. of
        # Mishchenko, Applied Optics (2000).
        s = tmat_result[:,0::2] + 1.0j*tmat_result[:,1::2]
        s = s*(-2j*np.pi/med_wavelen)
        # Now arrange them into a scattering matrix, noting that Mishchenko's 
        #basis vectors are different from B/H, so we need to account for that.
        scat_matr = np.array([[s[:,0], s[:,1]], [-s[:,2], -s[:,3]]]).transpose()

        if self.delete:
            shutil.rmtree(temp_dir)

        return scat_matr

    def _raw_fields(self, pos, scatterer, medium_wavevec, medium_index,
                    illum_polarization):
        if not (np.array(illum_polarization)[:2] == np.array([1,0])).all():
            raise ValueError("Our implementation of Tmatrix scattering can only handle [1,0] polarization. Adjust your reference frame accordingly.")

        scat_matr = self._raw_scat_matrs(scatterer, pos, 
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
