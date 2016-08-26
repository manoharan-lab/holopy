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
Compute holograms using Mishchenko's T-matrix method for axisymmetric scatterers.  Can use double or extended precision.

.. moduleauthor:: Anna Wang <annawang@seas.harvard.edu>
"""

from __future__ import division

import numpy as np
import subprocess
import tempfile
import glob
import os
import shutil
import time
from ..binding_method import binding, finish_binding
from ..scatterer import Sphere, Spheroid

from nose.plugins.skip import SkipTest

from .scatteringtheory import ScatteringTheory
from .mie_f import mieangfuncs
from ...core.helpers import _ensure_array

class DependencyMissing(SkipTest, Exception):
    def __init__(self, dep):
        self.dep = dep
    def __str__(self):
        return "External Dependency: " + self.dep + " could not be found, terminating."

class Tmatrix(ScatteringTheory):
    """
    This model requires an external scattering code:
    http://www.giss.nasa.gov/staff/mmishchenko/t_matrix.html
    
    Relevant paper:
    Michael I. Mishchenko, Applied Optics, Vol. 39, Issue 6, pp. 1026-1031 (2000)

    The compiled code, S.exe computes scattering using the axisymmetric T-matrix solution by Mishchenko.

    It can calculate scattering from axisymmetric scatterers. Calculations for particles which are very large and have high aspect ratios may not converge unless extended precision is used. 
    -----
    Does not handle near fields.
    """
    def __init__(self):
        super(Tmatrix, self).__init__()

    def _run_tmat(self, scatterer, optics, temp_dir):
        cmd = ['./S.exe']
        subprocess.check_call(cmd, cwd=temp_dir)
        return

    def _calc_field(self, scatterer, schema, delete=True):
        temp_dir = tempfile.mkdtemp()
        current_directory = os.getcwd()
        path, _ = os.path.split(os.path.abspath(__file__))
        tmatrixlocation = os.path.join(path, 'tmatrix_doubleprecision', 'S.exe')
        shutil.copy(tmatrixlocation, temp_dir)
        os.chdir(temp_dir)
        calc_points = schema.positions.kr_theta_phi(scatterer.location, schema.optics)

        angles = calc_points[:,1:] * 180/np.pi
        outf = file(os.path.join(temp_dir, 'tmatrix_tmp.inp'), 'w')
 
        # write the info into the scattering angles file in the following order:

        if isinstance(scatterer, Sphere):
            outf.write(str(scatterer.r)+'\n')
            outf.write(str(schema.optics.med_wavelen)+'\n')
            outf.write(str(scatterer.n.real/schema.optics.index)+'\n')
            outf.write(str(scatterer.n.imag/schema.optics.index)+'\n')
            outf.write(str(scatterer.r[0]/scatterer.r[1])+'\n')
            outf.write(str(0)+'\n')
            outf.write(str(0)+'\n')
            outf.write(str(angles.shape[0])+'\n')
        elif scatterer.shape == -1:            
            outf.write(str((scatterer.r[1]*scatterer.r[0]**2)**(1/3.))+'\n')
            outf.write(str(schema.optics.med_wavelen)+'\n')
            outf.write(str(scatterer.n.real/schema.optics.index)+'\n')
            outf.write(str(scatterer.n.imag/schema.optics.index)+'\n')
            outf.write(str(scatterer.r[0]/scatterer.r[1])+'\n')
            outf.write(str(scatterer.rotation[1]*180/np.pi)+'\n')
            outf.write(str(scatterer.rotation[0]*180/np.pi)+'\n')
            outf.write(str(scatterer.shape)+'\n')
            outf.write(str(angles.shape[0])+'\n')
        elif scatterer.shape == -2:            
            outf.write(str((3/2.*scatterer.r[1]*scatterer.r[0]**2)**(1/3.))+'\n')
            outf.write(str(schema.optics.med_wavelen)+'\n')
            outf.write(str(scatterer.n.real/schema.optics.index)+'\n')
            outf.write(str(scatterer.n.imag/schema.optics.index)+'\n')
            outf.write(str(scatterer.r[0]/scatterer.r[1])+'\n')
            outf.write(str(scatterer.rotation[1]*180/np.pi)+'\n')
            outf.write(str(scatterer.rotation[0]*180/np.pi)+'\n')
            outf.write(str(scatterer.shape)+'\n')
            outf.write(str(angles.shape[0])+'\n')

        # Now write all the angles
        np.savetxt(outf, angles)
        outf.close()

        self._run_tmat(scatterer, schema.optics, temp_dir)

        # Go into the results directory
        result_file = glob.glob(os.path.join(temp_dir, 'tmatrix_tmp.out'))[0]

        tmat_result = np.loadtxt(result_file)
        # columns in result are
        # s11.r s11.i s12.r s12.i s21.r s21.i s22.r s22.i
        # should be 
        # s11 s12
        # s21 s22

        # Combine the real and imaginary components from the file into complex
        # numbers. Then scale by -ki due to Mishchenko's conventions in eq 5. of 
        # Mishchenko, Applied Optics (2000).
        s = tmat_result[:,0::2] + 1.0j*tmat_result[:,1::2]
        s = s*(-2j*np.pi/schema.optics.med_wavelen)
        # Now arrange them into a scattering matrix, noting that Mishchenko's basis
        # vectors are different from the B/H, so we need to take that into account:
        scat_matr = np.array([[s[:,0], s[:,1]], [-s[:,2], -s[:,3]]]).transpose()

        fields = np.zeros_like(calc_points, dtype = scat_matr.dtype)

        for i, point in enumerate(calc_points):
            kr, theta, phi = point
            postfactor = np.array([[np.cos(phi),np.sin(phi)],[-np.sin(phi),np.cos(phi)]])
            escat_sph = mieangfuncs.calc_scat_field(kr, phi, np.dot(scat_matr[i],postfactor),
                                                    schema.optics.polarization)
            fields[i] = mieangfuncs.fieldstocart(escat_sph, theta, phi)

        os.chdir(current_directory)

        if delete:
            shutil.rmtree(temp_dir)

        return self._finalize_fields(scatterer.z, fields, schema)
