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
Compute holograms using the discrete dipole approximation (DDA).  Currently uses
ADDA (http://code.google.com/p/a-dda/) to do DDA calculations.

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
"""

#TODO: Adda currently fails if you call it with things specified in meters
#(values are too small), so we should probably nondimensionalize before talking
#to adda.

from __future__ import division

import numpy as np
import subprocess
import tempfile
import glob
import os
import shutil
import time
from ..binding_method import binding, finish_binding

from nose.plugins.skip import SkipTest

from .scatteringtheory import ScatteringTheory
from .mie_f import mieangfuncs
from ..scatterer import Sphere, Ellipsoid, Spheres
from ...core.marray import VectorGridSchema
from ...core.helpers import _ensure_array

class DependencyMissing(SkipTest, Exception):
    def __init__(self, dep):
        self.dep = dep
    def __str__(self):
        return "External Dependency: " + self.dep + " could not be found, terminating."

class DDA(ScatteringTheory):
    """
    Computes scattering using the the Discrete Dipole Approximation (DDA).

    It can (in principle) calculate scattering from any arbitrary scatterer.
    The DDA uses a numerical method that represents arbitrary scatterers as
    an array
    of point dipoles and then self-consistently solves Maxwell's equations
    to determine the scattered field. In practice, this model can be
    extremely computationally intensive, particularly if the size of the
    scatterer is larger than the wavelength of light.  This model requires an
    external scattering code: `a-dda <http://code.google.com/p/a-dda/>`_

    Attributes
    ----------
    n_cpu : int (optional)
        Number of threads to use for the DDA calculation
    max_dpl_size : float (optional)
        Force a maximum dipole size. This is useful for forcing extra dipoles if
        necessary to resolve features in an object. This may make dda
        calculations take much longer.
    Notes
    -----
    Does not handle near fields.  This introduces ~5% error at 10 microns.

    This can in principle handle any scatterer, but in practice it will need
    excessive memory or computation time for particularly large scatterers.
    """
    def __init__(self, n_cpu = 1, max_dpl_size=None):

        # Check that adda is present and able to run
        try:
            subprocess.check_call(['adda', '-V'])
        except (subprocess.CalledProcessError, OSError):
            raise DependencyMissing('adda')

        self.n_cpu = n_cpu
        self.max_dpl_size = max_dpl_size
        super(DDA, self).__init__()

    def _run_adda(self, scatterer, optics, temp_dir):
        if self.n_cpu == 1:
            cmd = ['adda']
        if self.n_cpu > 1:
            cmd = ['mpiexec', '-n', str(self.n_cpu), 'adda_mpi']
        cmd.extend(['-scat_matr', 'ampl'])
        cmd.extend(['-store_scat_grid'])
        cmd.extend(['-lambda', str(optics.med_wavelen)])
        cmd.extend(['-save_geom'])

        if isinstance(scatterer, Ellipsoid):
            scat_args = self._adda_ellipsoid(scatterer, optics, temp_dir)
        else:
            scat_args = self._adda_scatterer(scatterer, optics, temp_dir)

        cmd.extend(scat_args)

        subprocess.check_call(cmd, cwd=temp_dir)

    # TODO: figure out why our discritzation gives a different result
    # and fix so that we can use that and eliminate this.
    def _adda_ellipsoid(self, scatterer, optics, temp_dir):
        cmd = []
        cmd.extend(['-eq_rad', str(scatterer.r[0])])
        cmd.extend(['-shape', 'ellipsoid'])
        cmd.extend([str(r_i/scatterer.r[0]) for r_i in scatterer.r[1:]])
        cmd.extend(['-m', str(scatterer.n.real/optics.index),
                    str(scatterer.n.imag/optics.index)])

        return cmd

    def _adda_scatterer(self, scatterer, optics, temp_dir):
        spacing = self.required_spacing(optics, scatterer.n)
        outf = tempfile.NamedTemporaryFile(dir = temp_dir, delete=False)

        vox = scatterer.voxelate_domains(spacing)
        idx = np.concatenate([g[..., np.newaxis] for g in
                              np.mgrid[[slice(0,d) for d in vox.shape]]],
                             3).reshape((-1, 3))
        vox = vox.flatten()
        ns = _ensure_array(scatterer.n)
        n_domains = len(ns)
        if n_domains > 1:
            out = np.hstack((idx, vox[...,np.newaxis]))
            outf.write("Nmat={0}\n".format(n_domains))
        else:
            out = idx
        np.savetxt(outf, out[np.nonzero(vox)], fmt='%d')
        outf.close()

        cmd = []
        cmd.extend(['-shape', 'read', outf.name])
        cmd.extend(['-dpl', str(self._dpl(optics, scatterer.n))])
        cmd.extend(['-m'])
        for n in ns:
            cmd.extend([str(n.real/optics.index), str(n.imag/optics.index)])

        return cmd



    @classmethod
    @binding
    def _dpl(cls_self, optics, n):
        # if the object has multiple domains, we need to pick the
        # largest required dipole number
        n = np.abs(n)
        if not np.isscalar(n):
            n = max(n)
        dpl = 10*(n/optics.index)
        # This allows you to fix a largest allowable dipole size (ie
        # so you can resolve features in an object)
        if cls_self.max_dpl_size is not None:
            dpl = max(dpl, optics.med_wavelen / cls_self.max_dpl_size)
        return dpl

    @classmethod
    @binding
    def required_spacing(cls_self, optics, n):
        return optics.med_wavelen / cls_self._dpl(optics, n)


    def _calc_field(self, scatterer, schema, delete=True):
        temp_dir = tempfile.mkdtemp()

        calc_points = schema.positions.kr_theta_phi(scatterer.location, schema.optics)

        angles = calc_points[:,1:] * 180/np.pi

        outf = file(os.path.join(temp_dir, 'scat_params.dat'), 'w')

        # write the header on the scattering angles file
        header = ["global_type=pairs", "N={0}".format(len(angles)), "pairs="]
        outf.write('\n'.join(header)+'\n')
        # Now write all the angles
        np.savetxt(outf, angles)
        outf.close()

        self._run_adda(scatterer, schema.optics, temp_dir)

        # Go into the results directory, there should only be one run
        result_dir = glob.glob(os.path.join(temp_dir, 'run000*'))[0]
        if not delete:
            self._last_result_dir = result_dir

        adda_result = np.loadtxt(os.path.join(result_dir, 'ampl_scatgrid'),
                                 skiprows=1)
        # columns in result are
        # theta phi s1.r s1.i s2.r s2.i s3.r s3.i s4.r s4.i

        # Combine the real and imaginary components from the file into complex
        # numbers
        s = adda_result[:,2::2] + 1.0j*adda_result[:,3::2]

        # Now arrange them into a scattering matrix, see Bohren and Huffman p63
        # eq 3.12
        scat_matr = np.array([[s[:,1], s[:,2]], [s[:,3], s[:,0]]]).transpose()

        fields = np.zeros_like(calc_points, dtype = scat_matr.dtype)

        for i, point in enumerate(calc_points):
            kr, theta, phi = point
            escat_sph = mieangfuncs.calc_scat_field(kr, phi, scat_matr[i],
                                                    schema.optics.polarization)
            fields[i] = mieangfuncs.fieldstocart(escat_sph, theta, phi)

        if delete:
            shutil.rmtree(temp_dir)

        return self._finalize_fields(scatterer.z, fields, schema)
