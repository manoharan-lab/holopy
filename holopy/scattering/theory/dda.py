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
"""
Compute holograms using the discrete dipole approximation (DDA).  Currently uses
ADDA (https://github.com/adda-team/adda) to do DDA calculations.
.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
"""

#TODO: Adda currently fails if you call it with things specified in meters
#(values are too small), so we should probably nondimensionalize before talking
#to adda.

import subprocess
import tempfile
import glob
import os
import shutil
import time
import warnings

import numpy as np

from holopy.core.utils import ensure_array, SuppressOutput
from holopy.scattering.scatterer import (
    Ellipsoid, Capsule, Cylinder, Bisphere, Sphere, Scatterer, Spheroid)
from holopy.core.errors import DependencyMissing
from holopy.scattering.theory.scatteringtheory import ScatteringTheory


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
        Force a maximum dipole size. This is useful for forcing extra
        dipoles if necessary to resolve features in an object. This may
        make dda calculations take much longer.
    use_indicators : bool
        If true, a scatterer's indicators method will be used instead of
        its built-in adda definition
    keep_raw_calculations : bool
        If true, do not delete the temporary file we run ADDA in,
        instead print its path so you can inspect its raw results

    Notes
    -----
    Does not handle near fields. This introduces ~5% error at 10
    microns. This can in principle handle any scatterer, but in practice
    it will need excessive memory or computation time for particularly
    large scatterers.
    """
    def __init__(self, n_cpu=1, use_gpu=False, gpu_id=None, max_dpl_size=None,
                 use_indicators=True, keep_raw_calculations=False, addacmd=[],
                 suppress_C_output=True):

        # Check that adda is present and able to run
        try:
            with SuppressOutput(suppress_output=suppress_C_output):
                subprocess.check_call(['adda', '-V'])
        except (subprocess.CalledProcessError, OSError):
            raise DependencyMissing('adda', "adda is not included with HoloPy "
                "and must be installed separately. You should be able to run "
                "the command 'adda' from a terminal.")

        self.n_cpu = n_cpu
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.max_dpl_size = max_dpl_size
        self.use_indicators = use_indicators
        self.keep_raw_calculations = keep_raw_calculations
        self.addacmd = addacmd
        self.suppress_C_output = suppress_C_output
        if use_gpu and n_cpu>1: warnings.warn("Adda cannot run on multiple CPUs, when running on GPU. 1 CPU will be used.")
        super().__init__()

    @classmethod
    def can_handle(cls, scatterer):
        # For now DDA is our most general theory, eventually this will have to
        # change if we add other theorys that can compute things ADDA can't (or
        # shouldn't, because it would take crazy long)
        return isinstance(scatterer, Scatterer)

    def _run_adda(self, scatterer, medium_wavevec, medium_index, temp_dir):
        medium_wavelen = 2*np.pi/medium_wavevec
        if self.use_gpu:
            cmd = ['adda_ocl']
            if self.gpu_id is not None: cmd.extend(['-gpu',str(self.gpu_id)])
        elif self.n_cpu == 1:
            cmd = ['adda']
        elif self.n_cpu > 1:
            cmd = ['mpiexec', '-n', str(self.n_cpu), 'adda_mpi']
        cmd.extend(['-scat_matr', 'ampl'])
        cmd.extend(['-store_scat_grid'])
        cmd.extend(['-lambda', str(medium_wavelen)])
        cmd.extend(['-save_geom'])
        cmd.extend(self.addacmd)

        predefined = isinstance(scatterer, tuple(_get_predefined_shape.keys()))
        layered=isinstance(scatterer, Sphere) and not np.isscalar(scatterer.r)
        if not predefined or self.use_indicators or layered:
            scat_args = self._adda_discretized(scatterer, medium_wavelen, medium_index, temp_dir)
        else:
            scat_args = self._adda_predefined(scatterer, medium_wavelen, medium_index, temp_dir)
        cmd.extend(scat_args)
        with SuppressOutput(suppress_output=self.suppress_C_output):
            subprocess.check_call(cmd, cwd=temp_dir)

    # TODO: figure out why our discretization gives a different result
    # and fix so that we can use that and eliminate this.
    def _adda_predefined(self, scatterer, medium_wavelen, medium_index, temp_dir):
        scatterer_pars = _get_predefined_shape[scatterer.__class__](scatterer)
        cmd = []
        cmd.extend(['-eq_rad', str(scatterer_pars[0]), '-shape'])
        cmd.extend(scatterer_pars[1])
        cmd.extend(['-m', str(scatterer.n.real/medium_index),
                    str(scatterer.n.imag/medium_index)])
        if hasattr(scatterer, 'rotation'):
            cmd.extend(['-orient'])
            cmd.extend([str(angle*180/np.pi) for angle in reversed(scatterer.rotation)])
            # rotation angles are gamma, beta, alpha in adda reference frame
        return cmd

    def _adda_discretized(self, scatterer, medium_wavelen, medium_index, temp_dir):
        spacing = self.required_spacing(scatterer.bounds, medium_wavelen, medium_index, scatterer.n)
        outf = tempfile.NamedTemporaryFile(dir = temp_dir, delete=False)

        vox = scatterer.voxelate_domains(spacing)
        idx = np.concatenate([g[..., np.newaxis] for g in
                              np.mgrid[[slice(0,d) for d in vox.shape]]],
                             3).reshape((-1, 3))
        vox = vox.flatten()
        ns = ensure_array(scatterer.n)
        n_domains = len(ns)
        if n_domains > 1:
            out = np.hstack((idx, vox[...,np.newaxis]))
            outf.write("Nmat={0}\n".format(n_domains).encode('utf-8'))
        else:
            out = idx
        np.savetxt(outf, out[np.nonzero(vox)], fmt='%d')
        outf.close()

        cmd = []
        cmd.extend(['-shape', 'read', outf.name])
        cmd.extend(
            ['-dpl', str(self._dpl(scatterer.bounds, medium_wavelen, medium_index, scatterer.n))])
        cmd.extend(['-m'])
        for n in ns:
            m = n.real/medium_index
            if m == 1:
                warnings.warn("Adda cannot compute particles with index equal to medium index, adjusting particle index {} to {}".format(m, m+1e-6))
                m += 1e-6
            cmd.extend([str(m), str(n.imag/medium_index)])
        return cmd

    def _dpl(self, bounds, medium_wavelen, medium_index, n):
        # for objects much smaller than wavelength we should use
        # at least 10 dipoles per smallest dimension
        dpl = 10*medium_wavelen / min([np.abs(b[1]-b[0]) for b in bounds])
        # if the object has multiple domains, we need to pick the
        # largest required dipole number
        n = np.abs(n)
        if not np.isscalar(n):
            n = max(n)
        dpl = max(dpl, 10*(n/medium_index))
        # This allows you to fix a largest allowable dipole size (ie
        # so you can resolve features in an object)
        if self.max_dpl_size is not None:
            dpl = max(dpl, medium_wavelen / self.max_dpl_size)
        return dpl

    def required_spacing(self, bounds, medium_wavelen, medium_index, n):
        return medium_wavelen / self._dpl(bounds, medium_wavelen, medium_index, n)

    def raw_scat_matrs(self, scatterer, pos, medium_wavevec, medium_index):
        angles = pos.T[:, 1:] * 180/np.pi
        temp_dir = tempfile.mkdtemp()

        outf = open(os.path.join(temp_dir, 'scat_params.dat'), 'wb')

        # write the header on the scattering angles file
        header = ["global_type=pairs", "N={0}".format(len(angles)), "pairs="]
        outf.write(('\n'.join(header)+'\n').encode('utf-8'))
        # Now write all the angles
        np.savetxt(outf, angles)
        outf.close()

        self._run_adda(
            scatterer, medium_wavevec=medium_wavevec,
            medium_index=medium_index, temp_dir=temp_dir)

        # Go into the results directory, there should only be one run
        result_dir = glob.glob(os.path.join(temp_dir, 'run000*'))[0]
        if self.keep_raw_calculations:
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

        if self.keep_raw_calculations:
            print(("Raw calculations are in: {0}".format(temp_dir)))
        else:
            shutil.rmtree(temp_dir)

        return scat_matr


_get_predefined_shape = {
        Ellipsoid: lambda s:(s.r[0], ['ellipsoid'] +
                                        [str(r_i/s.r[0]) for r_i in s.r[1:]]),
        Spheroid: lambda s: (s.r[0], ['ellipsoid', '1', str(s.r[1]/s.r[0])]),
        Capsule: lambda s: ((s.h+s.d)/2, ['capsule', str(s.h/s.d)]),
        Cylinder: lambda s: (s.h/2, ['cylinder', str(s.h/s.d)]),
        Bisphere: lambda s: ((s.h+s.d)/2, ['bisphere', str(s.h/s.d)]),
        Sphere: lambda s: (s.r, ['sphere'])}
