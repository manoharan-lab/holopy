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
ADDA (http://code.google.com/p/a-dda/) to do DDA calculations.

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
"""

#TODO: Adda currently fails if you call it with things specified in meters
#(values are too small), so we should probably nondimensionalize before talking
#to adda.



import numpy as np
import subprocess
import tempfile
import glob
import os
import shutil
import time
import warnings

from .scatteringtheory import ScatteringTheory
from .mie_f import mieangfuncs
from ..scatterer import Sphere, Ellipsoid, Spheres, Capsule, Cylinder, Bisphere, Sphere_builtin, JanusSphere
from holopy.scattering.scatterer.csg import CsgScatterer
from ...core.utils import _ensure_array
from ..errors import DependencyMissing

scatterers_handled = Sphere, JanusSphere, Ellipsoid, Spheres, Capsule, Cylinder, Bisphere, Sphere_builtin, CsgScatterer

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
    keep_raw_calculations : bool
        If true, do not delete the temporary file we run ADDA in, instead print
        its path so you can inspect its raw results
    Notes
    -----
    Does not handle near fields.  This introduces ~5% error at 10 microns.

    This can in principle handle any scatterer, but in practice it will need
    excessive memory or computation time for particularly large scatterers.
    """
    def __init__(self, n_cpu = 1, max_dpl_size=None, keep_raw_calculations=False,
            addacmd=[]):

        # Check that adda is present and able to run
        try:
            subprocess.check_call(['adda', '-V'])
        except (subprocess.CalledProcessError, OSError):
            raise DependencyMissing('adda')

        self.n_cpu = n_cpu
        self.max_dpl_size = max_dpl_size
        self.keep_raw_calculations = keep_raw_calculations
        self.addacmd = addacmd
        super().__init__()

    def _can_handle(self, scatterer):
        # TODO: replace this with actually determining if dda we can handle it (though this isn't too far off, since dda can handle almost anything)
        return True

    def _run_adda(self, scatterer, medium_wavevec, medium_index, temp_dir):
        medium_wavelen = 2*np.pi/medium_wavevec
        if self.n_cpu == 1:
            cmd = ['adda']
        if self.n_cpu > 1:
            cmd = ['mpiexec', '-n', str(self.n_cpu), 'adda_mpi']
        cmd.extend(['-scat_matr', 'ampl'])
        cmd.extend(['-store_scat_grid'])
        cmd.extend(['-lambda', str(medium_wavelen)])
        cmd.extend(['-save_geom'])
        cmd.extend(self.addacmd)

        if isinstance(scatterer, Ellipsoid):
            scat_args = self._adda_ellipsoid(scatterer, medium_wavelen, medium_index, temp_dir)
        elif isinstance(scatterer, Capsule):
            scat_args = self._adda_capsule(scatterer, medium_wavelen, medium_index, temp_dir)
        elif isinstance(scatterer, Cylinder):
            scat_args = self._adda_cylinder(scatterer, medium_wavelen, medium_index, temp_dir)
        elif isinstance(scatterer, Bisphere):
            scat_args = self._adda_bisphere(scatterer, medium_wavelen, medium_index, temp_dir)
        elif isinstance(scatterer, Sphere_builtin):
            scat_args = self._adda_sphere_builtin(scatterer, medium_wavelen, medium_index, temp_dir)
        else:
            scat_args = self._adda_scatterer(scatterer, medium_wavelen, medium_index, temp_dir)

        cmd.extend(scat_args)

        subprocess.check_call(cmd, cwd=temp_dir)

    # TODO: figure out why our discritzation gives a different result
    # and fix so that we can use that and eliminate this.
    def _adda_ellipsoid(self, scatterer, medium_wavelen, medium_index, temp_dir):
        cmd = []
        cmd.extend(['-eq_rad', str(scatterer.r[0])])
        cmd.extend(['-shape', 'ellipsoid'])
        cmd.extend([str(r_i/scatterer.r[0]) for r_i in scatterer.r[1:]])
        cmd.extend(['-m', str(scatterer.n.real/medium_index),
                    str(scatterer.n.imag/medium_index)])
        cmd.extend(['-orient'])
        cmd.extend([str(angle) for angle in scatterer.rotation])

        return cmd

    def _adda_capsule(self, scatterer, medium_wavelen, medium_index, temp_dir):
        cmd = []
        cmd.extend(['-eq_rad', str((scatterer.h+scatterer.d)/2.0)])
        cmd.extend(['-shape', 'capsule'])
        cmd.extend([str(scatterer.h/scatterer.d)])
        cmd.extend(['-m', str(scatterer.n.real/medium_index),
                    str(scatterer.n.imag/mediumindex)])
        cmd.extend(['-orient'])
        cmd.extend([str(angle) for angle in scatterer.rotation])

        return cmd

    def _adda_cylinder(self, scatterer, medium_wavelen, medium_index, temp_dir):
        cmd = []
        cmd.extend(['-eq_rad', str(scatterer.h/2.0)])
        cmd.extend(['-shape', 'cylinder'])
        cmd.extend([str(scatterer.h/scatterer.d)])
        cmd.extend(['-m', str(scatterer.n.real/medium_index),
                    str(scatterer.n.imag/medium_index)])
        cmd.extend(['-orient'])
        cmd.extend([str(angle) for angle in scatterer.rotation])

        return cmd

    def _adda_bisphere(self, scatterer, medium_wavelen, medium_index, temp_dir):
        cmd = []
        cmd.extend(['-eq_rad', str((scatterer.h+scatterer.d)/2.0)])
        cmd.extend(['-shape', 'bisphere'])
        cmd.extend([str(scatterer.h/scatterer.d)])
        cmd.extend(['-m', str(scatterer.n.real/medium_index),
                    str(scatterer.n.imag/medium_index)])
        cmd.extend(['-orient'])
        cmd.extend([str(angle) for angle in scatterer.rotation])

        return cmd

    def _adda_sphere_builtin(self, scatterer, medium_wavelen, medium_index, temp_dir):
        cmd = []
        cmd.extend(['-eq_rad', str(scatterer.r)])
        cmd.extend(['-shape', 'sphere'])
        cmd.extend(['-m', str(scatterer.n.real/medium_index),
                    str(scatterer.n.imag/medium_index)])
        return cmd

    def _adda_scatterer(self, scatterer, medium_wavelen, medium_index, temp_dir):
        spacing = self.required_spacing(medium_wavelen, medium_index, scatterer.n)
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
            outf.write("Nmat={0}\n".format(n_domains).encode('utf-8'))
        else:
            out = idx
        np.savetxt(outf, out[np.nonzero(vox)], fmt='%d')
        outf.close()

        cmd = []
        cmd.extend(['-shape', 'read', outf.name])
        cmd.extend(['-dpl', str(self._dpl(medium_wavelen, medium_index, scatterer.n))])
        cmd.extend(['-m'])
        for n in ns:
            m = n.real/medium_index
            if m == 1:
                warnings.warn("Adda cannot compute particles with index equal to medium index, adjusting particle index {} to {}".format(m, m+1e-6))
                m += 1e-6
            cmd.extend([str(m), str(n.imag/medium_index)])
        return cmd



    def _dpl(self, medium_wavelen, medium_index, n):
        # if the object has multiple domains, we need to pick the
        # largest required dipole number
        n = np.abs(n)
        if not np.isscalar(n):
            n = max(n)
        dpl = 10*(n/medium_index)
        # This allows you to fix a largest allowable dipole size (ie
        # so you can resolve features in an object)
        if self.max_dpl_size is not None:
            dpl = max(dpl, medium_wavelen / self.max_dpl_size)
        return dpl

    def required_spacing(self, medium_wavelen, medium_index, n):
        return medium_wavelen / self._dpl(medium_wavelen, medium_index, n)

    def _raw_scat_matrs(self, scatterer, pos, medium_wavevec, medium_index):
        angles = pos[:, 1:] * 180/np.pi
        temp_dir = tempfile.mkdtemp()

        outf = open(os.path.join(temp_dir, 'scat_params.dat'), 'wb')

        # write the header on the scattering angles file
        header = ["global_type=pairs", "N={0}".format(len(angles)), "pairs="]
        outf.write(('\n'.join(header)+'\n').encode('utf-8'))
        # Now write all the angles
        np.savetxt(outf, angles)
        outf.close()

        self._run_adda(scatterer, medium_wavevec=medium_wavevec, medium_index=medium_index, temp_dir=temp_dir)

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

    def _raw_fields(self, pos, scatterer, medium_wavevec, medium_index, illum_polarization):
        pos = pos.T
        scat_matr = self._raw_scat_matrs(scatterer, pos, medium_wavevec=medium_wavevec, medium_index=medium_index)
        fields = np.zeros_like(pos, dtype = scat_matr.dtype)

        for i, point in enumerate(pos):
            kr, theta, phi = point
            escat_sph = mieangfuncs.calc_scat_field(kr, phi, scat_matr[i],
                                                    illum_polarization.values[:2])
            fields[i] = mieangfuncs.fieldstocart(escat_sph, theta, phi)
        return fields.T
