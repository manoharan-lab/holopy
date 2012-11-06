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
import time

from nose.plugins.skip import SkipTest

from .scatteringtheory import ScatteringTheory
from .mie_f import mieangfuncs
from ..errors import TheoryNotCompatibleError
from ..scatterer import (Sphere, CoatedSphere, VoxelatedScatterer,
                         ScattererByFunction, MultidomainScattererByFunction,
                         Ellipsoid, Spheres)

class DependencyMissing(SkipTest, Exception):
    def __init__(self, dep):
        self.dep = dep
    def __str__(self):
        return "External Dependency: " + self.dep + " could not be found, terminating."

class DDA(ScatteringTheory):
    """
    Computes scattering using the the Discrete Dipole Approximation (DDA).

    It can (in principle) calculate scattering from any arbitrary scatterer.
    The DDA uses a numerical method that represents arbitrary scatterers as an array
    of point dipoles and then self-consistently solves Maxwell's equations
    to determine the scattered field. In practice, this model can be
    extremely computationally intensive, particularly if the size of the
    scatterer is larger than the wavelength of light.  This model requires an
    external scattering code: `a-dda <http://code.google.com/p/a-dda/>`_

    Attributes
    ----------
    imshape : float or tuple (optional)
        Size of grid to calculate scattered fields or
        intensities. This is the shape of the image that calc_field or
        calc_intensity will return
    phi : array
        Specifies azimuthal scattering angles to calculate (incident
        direction is z)
    theta : array
        Specifies polar scattering angles to calculate
    optics : :class:`holopy.optics.Optics` object
        specifies optical train

    Notes
    -----
    Does not handle near fields.  This introduces ~5% error at 10 microns.

    This can in principle handle any scatterer, but in practice it will need
    excessive memory or computation time for particularly large scatterers.
    """
    def __init__(self):
        # Check that adda is present and able to run
        try:
            subprocess.check_call(['adda', '-V'])
        except (subprocess.CalledProcessError, OSError):
            raise DependencyMissing('adda')

        super(DDA, self).__init__()

    def _run_adda(self, scatterer, optics, temp_dir):
        cmd = ['adda']
        cmd.extend(['-scat_matr', 'ampl'])
        cmd.extend(['-store_scat_grid'])
        cmd.extend(['-lambda', str(optics.med_wavelen)])
        cmd.extend(['-save_geom'])

        if isinstance(scatterer, Sphere):
            scat_args =  self._adda_sphere(scatterer, optics, temp_dir)
        elif isinstance(scatterer, CoatedSphere):
            scat_args = self._adda_coated(scatterer, optics, temp_dir)
        elif isinstance(scatterer, VoxelatedScatterer):
            scat_args = self._adda_general(scatterer, optics, temp_dir)
        elif isinstance(scatterer, ScattererByFunction):
            scat_args = self._adda_function_scatterer(scatterer, optics, temp_dir)
        elif isinstance(scatterer, Ellipsoid):
            scat_args = self._adda_ellipsoid(scatterer, optics, temp_dir)
        elif isinstance(scatterer, Spheres):
            scat_args = self._adda_bisphere(scatterer, optics, temp_dir)
        else:
            raise TheoryNotCompatibleError(self, scatterer)

        cmd.extend(scat_args)

        subprocess.check_call(cmd, cwd=temp_dir)

    def _adda_sphere(self, scatterer, optics, temp_dir):
        cmd = []
        cmd.extend(['-eq_rad', str(scatterer.r)])
        cmd.extend(['-m', str(scatterer.n.real/optics.index),
                    str(scatterer.n.imag/optics.index)])

        return cmd

    def _adda_ellipsoid(self, scatterer, optics, temp_dir):
        cmd = []
        cmd.extend(['-eq_rad', str(scatterer.r[0])])
        cmd.extend(['-shape', 'ellipsoid'])
        cmd.extend([str(r_i/scatterer.r[0]) for r_i in scatterer.r[1:]])
        cmd.extend(['-m', str(scatterer.n.real/optics.index),
                    str(scatterer.n.imag/optics.index)])

        return cmd

    def _adda_bisphere(self, scatterer, optics, temp_dir):
        # A-DDA bisphere only takes a pair of identical spheres.  We could handle
        # more complicated things by voxelating ourselves, but they are better
        # than us at voxelating, so lets use their restrictions for now.
        if (len(scatterer.r) != 2 or scatterer.r[0] != scatterer.r[1] or
            scatterer.n[0] != scatterer.n[1]):
            raise UnrealizableScatterer(self, scatterer, 'adda bisphere only '
                                        'works for 2 identical spheres')

        sep = hp.process.math.cartesian_distance(*scatterer.centers)

        cmd = []
        #        cmd.extend(['-size',
        #        str(scatterer.r[0]/self.optics.med_wavelen)])
        cmd.extend(['-eq_rad', str(scatterer.r[0])])
        cmd.extend(['-shape', 'bisphere', str(sep/(scatterer.r[0]*2))])
        cmd.extend(['-m', str(scatterer.n[0].real/optics.index),
                    str(scatterer.n[0].imag/optics.index)])

        return cmd


    def _adda_coated(self, scatterer, optics, temp_dir):
        cmd = []
        cmd.extend(['-eq_rad', str(scatterer.r[1])])
        cmd.extend(['-shape', 'coated', str(scatterer.r[0]/scatterer.r[1])])
        # A-DDA thinks of it as a sphere with an inclusion, so their first index
        # (the sphere) is our second, the outer layer.
        cmd.extend(['-m', str(scatterer.n[1].real/optics.index),
                    str(scatterer.n[1].imag/optics.index),
                    str(scatterer.n[0].real/optics.index),
                    str(scatterer.n[0].imag/optics.index)])

        return cmd

    def _adda_general(self, scatterer, optics, temp_dir):
        ms = []
        for n in scatterer.n:
            ms.append(str(n.real/optics.index))
            ms.append(str(n.imag/optics.index))

        shape = scatterer.write_adda_file(temp_dir)

        cmd = []
        cmd.extend(['-shape', 'read', shape.name])
        cmd.extend(['-dpl', str(scatterer.voxels_per_wavelen)])
        cmd.extend(['-m'])
        cmd.extend(ms)

        return cmd
        # TODO: figure out how adda is doing recentering and if we need to
        # adjust for that

    def _adda_function_scatterer(self, scatterer, optics, temp_dir):
        outf = tempfile.NamedTemporaryFile(dir = temp_dir, delete=False)
        if isinstance(scatterer, MultidomainScattererByFunction):
            outf.write("Nmat={0}\n".format(len(scatterer.domains)))
            for point in scatterer._points(self.required_spacing(optics, scatterer.n)):
                outf.write('{0} {1} {2} {3}'.format(point[0], point[1],
                                                    point[2], point[3]+1))
        else:
            for point in scatterer._points(self.required_spacing(optics, scatterer.n)):
                outf.write('{0} {1} {2}\n'.format(*point))
        outf.flush()

        cmd = []
        cmd.extend(['-shape', 'read', outf.name])
        cmd.extend(['-dpl', str(self._dpl(optics, scatterer.n))])
        cmd.extend(['-m'])
        def add_ns(domain):
            cmd.extend([str(domain.n.real/optics.index),
                    str(domain.n.imag/optics.index)])
        if isinstance(scatterer, MultidomainScattererByFunction):
            for domain in scatterer.domains:
                add_ns(domain)
        else:
            add_ns(scatterer)


        return cmd

    @classmethod
    def _dpl(cls, optics, n):
        return 10*(abs(n)/optics.index)

    @classmethod
    def required_spacing(cls, optics, n):
        return optics.med_wavelen / cls._dpl(optics, n)


    def _calc_field(self, scatterer, schema):
        time_start = time.time()

        temp_dir = tempfile.mkdtemp()

        calc_points = schema.positions_kr_theta_phi(scatterer.center)

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

        return self._finalize_fields(scatterer.z, fields, schema)

    def calc_field_volume(self, volume, incident):
        """
        Compute the Electric field in a volume

        Parameters
        ----------
        volume : :class:`.VolumeScatterer`
            The volume to propagate light into
        incident : :class:`.VectorField`
            The incident electric field

        Returns
        -------
        field : :class:`.VectorField`
            The electric field at every point in the volume

        """
        assert volume.shape == incident.shape
        assert volume.spacing == incident.spacing

        indicies = np.array(sorted(set(volume)))
        # use the lowest index as the "medium"
        medium = indicies[0]
        # normalize with respect to the medium
        indicies /= medium
        # make sure the medium index is different from unity so a-dda will place
        # dipoles so we can get a field value
        indicies[0] += 1e-4

        index_lookup = dict(enumerate(indicies))

        outf = tempfile.NamedTemporaryFile(dir = temp_dir, delete=False)
        outf.write("Nmat={0}\n".format(len(indicies)))
        for i in range(volume.shape[0]):
            for j in range(volume.shape[1]):
                for k in range(volume.shape[2]):
                    outf.write('{0} {1} {2} {3}\n'.format(
                        i, j, k, index_lookup[volume[i, j, k]]))

        cmd = []
        cmd.extend(['-shape', 'read', outf.name])
        cmd.extend(['-dpl', str(self._dpl(optics, scatterer.n))])
        cmd.extend(['-m', str()])
        cmd.extend(['-store_ind_field'])
