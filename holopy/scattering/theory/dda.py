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
    def __init__(self, n_cpu = 1):
        # Check that adda is present and able to run
        try:
            subprocess.check_call(['adda', '-V'])
        except (subprocess.CalledProcessError, OSError):
            raise DependencyMissing('adda')

        self.n_cpu = n_cpu
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
        bound = scatterer.indicators.bound
        spacing = self.required_spacing(optics, scatterer.n)
        outf = tempfile.NamedTemporaryFile(dir = temp_dir, delete=False)
        line = "{p[0]} {p[1]} {p[2]}"
        n = _ensure_array(scatterer.n)
        if len(n) > 1:
            outf.write("Nmat={0}\n".format(len(n)))
            line += " {d}"
        line += '\n'

        for i, x in enumerate(np.arange(bound[0][0], bound[0][1], spacing)):
            for j, y in enumerate(np.arange(bound[1][0], bound[1][1], spacing)):
                for k, z in enumerate(np.arange(bound[2][0], bound[2][1], spacing)):
                    point = np.array((x, y, z)) + scatterer.location
                    domain = scatterer.in_domain(point)
                    if domain is not None:
                        # adda expects domain numbers to start with 1,
                        # holopy follows the python convention and has
                        # them start with 0
                        outf.write(line.format(p = (i, j, k), d = domain+1))
        outf.flush()

        cmd = []
        cmd.extend(['-shape', 'read', outf.name])
        cmd.extend(['-dpl', str(self._dpl(optics, scatterer.n))])
        cmd.extend(['-m'])
        for n in n:
            cmd.extend([str(n.real/optics.index), str(n.imag/optics.index)])

        return cmd



    @classmethod
    def _dpl(cls, optics, n):
        # if the object has multiple domains, we need to pick the
        # largest required dipole number
        n = np.abs(n)
        if not np.isscalar(n):
            n = max(n)
        dpl = 10*(n/optics.index)
        return dpl

    @classmethod
    def required_spacing(cls, optics, n):
        return optics.med_wavelen / cls._dpl(optics, n)


    def _calc_field(self, scatterer, schema):
        temp_dir = tempfile.mkdtemp()

        calc_points = schema.positions_kr_theta_phi(scatterer.location)

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

    def calc_field_volume(self, volume):
        """
        Compute the Electric field in a volume

        Parameters
        ----------
        volume : :class:`.VolumeScatterer`
            The volume to propagate light into, including the index of
            refraction at every point.  This volume must be at the correct
            spacing for dda.

        Returns
        -------
        field : :class:`.VectorField`
            The electric field at every point in the volume

        """

        temp_dir = tempfile.mkdtemp()
        print(temp_dir)

        indicies = np.array(sorted(set(volume.ravel())))
        # use the lowest index as the "medium"
        medium = indicies[0]
        # normalize with respect to the medium
        indicies /= medium
        index_lookup = dict([(b, a+1) for a, b in enumerate(indicies)])
        # make sure the medium index is different from unity so a-dda will place
        # dipoles so we can get a field value

        indicies[0] += 1e-4


        outf = tempfile.NamedTemporaryFile(dir = temp_dir, delete=False)
        outf.write("Nmat={0}\n".format(len(indicies)))
        for i in range(volume.shape[0]):
            for j in range(volume.shape[1]):
                for k in range(volume.shape[2]):
                    outf.write('{0} {1} {2} {3}\n'.format(
                        i, j, k, index_lookup[volume[i, j, k]]))
        outf.flush()

        cmd = compose_adda_command(indicies,
                                   self._dpl(volume.optics, indicies[-1]))
        cmd.extend(['-shape', 'read', outf.name])

        cmd.extend(['-store_int_field'])

        # we just want the internal fields here, so suppress calculating a
        # scattering matrix to save around a third on dda time.
        cmd.extend(['-scat_matr', 'none'])

        subprocess.check_call(cmd, cwd=temp_dir)

        result_dir = glob.glob(os.path.join(temp_dir, 'run000*'))[0]
        adda_result = np.loadtxt(os.path.join(result_dir, 'IntField-X'),
                                 skiprows=1)
        e_mag = np.sqrt(adda_result[:,3])
        e = np.zeros((adda_result.shape[0], 3), dtype='complex')
        for i in range(3):
            e[:,i] = (adda_result[:,(4+2*i)] + 1.0j *
                      adda_result[:,(5+2*i)]) * e_mag

        schema = VectorGridSchema.from_schema(volume)

        return schema.interpret_1d(e)

def compose_adda_command(indicies, dpl):
    cmd = ['adda']
    cmd.extend(['-dpl', str(dpl)])
    cmd.append('-m')
    for index in indicies:
        cmd.extend([str(np.real(index)), str(np.imag(index))])
    return cmd
