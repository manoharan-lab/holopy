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

import subprocess
import tempfile
import shutil
import glob
import os
import numpy as np
from numpy.testing import assert_allclose
from .scatteringtheory import ScatteringTheory

class DependencyMissing(Exception):
    def __init__(self, dep):
        self.dep = dep
    def __str__(self):
        return "External Dependency: " + self.dep + " could not be found.  Is \
it installed and configured properly?"

class DDA(ScatteringTheory):
    """
    Scattering theory class that calculates holograms using the Discrete Dipole
    Approximation (DDA).  Can in principle handle any scatterer

    Attributes
    ----------
    imshape : float or tuple (optional)
        Size of grid to calculate scattered fields or
        intensities. This is the shape of the image that calc_field or
        calc_intensity will return
    phis : array 
        Specifies azimuthal scattering angles to calculate (incident
        direction is z)
    thetas : array 
        Specifies polar scattering angles to calculate
    optics : :class:`holopy.optics.Optics` object
        specifies optical train    

    Notes
    -----
    This can in principle handle any scatterer, but in practice it will need
    excessive memory or computation time for particularly large scatterers.  
    """
    def __init__(self, imshape=(256,256), thetas=None, phis=None, optics=None):
        # Check that adda is present and able to run
        try:
            subprocess.check_output(['adda', '-V'])
        except (subprocess.CalledProcessError, OSError):
            raise DependencyMissing('adda')

        super(DDA, self).__init__(imshape, thetas, phis, optics)

    def calc_field(self, scatterer):
        d = tempfile.mkdtemp()
        
        grid = self._spherical_grid(*scatterer.center)
        theta = grid[...,1].ravel()
        phi = grid[...,2].ravel()
        kr = grid[...,0].ravel()

        angles = np.vstack((theta, phi)).transpose()

        # Leave filename hardcoded for now since it is the default name for adda
        outf = file(os.path.join(d, 'scat_params.dat'), 'w')

        # write the header on the scattering angles file
        header = """global_type=pairs
N={0}
pairs=
""".format(len(phi))
        outf.write(header)
        # Now write all the angles
        np.savetxt(outf, angles)
        outf.close()

        # TODO, have it actually look at the scatterer 
        subprocess.check_call(['adda', '-scat_matr', 'ampl',
                               '-store_scat_grid'], cwd=d)

        # Go into the results directory, there should only be one run
        result_dir = glob.glob(os.path.join(d, 'run000*'))[0]

        adda_result = np.loadtxt(os.path.join(result_dir, 'ampl_scatgrid'),
                                 skiprows=1)
        # columns in result are
        # theta phi s1.r s1.i s2.r s2.i s3.r s3.i s4.r s4.i
        
        out_theta = adda_result[:,0]
        out_phi = adda_result[:,1]

        # Sanity check that the output angles are the same as the input ones
        # need relatively loose tolerances because adda appears to round off the
        # values we give it.  This may be a problem later, we will have to see
        assert_allclose(out_theta, theta, rtol=.1)
        assert_allclose(out_phi, phi, rtol=.1)
        # TODO: kr will not line up perfectly with the angles things were
        # actually calculated at, need to figure out which sets of coordinates
        # to use.  
        
        # Combine the real and imaginary components from the file into complex
        # numbers
        s = adda_result[:,2::2] + 1.0j*adda_result[:,3::2]

        # Now arrange them into a scattering matrix, see Bohren and Huffman p63
        # eq 3.12
        scat_matr = np.array([[s[:,1], s[:,2]], [s[:,3], s[:,0]]]).transpose()
        
        shutil.rmtree(d)

        return self.fields_from_scat_matr(scat_matr, kr)
        
