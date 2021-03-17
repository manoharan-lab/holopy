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
'''
Tests non-spherical T-matrix code calculations against Mie code

.. moduleauthor:: Anna Wang <annawang@seas.harvard.edu>
.. moduleauthor:: Ron Alexander <ralex0@users.noreply.github.com>
'''
import unittest

from numpy.testing import assert_raises, assert_allclose
import numpy as np
import pandas as pd
import yaml
from nose.plugins.attrib import attr
from nose.plugins.skip import SkipTest

import holopy as hp
from holopy.scattering import (
    Tmatrix, DDA, Sphere, Spheroid, Ellipsoid, Cylinder, calc_holo)
from holopy.scattering.theory import Mie
from holopy.core.errors import DependencyMissing
from holopy.core import detector_grid, update_metadata

from holopy.scattering.theory.tmatrix_f.S import ampld


SCHEMA = update_metadata(
    detector_grid(shape=20, spacing=0.1),
    illum_wavelen=.660, medium_index=1.33, illum_polarization=[1, 0])

MISHCHENKO_PARAMS = dict((
    ('axi', 10.0),
    ('rat', 0.1),
    ('lam', 2 * np.pi),
    ('mrr', 1.5),
    ('mri', 0.02),
    ('eps', 0.5),
    ('np', -1),
    ('ndgs', 2),
    ('alpha', 145.),
    ('beta', 52.),
    ('thet0', 56.),
    ('thet', 65.),
    ('phi0', 114.),
    ('phi', 128.),
    ('nang', 1)))


class TestTMatrix(unittest.TestCase):
    @attr('fast')
    def test_calc_scattering_matrix(self):
        """
        The original [ampld.lpd.f]
        (https://www.giss.nasa.gov/staff/mmishchenko/t_matrix.html) begins with
        this preamble:
        ------------------------------------------------------------------------
        This test result was calculated by the code in its
        curent setting.

        ICHOICE=1  NCHECK=1
        RAT= .963711
        PROLATE SPHEROIDS, A/B=   .5000000
        LAM=  6.283185   MRR= .1500D+01   MRI= .2000D-01
        ACCURACY OF COMPUTATIONS DDELT =  .10D-02
        EQUAL-SURFACE-AREA-SPHERE RADIUS= 10.0000
        thet0= 56.00  thet= 65.00  phi0=114.00  phi=128.00  alpha=145.00  beta= 52.00
        AMPLITUDE MATRIX
        S11=-.50941D+01 + i* .24402D+02
        S12=-.19425D+01 + i* .19971D+01
        S21=-.11521D+01 + i*-.30977D+01
        S22=-.69323D+01 + i* .24748D+02
        PHASE MATRIX
          650.3172  -17.9846   10.0498  -12.7580
          -21.1462  631.6322 -127.3059   87.2144
            6.8322  132.6131  635.2767  -34.7730
           -9.6629  -78.1229   51.4094  643.1738
         time =     .03 min
        ------------------------------------------------------------------------
        Here, we implement this test. For the amplitude scattering matrix. We,
        could, in principle, also calculate and test the phase matrix but this is
        currently not implemented.
        """
        params = MISHCHENKO_PARAMS
        expected_results = {'s11': -.50941E1 + .24402E2j,
                            's12': -.19425E1 + .19971E1j,
                            's21': -.11521E1 - .30977E1j,
                            's22': -.69323E1 + .24748E2j}
        s = ampld(*list(params.values()))
        results = {k: v for k, v in zip(['s11', 's12', 's21', 's22'], s)}
        ok = [np.allclose(x, y, atol=5e-4) for x, y in
              zip(expected_results.values(), results.values())]
        self.assertTrue(all(ok))

    @attr('slow')
    def test_sphere(self):
        s = Sphere(n=1.59, r=0.9, center=(2, 2, 80))
        mie_holo = calc_holo_safe(SCHEMA, s)
        tmat_holo = calc_holo_safe(SCHEMA, s, theory=Tmatrix)
        assert_allclose(mie_holo, tmat_holo, atol=.008)


    @attr("slow")
    def test_spheroid(self):
        s = Spheroid(n=1.5, r=[.4, 1.],
                     rotation=(0, np.pi/2, np.pi/2), center=(5, 5, 15))
        holo = calc_holo_safe(SCHEMA, s)
        test_values = _load_verification_data('tmatrix_spheroid')
        min_ok = np.allclose(test_values['min'], np.min(holo.values), rtol=1e-6)
        max_ok = np.allclose(test_values['max'], np.max(holo.values), rtol=1e-6)
        mean_ok = np.allclose(test_values['mean'], np.mean(holo.values), rtol=1e-6)
        std_ok = np.allclose(test_values['std'], np.std(holo.values), rtol=1e-6)
        self.assertTrue(all([min_ok, max_ok, mean_ok, std_ok]))


    @attr("slow")
    def test_cylinder(self):
        s = Cylinder(
            n=1.5, d=.8, h=2, rotation=(0, np.pi/2, np.pi/2), center=(5, 5, 15))
        holo = calc_holo_safe(SCHEMA, s)
        test_values = _load_verification_data('tmatrix_cylinder')
        min_ok = np.allclose(test_values['min'], np.min(holo.values), rtol=1e-6)
        max_ok = np.allclose(test_values['max'], np.max(holo.values), rtol=1e-6)
        mean_ok = np.allclose(test_values['mean'], np.mean(holo.values), rtol=1e-6)
        std_ok = np.allclose(test_values['std'], np.std(holo.values), rtol=1e-6)
        self.assertTrue(all([min_ok, max_ok, mean_ok, std_ok]))

    @attr("slow", "dda")
    def test_vs_dda(self):
        s = Spheroid(n=1.5, r=[.4, 1.],
                     rotation=(0, np.pi/2, np.pi/2), center=(5, 5, 50))
        try:
            dda_holo = calc_holo_safe(SCHEMA, s, theory=DDA)
        except DependencyMissing:
            raise SkipTest()
        tmat_holo = calc_holo_safe(SCHEMA, s, theory=Tmatrix)
        assert_allclose(dda_holo, tmat_holo, atol=.05)

    @attr("fast")
    def test_calc_scattering_matrix_multiple_angles(self):
        params = MISHCHENKO_PARAMS
        params['thet'] = np.ones(2) * params['thet']
        params['phi'] = np.ones(2) * params['phi']
        params['nang'] = 2
        s = ampld(*list(params.values()))
        self.assertTrue(len(s[0]) == 2)

    @attr("fast")
    def test_raw_scat_matrs_same_as_mie(self):
        theory_mie = Mie()
        theory_tmat = Tmatrix()

        pos = np.array([10, 0, 0])[:,None]
        s = Sphere(n=1.59, r=0.9, center=(2, 2, 80))

        s_mie = theory_mie.raw_scat_matrs(s, pos, 2*np.pi/.660, 1.33)
        s_tmat = theory_tmat.raw_scat_matrs(s, pos, 2*np.pi/.660, 1.33)
        self.assertTrue(np.allclose(s_mie, s_tmat))

    @attr("fast")
    def test_raw_fields_similar_to_mie(self):
        theory_mie = Mie(False, False)
        theory_tmat = Tmatrix()

        pos = np.array([10, 0, 0])[:,None]
        s = Sphere(n=1.59, r=0.9, center=(2, 2, 80))
        pol = pd.Series([1, 0])

        fields_mie = theory_mie.raw_fields(pos, s, 2*np.pi/.660, 1.33, pol)
        fields_tmat = theory_tmat.raw_fields(pos, s, 2*np.pi/.660, 1.33, pol)
        self.assertTrue(np.allclose(fields_mie, fields_tmat))


def calc_holo_safe(
        schema, scatterer, medium_index=None, illum_wavelen=None, **kwargs):
    try:
        holo = calc_holo(
            schema, scatterer, medium_index, illum_wavelen, **kwargs)
        return holo
    except DependencyMissing:
        raise SkipTest()


def _load_verification_data(name):
    hp_root = hp.__path__[0]
    fname = hp_root + '/scattering/tests/gold/gold_' + name + '.yaml'
    with open (fname, 'r') as f:
        test_values = yaml.safe_load(f)
    return test_values


if __name__ == '__main__':
    unittest.main()

