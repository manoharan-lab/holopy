import unittest

import numpy as np
from numpy.testing import assert_allclose

from holopy.scattering.theory import Mie, MieLens
from holopy.scattering.theory.lens import LensScatteringTheory
import holopy.scattering.tests.common as test_common

LENS_ANGLE = 1.
QLIM_TOL = {'atol': 1e-2, 'rtol': 1e-2}
MIE_THEORY = LensScatteringTheory(lens_angle=LENS_ANGLE, theory=Mie)

class TestLensScatteringTheory(unittest.TestCase):
    def test_can_handle(self):
        theory = MIE_THEORY
        self.assertTrue(theory._can_handle(test_common.sphere))

    def test_theta_quad_pts_min(self):
        min_ok = np.allclose(np.min(MIE_THEORY._theta_pts), 0., **QLIM_TOL)
        self.assertTrue(min_ok)

    def test_theta_quad_pts_max(self):
        max_ok = np.allclose(np.max(MIE_THEORY._theta_pts), LENS_ANGLE, **QLIM_TOL)
        self.assertTrue(max_ok)

    def test_theta_quad_pts_num(self):
        num_ok = len(MIE_THEORY._theta_pts) == len(MIE_THEORY._theta_wts)
        self.assertTrue(num_ok)

    def test_phi_quad_pts_min(self):
        min_ok = np.allclose(np.min(MIE_THEORY._phi_pts), 0., **QLIM_TOL)
        self.assertTrue(min_ok)

    def test_phi_quad_pts_max(self):
        max_ok = np.allclose(np.max(MIE_THEORY._phi_pts), 2*np.pi, **QLIM_TOL)
        self.assertTrue(max_ok)

    def test_phi_quad_pts_num(self):
        num_ok = len(MIE_THEORY._phi_pts) == len(MIE_THEORY._phi_wts)
        self.assertTrue(num_ok)

    def test_integrate_over_theta_with_quad_points(self):
        pts = MIE_THEORY._theta_pts
        wts = MIE_THEORY._theta_wts
        func = np.cos
        integral = np.sum(func(pts) * wts)
        expected_val = np.sin(1.) # analytic result
        assert_allclose(integral, expected_val)

    def test_integrate_over_phi_with_quad_points(self):
        pts = MIE_THEORY._phi_pts
        wts = MIE_THEORY._phi_wts
        func = lambda x: np.cos(np.pi * x)
        integral = np.sum(func(pts) * wts)
        expected_val = np.sin(2 * np.pi ** 2) /  np.pi # analytic result
        assert_allclose(integral, expected_val)

    def test_integrate_over_2D_with_quad_points(self):
        pts_theta = MIE_THEORY._theta_pts
        wts_theta = MIE_THEORY._theta_wts

        pts_phi = MIE_THEORY._phi_pts[:, np.newaxis]
        wts_phi = MIE_THEORY._phi_wts[:, np.newaxis]

        func = lambda theta, phi: np.cos(theta) * np.cos(np.pi * phi)
        integral = np.sum(func(pts_theta, pts_phi) * wts_theta * wts_phi)
        expected_val = np.sin(1.) * np.sin(2 * np.pi ** 2) /  np.pi
        assert_allclose(integral, expected_val)

    def test_quadrature_scattering_matrix_size(self):
        detector = test_common.xschema
        scatterer = test_common.sphere
        medium_wavevec = 2 * np.pi / test_common.wavelen
        medium_index = test_common.index

        theory = LensScatteringTheory(lens_angle=LENS_ANGLE, theory=Mie)

        quad_s_matrix = theory._compute_scattering_matrices_quad_pts(
                                       scatterer, medium_wavevec, medium_index)
        actual_size = quad_s_matrix.size
        expected_size = 4 * theory.quad_npts ** 2
        self.assertTrue(actual_size==expected_size)

    def test_lens_plus_mie_fields_same_as_mielens(self):
        detector = test_common.xschema
        scatterer = test_common.sphere
        medium_wavevec = 2 * np.pi / test_common.wavelen
        medium_index = test_common.index
        illum_polarization = test_common.xpolarization

        theory_old = MieLens(lens_angle=LENS_ANGLE)
        theory_new = LensScatteringTheory(lens_angle=LENS_ANGLE, theory=Mie)

        fields_old = theory_old.calculate_scattered_field(scatterer, detector)
        fields_new = theory_new.calculate_scattered_field(scatterer, detector)

        fields_ok = np.allclose(fields_old, fields_new)
        self.assertTrue(fields_ok)

if __name__ == '__main__':
    unittest.main()
