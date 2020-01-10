import unittest

import numpy as np

from holopy.scattering.theory import Mie, MieLens
from holopy.scattering.theory.lens import LensScatteringTheory
import holopy.scattering.tests.common as test_common


class TestLensScatteringTheory(unittest.TestCase):
    def test_can_handle(self):
        theory = Mie
        lens_angle = 1.0
        lens_theory = LensScatteringTheory(lens_angle=lens_angle, theory=theory)
        self.assertTrue(lens_theory._can_handle(test_common.sphere))

    def test_quadrature_scattering_matrix_size(self):
        detector = test_common.xschema
        scatterer = test_common.sphere
        medium_wavevec = 2 * np.pi / test_common.wavelen
        medium_index = test_common.index
        lens_angle = 1.0

        theory = LensScatteringTheory(lens_angle=lens_angle, theory=Mie)

        quad_s_matrix = theory._compute_scattering_matrices_quad_pts(
                                       scatterer, medium_wavevec, medium_index)
        actual_size = quad_s_matrix.size
        expected_size = 4 * theory._theta_pts.size * theory._phi_pts.size
        self.assertTrue(actual_size==expected_size)


    def test_lens_plus_mie_fields_same_as_mielens(self):
        detector = test_common.xschema
        scatterer = test_common.sphere
        medium_wavevec = 2 * np.pi / test_common.wavelen
        medium_index = test_common.index
        illum_polarization = test_common.xpolarization
        lens_angle = 1.0

        theory_old = MieLens(lens_angle=lens_angle)
        theory_new = LensScatteringTheory(lens_angle=lens_angle, theory=Mie)

        fields_old = theory_old.calculate_scattered_field(scatterer, detector)
        fields_new = theory_new.calculate_scattered_field(scatterer, detector)

        fields_ok = np.allclose(fields_old, fields_new)
        self.assertTrue(fields_ok)

if __name__ == '__main__':
    unittest.main()
