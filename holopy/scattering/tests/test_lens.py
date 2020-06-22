import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_equal

try:
    import numexpr as ne
    NUMEXPR_INSTALLED = True
except ModuleNotFoundError:
    NUMEXPR_INSTALLED = False

from holopy.core import detector_points, update_metadata, detector_grid
from holopy.scattering import calc_holo
from holopy.scattering.theory import Mie, MieLens
from holopy.scattering.theory.lens import Lens
from holopy.scattering.theory.mielensfunctions import MieLensCalculator
import holopy.scattering.tests.common as test_common

LENS_ANGLE = 1.
QLIM_TOL = {'atol': 1e-2, 'rtol': 1e-2}
LENSMIE = Lens(lens_angle=LENS_ANGLE, theory=Mie(False, False))


SMALL_DETECTOR = update_metadata(
    detector_grid(shape=16, spacing=test_common.pixel_scale),
    illum_wavelen=test_common.wavelen,
    medium_index=test_common.index,
    illum_polarization=test_common.xpolarization)


class TestLens(unittest.TestCase):
    def test_can_handle(self):
        theory = LENSMIE
        self.assertTrue(theory._can_handle(test_common.sphere))

    def test_theta_quad_pts_min(self):
        assert_allclose(np.min(LENSMIE._theta_pts), 0., **QLIM_TOL)

    def test_theta_quad_pts_max(self):
        assert_allclose(np.max(LENSMIE._theta_pts), LENS_ANGLE, **QLIM_TOL)

    def test_theta_quad_pts_num(self):
        assert_equal(len(LENSMIE._theta_pts), len(LENSMIE._theta_wts))

    def test_phi_quad_pts_min(self):
        assert_allclose(np.min(LENSMIE._phi_pts), 0., **QLIM_TOL)

    def test_phi_quad_pts_max(self):
        assert_allclose(np.max(LENSMIE._phi_pts), 2*np.pi, **QLIM_TOL)

    def test_phi_quad_pts_num(self):
        assert_equal(len(LENSMIE._phi_pts), len(LENSMIE._phi_wts))

    def test_integrate_over_theta_with_quad_points(self):
        pts = LENSMIE._theta_pts
        wts = LENSMIE._theta_wts
        func = np.cos
        integral = np.sum(func(pts) * wts)
        expected_val = np.sin(LENSMIE.lens_angle)  # analytic result
        assert_allclose(integral, expected_val)

    def test_integrate_over_phi_with_quad_points(self):
        pts = LENSMIE._phi_pts
        wts = LENSMIE._phi_wts

        def func(x):
            return np.cos(np.pi * x)

        integral = np.sum(func(pts) * wts)
        expected_val = np.sin(2 * np.pi ** 2) / np.pi  # analytic result
        assert_allclose(integral, expected_val)

    def test_integrate_over_2D_with_quad_points(self):
        pts_theta = LENSMIE._theta_pts
        wts_theta = LENSMIE._theta_wts

        pts_phi = LENSMIE._phi_pts
        wts_phi = LENSMIE._phi_wts

        pts_theta, pts_phi = np.meshgrid(pts_theta, pts_phi)
        wts_theta, wts_phi = np.meshgrid(wts_theta, wts_phi)

        def func(theta, phi):
            return np.cos(theta) * np.cos(np.pi * phi)

        integral = np.sum(func(pts_theta, pts_phi) * wts_theta * wts_phi)
        expected_val = np.sin(1.) * np.sin(2 * np.pi ** 2) / np.pi
        assert_allclose(integral, expected_val)

    def test_quadrature_scattering_matrix_size(self):
        scatterer = test_common.sphere
        medium_wavevec = 2 * np.pi / test_common.wavelen
        medium_index = test_common.index

        theory = LENSMIE

        s_matrix = np.array(theory._calc_scattering_matrix(
                            scatterer, medium_wavevec, medium_index))

        actual_size = s_matrix.size
        actual_shape = s_matrix.shape

        expected_size = 4 * theory.quad_npts_theta * theory.quad_npts_phi
        expected_shape = (4, theory.quad_npts_theta, theory.quad_npts_phi, 1)

        self.assertTrue(actual_size == expected_size)
        self.assertTrue(actual_shape == expected_shape)

    def test_quadrature_scattering_matrix_same_as_mielens(self):
        scatterer = test_common.sphere
        medium_wavevec = 2 * np.pi / test_common.wavelen
        medium_index = test_common.index

        theory = LENSMIE

        theta, phi, s_matrix_new = _get_quad_pts_and_scattering_matrix(
                               theory, scatterer, medium_wavevec, medium_index)

        sinphi = np.sin(phi)
        cosphi = np.cos(phi)
        mielens_calculator = _setup_mielens_calculator(scatterer,
                                                       medium_wavevec,
                                                       medium_index)

        s_perp = mielens_calculator._scat_perp_values.ravel()
        s_prll = mielens_calculator._scat_prll_values.ravel()

        S11, theta_prll = _get_smatrix_theta_near_phi_is_zero(
                                                  s_matrix_new[:, 0, 0],
                                                  cosphi,
                                                  phi,
                                                  theta)
        S22, theta_perp = _get_smatrix_theta_near_phi_is_pi_over_2(
                                                  s_matrix_new[:, 1, 1],
                                                  sinphi,
                                                  phi,
                                                  theta)

        s_perp_new = np.interp(mielens_calculator._theta_pts, theta_perp, S22)
        s_prll_new = np.interp(mielens_calculator._theta_pts, theta_prll, S11)

        assert_allclose(s_perp_new, s_perp, rtol=5e-3)
        assert_allclose(s_prll_new, s_prll, rtol=5e-3)

    def test_raw_fields_similar_mielens(self):
        detector = SMALL_DETECTOR
        scatterer = test_common.sphere
        medium_wavevec = 2 * np.pi / test_common.wavelen
        medium_index = test_common.index
        illum_polarization = detector.illum_polarization

        theory_old = MieLens(lens_angle=LENS_ANGLE)
        pos_old = theory_old._transform_to_desired_coordinates(
            detector, scatterer.center, wavevec=medium_wavevec)

        theory_new = LENSMIE
        pos_new = theory_new._transform_to_desired_coordinates(
            detector, scatterer.center, wavevec=medium_wavevec)

        args = (scatterer, medium_wavevec, medium_index, illum_polarization)
        f0x, f0y, f0z = theory_old._raw_fields(pos_old, *args)
        fx, fy, fz = theory_new._raw_fields(pos_new, *args)
        assert_allclose(f0x, fx, atol=2e-3)
        assert_allclose(f0y, fy, atol=2e-3)
        assert_allclose(f0z, fz, atol=2e-3)

    def test_lens_plus_mie_fields_same_as_mielens(self):
        detector = test_common.xschema_lens
        scatterer = test_common.sphere
        medium_wavevec = 2 * np.pi / test_common.wavelen
        medium_index = test_common.index
        illum_polarization = test_common.xpolarization

        theory_old = MieLens(lens_angle=LENS_ANGLE)
        theory_new = LENSMIE

        fields_old = theory_old.calculate_scattered_field(scatterer, detector)
        fields_new = theory_new.calculate_scattered_field(scatterer, detector)

        assert_allclose(fields_old, fields_new, atol=5e-3)

    def test_calc_holo_theta_npts_not_equal_phi_npts(self):
        scatterer = test_common.sphere
        pts = detector_grid(shape=4, spacing=test_common.pixel_scale)
        pts = update_metadata(pts, illum_wavelen=test_common.wavelen,
                              medium_index=test_common.index,
                              illum_polarization=test_common.xpolarization)
        theory = Lens(LENS_ANGLE, Mie(), quad_npts_theta=8, quad_npts_phi=10)
        holo = calc_holo(pts, scatterer, theory=theory)
        self.assertTrue(True)

    @unittest.skipUnless(NUMEXPR_INSTALLED, "numexpr package required")
    def test_numexpr_integrand_prefactor1(self):
        expr = LENSMIE.numexpr_integrand_prefactor1
        # integrand_prefactor1 = 'exp(1j * krho_p * sinth * cos(phi - phi_p))'

        krho_p, sinth, phi, phi_p = np.random.rand(4)

        result_numpy = np.exp(1j * krho_p * sinth * np.cos(phi - phi_p))
        result_numexpr = ne.evaluate(expr)
        assert_equal(result_numpy, result_numexpr)

    @unittest.skipUnless(NUMEXPR_INSTALLED, "numexpr package required")
    def test_numexpr_integrand_prefactor2(self):
        expr = LENSMIE.numexpr_integrand_prefactor2
        # integrand_prefactor2 = 'exp(1j * kz_p * (1 - costh))'

        kz_p, costh = np.random.rand(2)

        result_numpy = np.exp(1j * kz_p * (1 - costh))
        result_numexpr = ne.evaluate(expr)
        assert_equal(result_numpy, result_numexpr)

    @unittest.skipUnless(NUMEXPR_INSTALLED, "numexpr package required")
    def test_numexpr_integrand_prefactor3(self):
        expr = LENSMIE.numexpr_integrand_prefactor3
        # integrand_prefactor3 = 'sqrt(costh) * sinth * dphi * dth'

        costh, sinth, dphi, dth = np.random.rand(4)

        result_numpy = np.sqrt(costh) * sinth * dphi * dth
        result_numexpr = ne.evaluate(expr)
        assert_equal(result_numpy, result_numexpr)

    @unittest.skipUnless(NUMEXPR_INSTALLED, "numexpr package required")
    def test_numexpr_integrandl(self):
        expr = LENSMIE.numexpr_integrandl
        # integrandl = ('prefactor * (cosphi * (cosphi * S2 + sinphi * S3) +'
        #               + ' sinphi * (cosphi * S4 + sinphi * S1))')

        prefactor, cosphi, sinphi, S1, S2, S3, S4 = np.random.rand(7)

        result_numpy = prefactor * (cosphi * (cosphi * S2 + sinphi * S3)
                                    + sinphi * (cosphi * S4 + sinphi * S1))
        result_numexpr = ne.evaluate(expr)
        assert_equal(result_numpy, result_numexpr)

    @unittest.skipUnless(NUMEXPR_INSTALLED, "numexpr package required")
    def test_numexpr_integrandr(self):
        expr = LENSMIE.numexpr_integrandr
        # integrandr = ('prefactor * (sinphi * (cosphi * S2 + sinphi * S3) -'
        #                + ' cosphi * (cosphi * S4 + sinphi * S1))')

        prefactor, cosphi, sinphi, S1, S2, S3, S4 = np.random.rand(7)

        result_numpy = prefactor * (sinphi * (cosphi * S2 + sinphi * S3)
                                    - cosphi * (cosphi * S4 + sinphi * S1))
        result_numexpr = ne.evaluate(expr)
        assert_equal(result_numpy, result_numexpr)


def _setup_mielens_calculator(scatterer, medium_wavevec, medium_index):
    particle_kz = medium_wavevec * scatterer.z
    index_ratio = scatterer.n / medium_index
    size_parameter = medium_wavevec * scatterer.r

    calculator = MieLensCalculator(particle_kz=particle_kz,
                                   index_ratio=index_ratio,
                                   size_parameter=size_parameter,
                                   lens_angle=LENS_ANGLE)
    return calculator


def _get_quad_pts_and_scattering_matrix(theory, scatterer, medium_wavevec,
                                        medium_index):
    theta, phi = cartesian(theory._theta_pts.ravel(),
                           theory._phi_pts.ravel()).T
    pts = detector_points(theta=theta, phi=phi)
    illum_wavelen = 2 * np.pi * medium_index / medium_wavevec
    pts = update_metadata(pts, medium_index=medium_index,
                          illum_wavelen=illum_wavelen)
    matr = theory.theory.calculate_scattering_matrix(scatterer, pts)
    return theta, phi, np.conj(matr)


def cartesian(*dims):
    return np.array(np.meshgrid(*dims, indexing='ij')).T.reshape(-1, len(dims))


def _get_smatrix_theta_near_phi_is_zero(smatrix, cosphi, phi, theta):
    cp = cosphi[np.logical_and(cosphi == max(abs(cosphi)), phi < np.pi)]
    s = smatrix[np.logical_and(cosphi == max(abs(cosphi)), phi < np.pi)]
    t = theta[np.logical_and(cosphi == max(abs(cosphi)), phi < np.pi)]
    return s / cp, t


def _get_smatrix_theta_near_phi_is_pi_over_2(smatrix, sinphi, phi, theta):
    sp = sinphi[sinphi == max(abs(sinphi))]
    s = smatrix[sinphi == max(abs(sinphi))]
    t = theta[sinphi == max(abs(sinphi))]
    return s / sp, t


if __name__ == '__main__':
    unittest.main()
