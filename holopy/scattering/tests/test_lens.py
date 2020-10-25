import unittest

import numpy as np
import xarray as xr
from numpy.testing import assert_allclose, assert_equal
from nose.plugins.attrib import attr
from scipy.special import iv

try:
    import numexpr as ne
    NUMEXPR_INSTALLED = True
except ModuleNotFoundError:
    NUMEXPR_INSTALLED = False

from holopy.core import detector_points, update_metadata, detector_grid
from holopy.scattering import calc_holo, Sphere, Spheres
from holopy.scattering.theory import Mie, MieLens, Multisphere
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
            return np.exp(-3 * np.sin(x))

        integral = np.sum(func(pts) * wts)
        expected_val = 2 * np.pi * iv(0, 3)  # analytic result
        assert_allclose(integral, expected_val)

    def test_integrate_over_2D_with_quad_points(self):
        pts_theta = LENSMIE._theta_pts
        wts_theta = LENSMIE._theta_wts

        pts_phi = LENSMIE._phi_pts
        wts_phi = LENSMIE._phi_wts

        pts_theta, pts_phi = np.meshgrid(pts_theta, pts_phi)
        wts_theta, wts_phi = np.meshgrid(wts_theta, wts_phi)

        def func(theta, phi):
            return np.cos(theta) * np.exp(-3 * np.sin(phi))

        integral = np.sum(func(pts_theta, pts_phi) * wts_theta * wts_phi)
        expected_val = np.sin(LENS_ANGLE) * 2 * np.pi * iv(0, 3)
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

    def test_transforms_correctly_with_polarization_rotation(self):
        # We test that rotating the lab frame correctly rotates
        # the polarization.
        # If we rotate (x0, y0) -> (y1, -x1), then the polarization
        # in the new coordinates should be
        # E1x = E0y, E1y = -E1x
        scatterer = test_common.sphere
        medium_wavevec = 2 * np.pi / test_common.wavelen
        medium_index = test_common.index
        theory = Lens(
            lens_angle=LENS_ANGLE,
            theory=Mie(False, False),
            quad_npts_theta=200,
            quad_npts_phi=200,
            )

        krho = np.linspace(0, 100, 11)
        phi_0 = 0 * krho
        phi_1 = np.full_like(krho, -np.pi / 2)
        kz = np.full_like(krho, 20.0)

        pol_0 = xr.DataArray([1.0, 0, 0])
        pos_0 = np.array([krho, phi_0, kz])

        pol_1 = xr.DataArray([0, -1.0, 0])
        pos_1 = np.array([krho, phi_1, kz])

        args = (scatterer, medium_wavevec, medium_index)

        fields_0 = theory._raw_fields(pos_0, *args, pol_0)
        fields_1 = theory._raw_fields(pos_1, *args, pol_1)

        tols = {'atol': 1e-5, 'rtol': 1e-5}
        assert_allclose(fields_1[0],  fields_0[1], **tols)
        assert_allclose(fields_1[1], -fields_0[0], **tols)

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

    @attr('medium')
    def test_polarization_rotation_produces_small_changes_to_image(self):
        # we test that, for two sphere, rotating the polarization is
        # does not drastically change the image
        # We place the two spheres along the line phi = 0

        z_um = 3.0
        s1 = Sphere(r=0.5, center=(1.0, 0.0, z_um), n=1.59)
        s2 = Sphere(r=0.5, center=(2.5, 0.0, z_um), n=1.59)
        scatterer = Spheres([s1, s2])

        medium_wavevec = 2 * np.pi * 1.33 / 0.66
        medium_index = test_common.index
        theory = Lens(LENS_ANGLE, Multisphere())
        args = (scatterer, medium_wavevec, medium_index)

        rho_um = np.linspace(0, 5, 26)
        krho = medium_wavevec * rho_um
        phi = np.zeros(krho.size)
        kz = np.zeros(krho.size) + medium_wavevec * z_um
        pos = np.array([krho, phi, kz])

        fields_xpol = theory._raw_fields(pos, *args, xr.DataArray([1, 0, 0]))
        fields_ypol = theory._raw_fields(pos, *args, xr.DataArray([0, 1, 0]))

        intensity_xpol = np.linalg.norm(fields_xpol, axis=0)**2
        intensity_ypol = np.linalg.norm(fields_ypol, axis=0)**2

        tols = {'atol': 1e-3, 'rtol': 1e-3}
        self.assertTrue(np.allclose(intensity_xpol, intensity_ypol, **tols))


class TestLensVsMielens(unittest.TestCase):
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

    def test_raw_fields_similar_mielens_xpolarization(self):
        detector = SMALL_DETECTOR
        scatterer = test_common.sphere
        medium_wavevec = 2 * np.pi / test_common.wavelen
        medium_index = test_common.index
        illum_polarization = xr.DataArray([1.0, 0, 0])

        theory_old = MieLens(lens_angle=LENS_ANGLE)
        pos_old = theory_old._transform_to_desired_coordinates(
            detector, scatterer.center, wavevec=medium_wavevec)

        theory_new = LENSMIE
        pos_new = theory_new._transform_to_desired_coordinates(
            detector, scatterer.center, wavevec=medium_wavevec)

        args = (scatterer, medium_wavevec, medium_index, illum_polarization)
        f0x, f0y, f0z = theory_old._raw_fields(pos_old, *args)
        f1x, f1y, f1z = theory_new._raw_fields(pos_new, *args)
        assert_allclose(f0x, f1x, atol=2e-3)
        assert_allclose(f0y, f1y, atol=2e-3)
        assert_allclose(f0z, f1z, atol=2e-3)

    def test_raw_fields_similar_mielens_ypolarization(self):
        detector = SMALL_DETECTOR
        scatterer = test_common.sphere
        medium_wavevec = 2 * np.pi / test_common.wavelen
        medium_index = test_common.index
        illum_polarization = xr.DataArray([0, 1.0, 0])

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
