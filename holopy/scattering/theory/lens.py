import numpy as np

from holopy.core import detector_points, update_metadata
from holopy.scattering.theory.scatteringtheory import ScatteringTheory


class LensScatteringTheory(ScatteringTheory):
    """ Wraps a ScatteringTheory and overrides the _raw_fields to include the
    effect of an objective lens.
    """
    def __init__(self, lens_angle, theory, quad_npts=100):
        super(LensScatteringTheory, self).__init__()
        self.lens_angle = lens_angle
        self.theory = theory()
        self.quad_npts = quad_npts
        self._setup_quadrature()

    def _can_handle(self, scatterer):
        return self.theory._can_handle(scatterer)

    def _setup_quadrature(self):
        quad_theta_pts, quad_theta_wts = gauss_legendre_pts_wts(
             np.cos(self.lens_angle), 1.0, npts=self.quad_npts)
        quad_phi_pts, quad_phi_wts = gauss_legendre_pts_wts(
             0, 2 * np.pi, npts=self.quad_npts)

        self._costheta_pts = quad_theta_pts.reshape(-1, 1)
        self._theta_pts = np.arccos(quad_theta_pts)
        self._sintheta_pts = np.sin(self._theta_pts).reshape(-1, 1)
        self._theta_wts = quad_theta_wts.reshape(-1, 1)

        self._phi_pts = quad_phi_pts.ravel()
        self._cosphi_pts = np.cos(self._phi_pts)
        self._sinphi_pts = np.sin(self._phi_pts)
        self._phi_wts = quad_phi_wts.reshape(-1, 1)

    def _raw_fields(self, positions, scatterer, medium_wavevec, medium_index,
                    illum_polarization):
        scat_matrs = self._compute_scattering_matrices_quad_pts(scatterer,
                                                   medium_wavevec, medium_index)
        integral = self._compute_integral(scat_matrs, scatterer)
        prefactor = self._compute_field_prefactor()
        fields = prefactor * integral
        return fields

    # TODO: Implement the B&B method
    def _compute_integral(self, scat_matrs, scatterer):
        return 0.

    def _compute_field_prefactor(self):
        return -1.

    def _compute_scattering_matrices_quad_pts(self, scatterer, medium_wavevec,
                                              medium_index):

        pts = detector_points(theta=self._theta_pts, phi=self._phi_pts)
        illum_wavelen = 2 * np.pi / medium_wavevec
        pts = update_metadata(pts, medium_index=medium_index, illum_wavelen=illum_wavelen)
        matr = self.theory.calculate_scattering_matrix(scatterer, pts)
        return matr

    def _raw_scat_matrs(self, *args, **kwargs):
        return self.theory._raw_scat_matrs(*args, **kwargs)


def gauss_legendre_pts_wts(a, b, npts=100):
    """Quadrature points for integration on interval [a, b]"""
    pts_raw, wts_raw = np.polynomial.legendre.leggauss(npts)
    pts = pts_raw * (b - a) * 0.5
    wts = wts_raw * (b - a) * 0.5
    pts += 0.5 * (a + b)
    return pts, wts
