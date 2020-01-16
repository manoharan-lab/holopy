import numpy as np

from holopy.core import detector_points, update_metadata
from holopy.scattering.theory.scatteringtheory import ScatteringTheory


class LensScatteringTheory(ScatteringTheory):
    """ Wraps a ScatteringTheory and overrides the _raw_fields to include the
    effect of an objective lens.
    """
    desired_coordinate_system = 'cylindrical'

    def __init__(self, lens_angle, theory, quad_npts_theta=100,
                 quad_npts_phi=100):
        super(LensScatteringTheory, self).__init__()
        self.lens_angle = lens_angle
        self.theory = theory()
        self.quad_npts_theta = quad_npts_theta
        self.quad_npts_phi = quad_npts_phi
        self._setup_quadrature()

    def _can_handle(self, scatterer):
        return self.theory._can_handle(scatterer)

    def _setup_quadrature(self):
        """Calculate quadrature points and weights for 2D integration over lens
        pupil
        """
        quad_theta_pts, quad_theta_wts = gauss_legendre_pts_wts(
             0, self.lens_angle, npts=self.quad_npts_theta)
        quad_phi_pts, quad_phi_wts = gauss_legendre_pts_wts(
             0, 2 * np.pi, npts=self.quad_npts_phi)

        quad_theta_pts, quad_phi_pts = cartesian(quad_theta_pts, quad_phi_pts).T
        quad_theta_wts, quad_phi_wts = cartesian(quad_theta_wts, quad_phi_wts).T

        self._theta_pts = quad_theta_pts
        self._costheta_pts = np.cos(self._theta_pts)
        self._sintheta_pts = np.sin(self._theta_pts)

        self._phi_pts = quad_phi_pts
        self._cosphi_pts = np.cos(self._phi_pts)
        self._sinphi_pts = np.sin(self._phi_pts)

        self._scale_quadrature_wieghts(quad_theta_wts, quad_phi_wts)

    def _scale_quadrature_wieghts(self, theta_wieghts, phi_weights):
        """Scales the Gaussain quadrature wieghts so we can integrate over 2d
        space properly.
        """
        self._theta_wts = theta_wieghts / self.quad_npts_phi
        self._phi_wts = phi_weights / self.quad_npts_theta

    def _raw_fields(self, positions, scatterer, medium_wavevec, medium_index,
                    illum_polarization):
        scat_matrs = self._calc_scattering_matrices_at_quad_pts(scatterer,
                                                   medium_wavevec, medium_index)
        integral_x, integral_y = self._compute_integral(positions, scatterer,
                                                        scat_matrs,
                                                        illum_polarization)
        #fields = np.vstack([integral_x, integral_y, np.zeros_like(integral_x)])
        pol_angle = np.arctan2(
            illum_polarization.values[1], illum_polarization.values[0])
        parallel = np.array([np.cos(pol_angle), np.sin(pol_angle)])
        perpendicular = np.array([-np.sin(pol_angle), np.cos(pol_angle)])
        fields = np.zeros([3, integral_x.size], dtype='complex')
        for i in range(2):
            fields[i, :] += integral_x * parallel[i]
            fields[i, :] += integral_y * perpendicular[i]

        prefactor = self._compute_field_prefactor(scatterer, medium_wavevec)
        fields = prefactor * fields
        return fields

    def _calc_scattering_matrices_at_quad_pts(self, scatterer, medium_wavevec,
                                              medium_index):
        theta, phi = self._theta_pts, self._phi_pts
        pts = detector_points(theta=theta, phi=phi)
        illum_wavelen = 2 * np.pi * medium_index / medium_wavevec
        pts = update_metadata(pts, medium_index=medium_index, illum_wavelen=illum_wavelen)
        matr = self.theory.calculate_scattering_matrix(scatterer, pts)
        return np.conj(matr)

    def _compute_integral(self, positions, scatterer, scat_matrs, illum_polarization):
        int_x, int_y = self._compute_integrand(positions, scat_matrs, illum_polarization)
        integral_x = np.sum(int_x, axis=1)
        integral_y = np.sum(int_y, axis=1)
        return integral_x, integral_y

    def _compute_integrand(self, positions, scat_matrs, illum_polarization):
        krho_p, phi_p, kz_p = positions
        pol_angle = np.arctan2(illum_polarization[1], illum_polarization[0])
        phi_p += pol_angle.values
        phi_p %= (2 * np.pi)

        sinth = self._sintheta_pts
        costh = self._costheta_pts
        dth = self._theta_wts

        sinphi = self._sinphi_pts
        cosphi = self._cosphi_pts
        phi = self._phi_pts
        dphi = self._phi_wts

        prefactor = .5 * np.exp(1j * kz_p[:, None] * (1 - costh))
        prefactor *= np.exp(1j * krho_p[:, None] * sinth * np.cos(phi - phi_p[:, None]))
        prefactor *= np.sqrt(costh) * sinth * dphi * dth
        prefactor *= 1. / np.pi

        S11 = scat_matrs.values[:, 0, 0]
        S22 = scat_matrs.values[:, 1, 1]

        integrand_x = prefactor * (S11 * cosphi + S22 * sinphi)
        integrand_y = prefactor * (S11 * sinphi - S22 * cosphi)

        return integrand_x, integrand_y

    def _compute_field_prefactor(self, scatterer, medium_wavevec):
        return -1. * np.exp(1j * medium_wavevec * scatterer.center[2])

    def _raw_scat_matrs(self, *args, **kwargs):
        return self.theory._raw_scat_matrs(*args, **kwargs)


def gauss_legendre_pts_wts(a, b, npts=100):
    """Quadrature points for integration on interval [a, b]"""
    pts_raw, wts_raw = np.polynomial.legendre.leggauss(npts)
    pts = pts_raw * (b - a) * 0.5
    wts = wts_raw * (b - a) * 0.5
    pts += 0.5 * (a + b)
    return pts, wts

def cartesian(*dims):
    return np.array(np.meshgrid(*dims, indexing='ij')).T.reshape(-1, len(dims))
