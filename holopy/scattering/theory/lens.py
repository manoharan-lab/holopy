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
        self.theory = theory
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

        self._theta_pts = quad_theta_pts
        self._costheta_pts = np.cos(self._theta_pts)
        self._sintheta_pts = np.sin(self._theta_pts)

        self._phi_pts = quad_phi_pts
        self._cosphi_pts = np.cos(self._phi_pts)
        self._sinphi_pts = np.sin(self._phi_pts)

        self._theta_wts = quad_theta_wts
        self._phi_wts = quad_phi_wts

    def _raw_fields(self, positions, scatterer, medium_wavevec, medium_index,
                    illum_polarization):
        integral_x, integral_y = self._compute_integral(positions, scatterer,
                                                        medium_wavevec, medium_index,
                                                        illum_polarization)

        fields = self._transform_integral_from_lr_to_xyz(integral_x, integral_y,
                                                         illum_polarization)

        prefactor = self._compute_field_prefactor(scatterer, medium_wavevec)
        fields = prefactor * fields
        return fields

    def _compute_integral(self, positions, scatterer, medium_wavevec,
                          medium_index, illum_polarization):
        int_x, int_y = self._compute_integrand(positions, scatterer,
                                               medium_wavevec, medium_index,
                                               illum_polarization)
        integral_x = np.sum(int_x, axis=(0,1))
        integral_y = np.sum(int_y, axis=(0,1))
        return integral_x, integral_y

    def _compute_integrand(self, positions, scatterer, medium_wavevec,
                           medium_index, illum_polarization):
        krho_p, phi_p, kz_p = positions
        pol_angle = np.arctan2(illum_polarization[1], illum_polarization[0])
        phi_p += pol_angle.values
        phi_p %= (2 * np.pi)

        theta_shape = (self.quad_npts_theta, 1, 1)
        th = self._theta_pts.reshape(theta_shape)
        sinth = self._sintheta_pts.reshape(theta_shape)
        costh = self._costheta_pts.reshape(theta_shape)
        dth = self._theta_wts.reshape(theta_shape)

        phi_shape = (1, self.quad_npts_phi, 1)
        sinphi = self._sinphi_pts.reshape(phi_shape)
        cosphi = self._cosphi_pts.reshape(phi_shape)
        phi = self._phi_pts.reshape(phi_shape)
        dphi = self._phi_wts.reshape(phi_shape)

        pos_shape = (1, 1, len(kz_p))
        krho_p = krho_p.reshape(pos_shape)
        phi_p = phi_p.reshape(pos_shape)
        kz_p = kz_p.reshape(pos_shape)

        prefactor = np.exp(1j * krho_p * sinth * np.cos(phi - phi_p))
        prefactor *= np.exp(1j * kz_p * (1 - costh))
        prefactor *= np.sqrt(costh) * sinth * dphi * dth
        prefactor *= .5 / np.pi

        S1, S2, S3, S4 = self._calc_scattering_matrix(scatterer, medium_wavevec,
                                                      medium_index)

        integrand_x = prefactor * (cosphi * (cosphi * S2 + sinphi * S3)
                                   + sinphi * (cosphi * S4 + sinphi * S1))
        integrand_y = prefactor * (sinphi * (cosphi * S2 + sinphi * S3)
                                   - cosphi * (cosphi * S4 + sinphi * S1))

        return integrand_x, integrand_y

    def _calc_scattering_matrix(self, scatterer, medium_wavevec, medium_index):
        theta, phi = np.meshgrid(self._theta_pts, self._phi_pts)
        pts = detector_points(theta=theta.ravel(), phi=phi.ravel())
        illum_wavelen = 2 * np.pi * medium_index / medium_wavevec

        pts = update_metadata(pts, medium_index=medium_index,
                              illum_wavelen=illum_wavelen)
        S = self.theory.calculate_scattering_matrix(scatterer, pts)
        S = np.conj(S.values.reshape(self.quad_npts_theta,
                                     self.quad_npts_phi, 2, 2))
        S = np.swapaxes(S, 0, 1)
        S1 = S[:, :, 1, 1].reshape(self.quad_npts_theta, self.quad_npts_phi, 1)
        S2 = S[:, :, 0, 0].reshape(self.quad_npts_theta, self.quad_npts_phi, 1)
        S3 = S[:, :, 0, 1].reshape(self.quad_npts_theta, self.quad_npts_phi, 1)
        S4 = S[:, :, 1, 0].reshape(self.quad_npts_theta, self.quad_npts_phi, 1)
        return S1, S2, S3, S4

    def _transform_integral_from_lr_to_xyz(self, prll_component, perp_component,
                                           illum_polarization):
        pol_angle = np.arctan2(illum_polarization.values[1],
                               illum_polarization.values[0])
        parallel = np.array([np.cos(pol_angle), np.sin(pol_angle)])
        perpendicular = np.array([-np.sin(pol_angle), np.cos(pol_angle)])
        xyz = np.zeros([3, prll_component.size], dtype='complex')
        for i in range(2):
            xyz[i, :] += prll_component * parallel[i]
            xyz[i, :] += perp_component * perpendicular[i]
        return xyz

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
