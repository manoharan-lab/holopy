import warnings

import numpy as np

try:
    import numexpr as ne
    NUMEXPR_INSTALLED = True
except ModuleNotFoundError:
    NUMEXPR_INSTALLED = False
    from holopy.core.errors import PerformanceWarning
    _LENS_WARNING = ("numexpr not found. Falling back to using numpy only." +
                     " Note that Lens class is faster with numexpr")

from holopy.core import detector_points, update_metadata
from holopy.scattering.theory.scatteringtheory import ScatteringTheory


class Lens(ScatteringTheory):
    """ Wraps a ScatteringTheory and overrides the raw_fields to include the
    effect of an objective lens.
    """
    desired_coordinate_system = 'cylindrical'
    parameter_names = ('lens_angle',)

    numexpr_integrand_prefactor1 = (
        'exp(1j * krho_p * sintheta * cos(phi_relative))')
    numexpr_integrand_prefactor2 = 'exp(1j * kz_p * (1 - costheta))'
    numexpr_integrand_prefactor3 = (
        'sqrt(costheta) * sintheta * phi_wts * theta_wts')
    numexpr_integrandl = ('prefactor * (cosphi * (cosphi * S2 + sinphi * S3) +'
                          + ' sinphi * (cosphi * S4 + sinphi * S1))')
    numexpr_integrandr = ('prefactor * (sinphi * (cosphi * S2 + sinphi * S3) -'
                          + ' cosphi * (cosphi * S4 + sinphi * S1))')

    def __init__(self, lens_angle, theory, quad_npts_theta=100,
                 quad_npts_phi=100, use_numexpr=True):
        if not NUMEXPR_INSTALLED:
            warnings.warn(_LENS_WARNING, PerformanceWarning)
            use_numexpr = False
        super(Lens, self).__init__()
        self.lens_angle = lens_angle
        self.theory = theory
        self.quad_npts_theta = quad_npts_theta
        self.quad_npts_phi = quad_npts_phi

        self.use_numexpr = use_numexpr
        self._setup_quadrature()

    def can_handle(self, scatterer):
        return self.theory.can_handle(scatterer)

    def _setup_quadrature(self):
        """Calculate quadrature points and weights for 2D integration over lens
        pupil
        """
        quad_theta_pts, quad_theta_wts = gauss_legendre_pts_wts(
             0, self.lens_angle, npts=self.quad_npts_theta)
        quad_phi_pts, quad_phi_wts = pts_wts_for_phi_integrals(
            self.quad_npts_phi)

        self._theta_pts = quad_theta_pts.reshape(-1, 1, 1)
        self._theta_wts = quad_theta_wts.reshape(-1, 1, 1)
        self._costheta = np.cos(self._theta_pts)
        self._sintheta = np.sin(self._theta_pts)

        self._phi_pts = quad_phi_pts.reshape(1, -1, 1)
        self._phi_wts = quad_phi_wts.reshape(1, -1, 1)

    def raw_fields(self, positions, scatterer, medium_wavevec, medium_index,
                    illum_polarization):
        pol_angle = np.arctan2(illum_polarization.values[1],
                               illum_polarization.values[0])
        integral_l, integral_r = self._compute_integral(positions, scatterer,
                                                        medium_wavevec,
                                                        medium_index,
                                                        pol_angle)

        fields = self._transform_integral_from_lr_to_xyz(integral_l,
                                                         integral_r,
                                                         pol_angle)

        particle_kz = positions[2, 0]  # we assume a fixed z
        fields *= self._compute_field_phase(particle_kz)
        return fields

    def _compute_integral(self, positions, scatterer, medium_wavevec,
                          medium_index, pol_angle):
        int_l, int_r = self._compute_integrand(positions, scatterer,
                                               medium_wavevec, medium_index,
                                               pol_angle)
        integral_l = np.sum(int_l, axis=(0, 1))
        integral_r = np.sum(int_r, axis=(0, 1))
        return integral_l, integral_r

    def _compute_integrand(self, positions, scatterer, medium_wavevec,
                           medium_index, pol_angle):
        krho_p, phi_p, kz_p = positions
        pos_shape = (1, 1, len(kz_p))
        krho_p = krho_p.reshape(pos_shape)
        phi_p = phi_p.reshape(pos_shape)
        kz_p = kz_p.reshape(pos_shape)

        prefactor = self._integrand_prefactor(krho_p, phi_p, kz_p)

        scat_matrix = self._calc_scattering_matrix(scatterer,
                                                   medium_wavevec,
                                                   medium_index)
        integrand_l = self._integrand_prll(prefactor, pol_angle, *scat_matrix)
        integrand_r = self._integrand_perp(prefactor, pol_angle, *scat_matrix)
        return integrand_l, integrand_r

    def _integrand_prefactor(self, krho_p, phi_p, kz_p):
        # define variables for numexpr:
        sintheta = self._sintheta
        costheta = self._costheta
        phi_relative = self._phi_pts - phi_p
        phi_wts = self._phi_wts
        theta_wts = self._theta_wts
        if self.use_numexpr:
            prefactor = ne.evaluate(self.numexpr_integrand_prefactor1)
            prefactor *= ne.evaluate(self.numexpr_integrand_prefactor2)
            prefactor *= ne.evaluate(self.numexpr_integrand_prefactor3)
        else:
            prefactor = np.exp(
                1j * krho_p * sintheta * np.cos(phi_relative))
            prefactor *= np.exp(1j * kz_p * (1 - costheta))
            prefactor *= (np.sqrt(costheta) * sintheta *
                          phi_wts * theta_wts)
        prefactor *= .5 / np.pi
        return prefactor

    def _calc_scattering_matrix(self, scatterer, medium_wavevec, medium_index):
        theta, phi = np.meshgrid(self._theta_pts, self._phi_pts)
        illum_wavelen = 2 * np.pi * medium_index / medium_wavevec
        pos = np.array([0 * theta, theta, phi]).reshape(3, -1)
        S = self.theory.raw_scat_matrs(
            scatterer, pos, medium_wavevec, medium_index)
        S = np.conj(S).reshape(self.quad_npts_theta, self.quad_npts_phi, 2, 2)
        S = np.swapaxes(S, 0, 1)
        S1 = S[:, :, 1, 1].reshape(self.quad_npts_theta, self.quad_npts_phi, 1)
        S2 = S[:, :, 0, 0].reshape(self.quad_npts_theta, self.quad_npts_phi, 1)
        S3 = S[:, :, 0, 1].reshape(self.quad_npts_theta, self.quad_npts_phi, 1)
        S4 = S[:, :, 1, 0].reshape(self.quad_npts_theta, self.quad_npts_phi, 1)
        return S1, S2, S3, S4

    def _integrand_prll(self, prefactor, pol_angle, S1, S2, S3, S4):
        cosphi = np.cos(self._phi_pts - pol_angle)
        sinphi = np.sin(self._phi_pts - pol_angle)
        if self.use_numexpr:
            integrand_l = ne.evaluate(self.numexpr_integrandl)
        else:
            integrand_l = prefactor * (cosphi * (cosphi * S2 + sinphi * S3)
                                       + sinphi * (cosphi * S4 + sinphi * S1))
        return integrand_l

    def _integrand_perp(self, prefactor, pol_angle, S1, S2, S3, S4):
        cosphi = np.cos(self._phi_pts - pol_angle)
        sinphi = np.sin(self._phi_pts - pol_angle)
        if self.use_numexpr:
            integrand_r = ne.evaluate(self.numexpr_integrandr)
        else:
            integrand_r = prefactor * (sinphi * (cosphi * S2 + sinphi * S3)
                                       - cosphi * (cosphi * S4 + sinphi * S1))
        return integrand_r

    def _transform_integral_from_lr_to_xyz(self, prll_component,
                                           perp_component,
                                           pol_angle):
        parallel = np.array([      np.cos(pol_angle), np.sin(pol_angle)])
        perpendicular = np.array([-np.sin(pol_angle), np.cos(pol_angle)])
        xyz = np.zeros([3, prll_component.size], dtype='complex')
        for i in range(2):
            xyz[i, :] += prll_component * parallel[i]
            xyz[i, :] += perp_component * perpendicular[i]
        return xyz

    def _compute_field_phase(self, particle_kz):
        # This includes 2 effects:
        # 1. The phase shift of the beam at the particle's position
        #    relative to that at the focal plane (e^ikz), and
        # 2. The factor of -1 from the Gouy phase shift of the
        #    *incident* beam, which we include here since holopy assumes
        #    that the incident beam is unshifted.
        return -1. * np.exp(1j * particle_kz)


def gauss_legendre_pts_wts(a, b, npts=100):
    """Quadrature points for integration on interval [a, b]"""
    pts_raw, wts_raw = np.polynomial.legendre.leggauss(npts)
    pts = pts_raw * (b - a) * 0.5
    wts = wts_raw * (b - a) * 0.5
    pts += 0.5 * (a + b)
    return pts, wts


def pts_wts_for_phi_integrals(npts):
    """Quadrature points for integration on the periodic interval [0, pi]

    Since this interval is periodic, we use equally-spaced points with
    equal weights.
    """
    pts = np.linspace(0, 2 * np.pi, npts + 1)[:-1].copy()
    wts = np.full_like(pts, 2 * np.pi / pts.size)
    return pts, wts
