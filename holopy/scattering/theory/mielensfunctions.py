import numpy as np
from scipy.special import j0, j1, spherical_jn, spherical_yn
from scipy import interpolate


NPTS = 100
LEGGAUSS_PTS_WTS_NPTS = np.polynomial.legendre.leggauss(NPTS)


# TODO:
# fast integration of oscillatory functions.


class MieLensCalculator(object):
    def __init__(self, particle_kz=10.0, index_ratio=1.1, size_parameter=10.0,
                 lens_angle=1.0, quad_npts=100, interpolate_integrals='check',
                 interpolation_spacing=0.1):
        """Calculates the field from a Mie scatterer imaged in a high-NA lens.

        The incindent electric field is E e^{ikz}, with the particle
        position at z. The scattered field takes into account the
        varying phase of the incoming field.

        Parameters
        ----------
        particle_kz : float
            + z is away from the lens
        index_ratio : float > 0
        size_parameter : float > 0
        lens_angle : float on (0, pi/2)
        quad_npts : int, optional
            The number of points for numerical quadrature of the
            integrals over the lens pupil.
        interpolate_integrals : {'check', True, False}
            Whether or not to interpolate the internally-evaluated
            integrals for speed. Default is `'check'`, which interpolates
            if it will be faster or does direct numerical quadrature
            otherwise.
        interpolation_spacing : float, optional
            The spacing, in units of `1/k`, for the nodes of the
            CubicSpline interpolators. A lower value gives more accurate
            retuls; values greater than about 1 will be unreliable.
            Default is 0.1, which gives better than single-precision
            relative accuracy.

        Methods
        -------
        total_field(krho, phi)
            numpy.ndarray, of shape krho
        total_intensity(krho, phi)
            numpy.ndarray, of shape krho
        """
        self.particle_kz = particle_kz
        self.index_ratio = index_ratio
        self.size_parameter = size_parameter
        self.lens_angle = lens_angle

        self.quad_npts = quad_npts
        self.interpolate_integrals = interpolate_integrals
        self.interpolation_spacing = interpolation_spacing

        quad_pts, quad_wts = gauss_legendre_pts_wts(
            np.cos(self.lens_angle), 1.0, npts=self.quad_npts)

        # Precompute some quadrature points, mie functions that are
        # independent of rho and phi
        self._quad_pts = quad_pts.reshape(-1, 1)
        self._theta_pts = np.arccos(quad_pts)
        self._sintheta_pts = np.sin(self._theta_pts).reshape(-1, 1)
        self._quad_wts = quad_wts.reshape(-1, 1)

        self._precompute_fi()

    def calculate_scattered_field(self, krho, phi):
        """Calculates the field from a Mie scatterer imaged through a
        high-NA lens and excited with an electric field of unit strength
        directed along the optical axis.

            .. math::
                \vec{E}_{sc} = A \left[ I_{12} \sin(2\phi) \hat{y} +
                                       -I_{10} \hat{x} +
                                        I_{12} \cos(2\phi) \hat{x} +
                                       -I_{20} \hat{x} +
                                       -I_{22} \cos(2\phi) \hat{x} +
                                       -I_{22} \sin(2\phi) \hat{y} \right]

        Parameters
        ----------
        krho, phi : numpy.ndarray
            The position of the particle relative to the focal point of the
            lens, in (i) cylindrical coordinates and (ii) dimensionless
            wavevectur units. Must all be the same shape.

        Returns
        -------
        field_xcomp, field_ycomp : numpy.ndarray
            The (x, y) components of the electric field at the detector, where
            the initial field is polarized in the x-direction. Same shape as
            krho, phi

        Notes
        -----
        This will have problems for large rho, z, because of the quadrature
        points. Empirically this problem happens for rho >~ 4 * quad_npts.
        Could be adaptive if needed....
        """
        # 0. Check inputs:
        shape = krho.shape
        if (shape != phi.shape):
            raise ValueError('krho, phi must all be the same shape')
        # 1. Check for regions where rho is bad and set to 0:
        rho_too_big = krho > 3.9 * self.quad_npts
        rho_ok = ~rho_too_big

        # 2. Evaluate scattered fields only at valid rho's:
        ex_valid, ey_valid = self._calculate_scattered_fields(
            krho[rho_ok], phi[rho_ok])

        # 3. Return
        output_x = np.zeros(shape, dtype='complex')
        output_y = np.zeros(shape, dtype='complex')
        output_x[rho_ok] = ex_valid
        output_y[rho_ok] = ey_valid
        return output_x, output_y

    def calculate_total_field(self, krho, phi):
        """The total (incident + scattered) field at the detector
        """
        # Uses the incident field as
        #   E_in = E_0 \hat{x} * 4 pi * (f1 / f2) * e^{ik(f1 + f2)} * i
        # which is more-or-less from the brightfield writeups.
        # return 1j - 0.25 * mielens_field(krho, phi, **kwargs)
        fx, fy = self.calculate_scattered_field(krho, phi)
        return 1j + fx, fy

    def calculate_total_intensity(self, krho, phi):
        fx, fy = self.calculate_total_field(krho, phi)
        return np.abs(fx)**2 + np.abs(fy)**2

    def _calculate_scattered_fields(self, krho, phi):
        shape = phi.shape
        # 2. Evaluate the integrals:
        i_10 = np.reshape(
            self._eval_mielens_i_ij(krho, self._f1_values, j=0), shape)
        i_12 = np.reshape(
            self._eval_mielens_i_ij(krho, self._f1_values, j=2), shape)
        i_20 = np.reshape(
            self._eval_mielens_i_ij(krho, self._f2_values, j=0), shape)
        i_22 = np.reshape(
            self._eval_mielens_i_ij(krho, self._f2_values, j=2), shape)
        # 3. Sum for the field:
        c2p = np.cos(2 * phi)
        s2p = np.sin(2 * phi)
        field_xcomp = 0.25 * (i_10 + i_20 - (i_12 - i_22) * c2p)
        field_ycomp = 0.25 * (i_12 - i_22) * s2p
        return field_xcomp, field_ycomp

    def _precompute_fi(self):
        kwargs = {'index_ratio': self.index_ratio,
                  'size_parameter': self.size_parameter,
                  }
        f1_evaluator = FarfieldMieEvaluator(i=1, lazy=True, **kwargs)
        f2_evaluator = FarfieldMieEvaluator(i=2, lazy=True, **kwargs)
        self._f1_values = np.reshape(
            f1_evaluator._eval(self._theta_pts), (-1, 1))
        self._f2_values = np.reshape(
            f2_evaluator._eval(self._theta_pts), (-1, 1))

    def _eval_mielens_i_ij(self, krho, fi_values, j=0):
        """Calculates one of several similar integrals over the lens
        pupil which appear in the Mie + lens calculations

        This should only be called by `self.calculate_scattered_field`

        Parameters
        ----------
        krho : numpy.ndarray
            The rho values to evaluate the integrals at, in units of 1/k.
        fi_values : numpy.ndarray
            The values of the far-field Mie solutions, as precomputed
            by `self._precompute_fi`
        j : {0, 2}, optional
            The integers which determine which Bessel function
            the Mie fields to evaluate. Default is 0; should always be
            passed though.

        Returns
        -------
        numpy.ndarray
            The value of the integrand evaluated at the krho points.
        """
        if self.interpolate_integrals == 'check':
            n_interp_pnts = krho.ptp() / self.interpolation_spacing
            n_krho_pts = krho.size
            interpolate_integrals = n_interp_pnts < 1.1 * n_krho_pts
        else:
            interpolate_integrals = self.interpolate_integrals is True

        if interpolate_integrals:
            i_ij = self._interpolate_and_eval_mielens_i_ij(krho, fi_values, j)
        else:
            i_ij = self._direct_eval_mielens_i_ij(krho, fi_values, j)

        return i_ij

    def _direct_eval_mielens_i_ij(self, krho, fi_values, j=0):
        if j == 0:
            ji = j0
        elif j == 2:
            ji = j2
        else:
            raise ValueError('j must be one of {0, 2}')
        # We do the integral with the change of variables x = cos(theta),
        # from cos(lens_angle) to 1.0:
        # Placing things in order [quadrature points, rho-z values]
        rr = krho.reshape(1, -1)
        integrand = (np.exp(-1j * self.particle_kz * (self._quad_pts - 1)) *
                     fi_values * ji(rr * self._sintheta_pts))
        answer_flat = np.sum(integrand * self._quad_wts, axis=0)
        return answer_flat.reshape(krho.shape)

    def _interpolate_and_eval_mielens_i_ij(self, krho, fi_values, j=0):
        spacing = self.interpolation_spacing
        interp_pts = np.arange(krho.min(), krho.max() + 5 * spacing, spacing)
        interp_vals = self._direct_eval_mielens_i_ij(
            interp_pts, fi_values, j=0)
        interpolator = interpolate.CubicSpline(interp_pts, interp_vals)
        return interpolator(krho)


class FarfieldMieEvaluator(object):
    def __init__(self, i=1, index_ratio=1.1, size_parameter=1.0, max_l=None,
                 npts=None, lazy=False):
        """Interpolators for some derived Mie scattering functions, as
        defined in the module docstring.

        These could be better for large sizes by using asymptotic
        representations of the scattering field.

        Parameters
        ----------
        i : {1, 2}
            Which interpolator to use. i=1 is sin(phi), i=2 is cos(phi)
        index_ratio : float
            Index contrast of the particle.
        size_parameter : float
            Size of the sphere in units of 1/k = 1/wavevector
        max_l : int > 0
        npts : int > 0
        lazy : bool
            Whether or not to set up the interpolator right away or
            to wait until it is called.
        """
        self.i = i
        self.index_ratio = index_ratio
        self.size_parameter = size_parameter
        self.max_l = self._default_max_l() if max_l is None else max_l
        self.npts = self._default_npts() if npts is None else npts
        self.lazy = lazy
        if not lazy:
            self._setup_interpolator()
        else:
            self._interp = None

    def _setup_interpolator(self):
        self._true_pts = np.linspace(0, 0.5 * np.pi, self.npts)
        self._true_values = self._eval(self._true_pts)
        self._interp = interpolate.CubicSpline(
            self._true_pts, self._true_values)

    def _default_max_l(self):
        """An empirically good value for ~1e-7 accuracy"""
        return np.ceil(4 * self.size_parameter).astype('int')

    def _default_npts(self):
        # Since tau_l(theta), pi_l(theta) ~ d/dx P_l^1, there are O(l)
        # maxima / minima / zeros in the highest term, so we expect
        # structure on the scale of ~1/l. So we take 10 * l points:
        # This empirically works as well
        return 10 * self.max_l

    def _eval(self, theta):
        """Evaluate F_i(theta) the hard way"""
        ans = np.zeros(theta.size, dtype='float')  # real, not complex
        # Right now, the pi_l, tau_l functions calculate all values of
        # l at once. So we compute all at once then sum
        pils, tauls = calculate_pil_taul(theta, self.max_l)
        coeffs = np.array([(2 * l + 1) / (l * (l + 1))
                           for l in range(1, self.max_l + 1)]).reshape(1, -1)
        als_bls = [calculate_al_bl(self.index_ratio, self.size_parameter, l)
                   for l in range(1, self.max_l + 1)]
        als, bls = [np.array(i) for i in zip(*als_bls)]
        if self.i == 1:
            ans = np.sum(coeffs * (bls * tauls + als * pils), axis=1)
        elif self.i == 2:
            ans = np.sum(coeffs * (als * tauls + bls * pils), axis=1)
        if np.isnan(ans).any():
            raise RuntimeError('nan for this value of theta, ka, max_l')
        return ans

    def __call__(self, theta):
        # call the interpolator
        if self._interp is None:
            self._setup_interpolator()
        return self._interp(theta)


def j2(x):
    """A fast J_2(x) defined in terms of other special functions """
    clipped = np.clip(x, 1e-15, np.inf)
    return 2. / clipped * j1(clipped) - j0(clipped)


def spherical_h1n(n, z, derivative=False):
    """Spherical Hankel function H_n(z) or its derivative"""
    return spherical_jn(n, z, derivative) + 1j * spherical_yn(n, z, derivative)


def gauss_legendre_pts_wts(a, b, npts=NPTS):
    """Quadrature points for integration on interval [a, b]"""
    if npts == NPTS:
        pts_raw, wts_raw = LEGGAUSS_PTS_WTS_NPTS
    else:
        pts_raw, wts_raw = np.polynomial.legendre.leggauss(npts)
    pts = pts_raw * (b - a) * 0.5
    wts = wts_raw * (b - a) * 0.5
    pts += 0.5 * (a + b)
    return pts, wts


def calculate_al_bl(index_ratio, size_parameter, l):
    return AlBlFunctions.calculate_al_bl(index_ratio, size_parameter, l)


class AlBlFunctions(object):
    @staticmethod
    def calculate_al_bl(index_ratio, size_parameter, l):
        """
        Mie scattering coefficients for expressing the scattered field in
        terms of vector spherical harmonics.

        ..math::

            a_l = \frac{n\, \psi_l(nx)\,\psi_l'(x)-\psi_l(x)\,\psi_l'(nx)}
                       {n\, \psi_l(nx) \,\\xi_l'(x)-\\xi_l(x)\,\psi_l'(nx)},

            b_l = \frac{\psi_l(nx)\,\psi_l'(x)-n\,\psi_l(x)\,\psi_l'(nx)}
                       {\psi_l(nx)\, \\xi_l'(x)-n\, \\xi_l(x)\,\psi_l'(nx)},

        where :math:`\psi_l` and :math:`\\xi_l` are the Riccati-Bessel
        functions of the first and third kinds, respectively.

        Parameters
        ----------
        index_ratio : float
              relative index of refraction
        size_paramter : float
              Size parameter
        l : int, array-like
              Order of scattering coefficient

        Returns
        -------
        a_l, b_l : numpy.ndarray
        """
        psi_nx = AlBlFunctions.riccati_psin(
            l, index_ratio * size_parameter)
        dpsi_nx = AlBlFunctions.riccati_psin(
            l, index_ratio * size_parameter, derivative=True)

        psi_x = AlBlFunctions.riccati_psin(l, size_parameter)
        dpsi_x = AlBlFunctions.riccati_psin(l, size_parameter, derivative=True)

        xi_x = AlBlFunctions.riccati_xin(l, size_parameter)
        dxi_x = AlBlFunctions.riccati_xin(l, size_parameter, derivative=True)

        a = (index_ratio * psi_nx * dpsi_x - psi_x * dpsi_nx) / (
             index_ratio * psi_nx * dxi_x - xi_x * dpsi_nx)
        b = (psi_nx * dpsi_x - index_ratio * psi_x * dpsi_nx) / (
             psi_nx * dxi_x - index_ratio * xi_x * dpsi_nx)
        return a, b

    @staticmethod
    def riccati_psin(n, z, derivative=False):
        """Riccati-Bessel function of the first kind or its derivative.

        .. math:: \psi_n(z) = z\,j_n(z),
        where :math:`j_n(z)` is the spherical Bessel function of the
        first kind.

        Parameters
         ----------
        n : int, array_like
              Order of the Bessel function (n >= 0).
        z : complex or float, array_like
              Argument of the Bessel function.
        derivative : bool, optional
              If True, the value of the derivative (rather than the function
              itself) is returned.

        Returns
        -------
        psin : ndarray
        """
        if derivative:
            ricatti = (z * spherical_jn(n, z, derivative=True) +
                       spherical_jn(n, z))
        else:
            ricatti = z * spherical_jn(n, z)
        return ricatti

    @staticmethod
    def riccati_xin(order, z, derivative=False):
        """Riccati-Bessel function of the third kind or its derivative.

        .. math:: \\xi_n(z) = z\,h^{(1)}_n(z),

        where :math:`h^{(1)}_n(z)` is the first spherical Hankel function.

        Parameters
        ----------
        n : int, array_like
              Order of the Bessel function (n >= 0).
        z : complex or float, array_like
              Argument of the Bessel function.
        derivative : bool, optional
              If True, the value of the derivative (rather than the function
              itself) is returned.

        Returns
        -------
        xin : ndarray
        """
        if derivative:
            ricatti = (z * spherical_h1n(order, z, derivative=derivative) +
                       spherical_h1n(order, z))
        else:
            ricatti = z * spherical_h1n(order, z)
        return ricatti


def calculate_pil_taul(theta, max_order):
    """
    The 1st through Nth order angle dependent functions for Mie scattering,
    evaluated at theta. The functions :math`\pi(\theta)` and :math`\tau(\theta)
    are defined as:

    ..math::

    \pi_n(\theta) = \frac{1}{\sin \theta} P_n^1(\cos\theta)

    \tau_n(\theta) = \frac{\mathrm{d}}{\mathrm{d}\theta} P_n^1(\cos\theta)

    where :math:`P_n^m` is the associated Legendre function. The functions are
    computed by upward recurrence using the relations

    ..math::

    \pi_n = \frac{2n-1}{n-1}\cos\theta \, \pi_{n-1} - \frac{n}{n-1}\pi_{n-2}

    \tau_n = n \, \cos\theta \, \pi_n - (n+1)\pi_{n-1}

    beginning with :math:`pi_0 = 0` and :math:`pi_1 = 1`

    Parameters
    ----------
    theta :  array_like
        angles (in radians) at which to evaluate the angular functions
    max_order : int > 0
        Order at which to halt iteration. Must be > 0

    Returns
    -------
    pi, tau : ndarray
        2D arrays with shape (len(theta), max_order) containing the
        values of the angular functions evaluated at theta up to order
        `max_order`
    """
    theta = np.atleast_1d(theta)
    cos_th = np.cos(theta)

    pi = np.zeros([max_order + 1, theta.size])
    tau = np.zeros([max_order + 1, theta.size])

    pi[1] = 1
    tau[1] = cos_th

    for n in range(2, max_order + 1):
        pi[n] = (2 * n - 1) / (n - 1) * cos_th * pi[n-1] - n / (n-1) * pi[n-2]
        tau[n] = n * cos_th * pi[n] - (n + 1) * pi[n-1]

    return pi[1:].T, tau[1:].T