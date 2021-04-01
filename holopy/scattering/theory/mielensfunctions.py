import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from numpy.polynomial.legendre import legval
from scipy.special import j0, j1, spherical_jn, spherical_yn
from scipy import interpolate

from holopy.scattering.errors import MissingParameter

NPTS = 100
LEGGAUSS_PTS_WTS_NPTS = np.polynomial.legendre.leggauss(NPTS)


# TODO:
# fast integration of oscillatory functions.


class MieLensCalculator(object):
    must_be_specified = [
        'particle_kz', 'index_ratio', 'size_parameter', 'lens_angle']

    def __init__(self, particle_kz=None, index_ratio=None, size_parameter=None,
                 lens_angle=None, quad_npts=100, interpolate_integrals='check',
                 interpolator_window_size=30.0, interpolator_degree=32):
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

        Methods
        -------
        calculate_scattered_field(krho, phi)
            tuple of 2 numpy.ndarrays, of shape krho
        calculate_total_field(krho, phi)
            tuple of 2 numpy.ndarrays, of shape krho
        calculate_total_intensity(krho, phi)
            numpy.ndarray, of shape krho

        Other Parameters
        ----------------
        quad_npts : int, optional
            The number of points for numerical quadrature of the
            integrals over the lens pupil.
        interpolate_integrals : {'check', True, False}
            Whether or not to interpolate the internally-evaluated
            integrals for speed. Default is `'check'`, which interpolates
            if it will be faster or does direct numerical quadrature
            otherwise. Interpolation is done via a piecewise Chebyshev
            approximant.
        interpolator_window_size : float, optional
            The spacing, in units of `1/k`, for the windows of the
            piecewise Chebyshev approximants. A lower value gives more
            accurate results, although accuracy depends on the
            `interpolator_degree` parameter as well. The default is 39.
            which gives 5e-13 relative accuracy.
        interpolator_degree : int, optional
            The polynomial degree for the piecewise Chebyshev
            approximants. A higher value gives more accurate results,
            although accuracy depends on the `interpolator_window_size`
            parameter as well. The default is 32, which gives 5e-13
            relative accuracy.

            It is best to leave the interpolator parameter as-is; they
            are only exposed for testing and advanced usage.
        """
        self.particle_kz = particle_kz
        self.index_ratio = index_ratio
        self.size_parameter = size_parameter
        self.lens_angle = lens_angle
        self._check_parameters()

        self.quad_npts = quad_npts
        self.interpolate_integrals = interpolate_integrals
        self.interpolator_window_size = interpolator_window_size
        self.interpolator_degree = interpolator_degree

        quad_pts, quad_wts = gauss_legendre_pts_wts(
            np.cos(self.lens_angle), 1.0, npts=self.quad_npts)

        # Precompute some quadrature points, mie functions that are
        # independent of rho and phi
        self._quad_pts = quad_pts.reshape(-1, 1)
        self._theta_pts = np.arccos(quad_pts)
        self._sintheta_pts = np.sin(self._theta_pts).reshape(-1, 1)
        self._quad_wts = quad_wts.reshape(-1, 1)

        self._precompute_scattering_matrices()

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

        output_x = np.zeros(shape, dtype='complex')
        output_y = np.zeros(shape, dtype='complex')

        # 1. Check for regions where rho is bad and leave as 0:
        rho_small = krho < 3.9 * self.quad_npts
        rho_large = ~rho_small

        # 2. Evaluate scattered fields only at valid rho's:
        if rho_small.any():
            ex_lowrho, ey_lowrho = self._calculate_small_krho_scattered_field(
                krho[rho_small], phi[rho_small])
            output_x[rho_small] = ex_lowrho
            output_y[rho_small] = ey_lowrho
        if rho_large.any():
            ex_hirho, ey_hirho = self._calculate_large_krho_scattered_field(
                krho[rho_large], phi[rho_large])
            output_x[rho_large] = ex_hirho
            output_y[rho_large] = ey_hirho

        return output_x, output_y

    def calculate_total_field(self, krho, phi):
        """The total (incident + scattered) field at the detector
        """
        scattered_x, scattered_y = self.calculate_scattered_field(krho, phi)
        incident_x, incident_y = self.calculate_incident_field()
        return incident_x + scattered_x, incident_y + scattered_y

    def calculate_total_intensity(self, krho, phi):
        fx, fy = self.calculate_total_field(krho, phi)
        return np.abs(fx)**2 + np.abs(fy)**2

    def calculate_incident_field(self):
        """This is here so
        (i)  Any corrections in the theory to the scattered field
             have an easy place to enter, and
        (ii) Other modules can consistently use the same scattered
             field as this module.
        """
        return -1, 0

    def _calculate_small_krho_scattered_field(self, krho, phi):
        shape = phi.shape
        i_0 = np.reshape(self._eval_mielens_i_n(krho, n=0), shape)
        i_2 = np.reshape(self._eval_mielens_i_n(krho, n=2), shape)
        c2p = np.cos(2 * phi)
        s2p = np.sin(2 * phi)
        field_xcomp = 0.5 * (i_0 + i_2 * c2p)
        field_ycomp = 0.5 * i_2 * s2p
        return field_xcomp, field_ycomp

    def _calculate_large_krho_scattered_field(self, krho, phi):
        # For now, just return 0s:
        zero = np.zeros(krho.shape, dtype='complex')
        return zero, zero

    def _precompute_scattering_matrices(self):
        kwargs = {'index_ratio': self.index_ratio,
                  'size_parameter': self.size_parameter,
                  }
        scat_s_evaluator = MieScatteringMatrix(
            parallel_or_perpendicular='perpendicular', **kwargs)
        scat_p_evaluator = MieScatteringMatrix(
            parallel_or_perpendicular='parallel', **kwargs)
        self._scat_perp_values = np.reshape(
            scat_s_evaluator._eval(self._theta_pts), (-1, 1))
        self._scat_prll_values = np.reshape(
            scat_p_evaluator._eval(self._theta_pts), (-1, 1))

    def _eval_mielens_i_n(self, krho, n=0):
        """Calculates one of several similar integrals over the lens
        pupil which appear in the Mie + lens calculations

        This should only be called by
        `self._calculate_small_krho_scattered_field`

        Parameters
        ----------
        krho : numpy.ndarray
            The rho values to evaluate the integrals at, in units of 1/k.
        n : {0, 2}, optional
            Which integral to evaluate; 0 for S + P, 2 for S - P.
            Default is 0; should always be passed though.

        Returns
        -------
        numpy.ndarray
            The value of the integrand evaluated at the krho points.
        """
        if self.interpolate_integrals == 'check':
            n_interp_pnts = (self.interpolator_degree * krho.ptp() /
                             self.interpolator_window_size)
            n_krho_pts = krho.size
            interpolate_integrals = n_interp_pnts < 1.1 * n_krho_pts
        else:
            interpolate_integrals = self.interpolate_integrals is True
        if interpolate_integrals:
            i_n = self._interpolate_and_eval_mielens_i_n(krho, n)
        else:
            i_n = self._direct_eval_mielens_i_n(krho, n)
        return i_n

    def _direct_eval_mielens_i_n(self, krho, n=0):
        if n == 0:
            ji = j0
            scatmatrix_values = self._scat_perp_values + self._scat_prll_values
        elif n == 2:
            ji = j2
            scatmatrix_values = self._scat_perp_values - self._scat_prll_values
        else:
            raise ValueError('n must be one of {0, 2}')
        # We do the integral with the change of variables x = cos(theta),
        # from cos(lens_angle) to 1.0:
        # Placing things in order [quadrature points, rho-z values]
        rr = krho.reshape(1, -1)
        phase = self._calculate_phase()
        integrand = (np.exp(1j * phase) * scatmatrix_values *
                     ji(rr * self._sintheta_pts) * np.sqrt(self._quad_pts))
        answer_flat = np.sum(integrand * self._quad_wts, axis=0)
        return answer_flat.reshape(krho.shape)

    def _interpolate_and_eval_mielens_i_n(self, krho, n=0):
        window_size = self.interpolator_window_size
        window_start = np.floor(krho.min() / window_size)
        window_end = np.ceil(krho.max() / window_size + 1e-4) + 1
        window_breakpoints = window_size * np.arange(window_start, window_end)

        interpolator = PiecewiseChebyshevApproximant(
            lambda x: self._direct_eval_mielens_i_n(x, n=n),
            degree=self.interpolator_degree,
            window_breakpoints=window_breakpoints)
        return interpolator(krho)

    def _calculate_phase(self):
        return self.particle_kz * (1 - self._quad_pts)

    def _check_parameters(self):
        if any([getattr(self, p) is None for p in self.must_be_specified]):
            msg = "{} must be specified.".format(self.must_be_specified)
            raise MissingParameter(msg)


class AberratedMieLensCalculator(MieLensCalculator):
    must_be_specified = [
        'particle_kz', 'index_ratio', 'size_parameter', 'lens_angle',
        'spherical_aberration']

    def __init__(self, spherical_aberration=None, **kwargs):
        """
        See `MieLensCalculator` for a more complete docstring.

        Parameters
        ----------
        spherical_aberration : float or array-like of floats
            The spherical aberration, up to arbitrary order. If a float,
            just the coefficient of the 3rd-order aberration (4th-order
            in wavefront). When an array, the coefficients of
            aberrations in ascending order (3rd, 5th, 7th, etc), where
            the wavefront distortion for the nth-order aberration is of
            the form (cos(theta) - 1)^(n+1), where n = 3, 5, 7, etc
            Default is None, which raises an error.

        Other Parameters
        ----------------
        See MieLensCalculator
        """
        self.spherical_aberration = spherical_aberration
        super(AberratedMieLensCalculator, self).__init__(**kwargs)

    def _calculate_phase(self):
        unaberrated_phase = (
            super(AberratedMieLensCalculator, self)._calculate_phase())
        aberrated_phase = self._calculate_aberrated_phase()
        return unaberrated_phase + aberrated_phase

    def _calculate_aberrated_phase(self):
        coeffs_high_to_low = np.reshape(self.spherical_aberration, -1)
        aberrated_phase = (
            self._pupil_x_squared**2 *
            legval(self._pupil_x_squared, coeffs_high_to_low))
        return aberrated_phase

    @property
    def _pupil_x_squared(self):
        # Actually (cos(theta) - 1) instead of theta^2. Making this
        # choice since this corresponds to the defocus from the particle
        return (self._quad_pts - 1)


class MieScatteringMatrix(object):
    def __init__(self,
                 parallel_or_perpendicular='perpendicular',
                 index_ratio=None,
                 size_parameter=None,
                 max_l=None):
        """Calculations of Mie far-field scattering matrices.

        These work by summing the Mie series naively; for large sizes
        this could be better by using an asymptotic representation.

        Parameters
        ----------
        parallel_or_perpendicular : {'parallel', 'perpendicular'}
            Whether to calculate the parallel (Pi, ~sin(phi)) or
            perpendicular (S, ~cos(phi)) scattering matrices.
        index_ratio : float
            Index contrast of the particle.
        size_parameter : float
            Size of the sphere in units of 1/k = 1/wavevector
        max_l : int > 0, optional
            The maximum order of the series to sum to. Defaults to a
            good value that trades off numerical accuracy (more terms)
            and lack-of-errors (less terms).
        """
        self.parallel_or_perpendicular = parallel_or_perpendicular
        self.index_ratio = index_ratio
        self.size_parameter = size_parameter
        self.max_l = self._default_max_l() if max_l is None else max_l

    def _default_max_l(self):
        """An empirically good value for ~1e-6 accuracy"""
        return np.ceil(25 + 1.1 * self.size_parameter).astype('int')

    def _eval(self, theta):
        """Evaluate S_parallel, perpendicular(theta) directly"""
        # Right now, the pi_l, tau_l functions calculate all values of
        # l at once. So we compute all at once then sum

        # The al, bl calculation can produce nan's if the maximum l
        # value is made aggressively large, due to weirdness in the
        # complex arithmetic standard. (The spherical Hankel functions
        # can be (0 + 1j * inf) when l is large. But, due to the
        # implementation of complex arithmetic in Python and numpy,
        # 0 + 1j*inf gets cast as nan + 1j*inf. We then divide a
        # non-infinite number by the spherical Hankel's nan + 1j*inf,
        # and that gives nan's when it should give 0.) To avoid this, we
        # truncate the series at a "reasonable" value of l while
        # checking that no nans actually appear in the calculation. We
        # do this by stopping the series if we get a nan, but checking
        # that the previous term in the series is close to 0:
        als_bls = list()
        for l in range(1, self.max_l + 1):
            this_al_bl = calculate_al_bl(
                self.index_ratio, self.size_parameter, l)
            if np.isnan(this_al_bl).any():
                previous_term_is_nonzero = np.any(np.abs(als_bls[-1]) > 1e-30)
                if previous_term_is_nonzero:
                    raise RuntimeError('nan for this value of theta, ka, max_l')
                break
            else:
                truncated_max_l = l
                als_bls.append(this_al_bl)

        # Now we proceed with the calculation, but using the truncated
        # max l instead of what the user requested:
        als, bls = [np.array(i) for i in zip(*als_bls)]
        coeffs = np.array([
            (2 * l + 1) / (l * (l + 1))
            for l in range(1, truncated_max_l + 1)]).reshape(1, -1)
        pils, tauls = calculate_pil_taul(theta, truncated_max_l)

        if self.parallel_or_perpendicular == 'perpendicular':
            ans = np.sum(coeffs * (bls * tauls + als * pils), axis=1)
        elif self.parallel_or_perpendicular == 'parallel':
            ans = np.sum(coeffs * (als * tauls + bls * pils), axis=1)
        if np.isnan(ans).any():
            raise RuntimeError('nan for this value of theta, ka, max_l')
        return ans

    def __call__(self, theta):
        return self._eval(theta)


def j2(x):
    """A fast J_2(x) defined in terms of other special functions """
    clipped = np.clip(x, 1e-15, np.inf)
    return 2. / clipped * j1(clipped) - j0(clipped)


def spherical_h1n(n, z, derivative=False):
    """Spherical Hankel function H_n(z) or its derivative"""
    return spherical_jn(n, z, derivative) + 1j * spherical_yn(n, z, derivative)


def spherical_h2n(n, z, derivative=False):
    """Spherical Hankel function H_n(z) or its derivative"""
    return spherical_jn(n, z, derivative) - 1j * spherical_yn(n, z, derivative)


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
    """
    Group of functions for calculating the Mie scattering coefficients,
    used for expressing the scattered field in terms of vector spherical
    harmonics.

    The coefficients `a_l`, `b_l` are defined as

    ..math::

        a_l = \frac{\psi_l(x) \psi_l'(nx) -  n \psi_l(nx) \psi_l'(x)}
                   {\\xi_l(x) \psi_l'(nx) - n \psi_l(nx)  \\xi_l'(x)},

        b_l = \frac{\psi_l(nx) \psi_l'(x) - n \psi_l(x) \psi_l'(nx)}
                   {\psi_l(nx) \\xi_l'(x) - n \\xi_l(x) \psi_l'(nx)},

    where :math:`\psi_l` and :math:`\\xi_l` are the Riccati-Bessel
    functions of the first and third kinds, respectively. The
    definitions used here follow those of van de Hulst [1]_, which
    differ from those used in Bohren and Huffman [2]_.

    References
    ----------
    .. [1] H. C. van de Hulst, "Light Scattering by Small Particles",
           Dover (1981), pg 123.
    .. [2] C. F. Bohren and Donald R. Huffman, "Absorption and
           Scattering of Light by Small Particles", Wiley (2004),
           pg 101.
    """

    @staticmethod
    def calculate_al_bl(index_ratio, size_parameter, l):
        """Returns `a_l` and `b_l`; see class docstring.

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

        a = (dpsi_nx * psi_x - index_ratio * psi_nx * dpsi_x) / (
             dpsi_nx * xi_x - index_ratio * psi_nx * dxi_x)
        b = (index_ratio * dpsi_nx * psi_x - psi_nx * dpsi_x) / (
             index_ratio * dpsi_nx * xi_x - psi_nx * dxi_x)
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
            ricatti = (z * spherical_h2n(order, z, derivative=derivative) +
                       spherical_h2n(order, z))
        else:
            ricatti = z * spherical_h2n(order, z)
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                           Interpolation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class PiecewiseChebyshevApproximant(object):
    def __init__(self, function, degree, window_breakpoints, *args):
        """
        Approximates on [window_breakpoints[0], window_breakpoints[1])
        """
        self.function = function
        self.degree = degree
        self.window_breakpoints = window_breakpoints
        self.args = args

        self._domain = (window_breakpoints[0], window_breakpoints[-1])
        self._windows = self._setup_windows()
        self._approximants = self._setup_approximants()
        self._dtype = self._approximants[0].coef.dtype

    def _setup_windows(self):
        windows = [
            (start, stop) for start, stop in
            zip(self.window_breakpoints[:-1], self.window_breakpoints[1:])]
        return windows

    def _setup_approximants(self):
        return [Chebyshev.interpolate(
                    self.function, self.degree, domain=window, *self.args)
                for window in self._windows]

    def __call__(self, x):
        x = np.asarray(x)
        if x.max() >= self._domain[1] or x.min() < self._domain[0]:
            msg = "x must be within interpolation window [{}, {})".format(
                *self._domain)
            raise ValueError(msg)
        result = np.zeros(x.shape, dtype=self._dtype)
        for window, approximant in zip(self._windows, self._approximants):
            mask = self._mask_window(x, window)
            result[mask] = approximant(x[mask])
        return result

    @classmethod
    def _mask_window(cls, x, window):
        return (x >= window[0]) & (x < window[1])

