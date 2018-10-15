# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, Ryan McGorty, Anna Wang, Solomon Barkley
#
# This file is part of HoloPy.
#
# HoloPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HoloPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HoloPy.  If not, see <http://www.gnu.org/licenses/>.
"""
Calculates holograms of spheres using an analytical solution of
the Mie scattered field imaged by a perfect lens.
Uses superposition to calculate scattering from multiple spheres.

.. moduleauthor:: Brian D. Leahy <bleahy@seas.harvard.edu>
.. moduleauthor:: Ron Alexander <ralexander@g.harvard.edu>
"""

import numpy as np
from scipy.special import j0, j1, spherical_jn, spherical_yn
from scipy import interpolate

# from ...core.utils import ensure_array
# from ..errors import TheoryNotCompatibleError, InvalidScatterer
from ..scatterer import Sphere, Scatterers
from .scatteringtheory import ScatteringTheory


class MieLens(ScatteringTheory):
    def __init__(self, lens_angle=1.0, calculator_kwargs={}):
        # some things to add -- number of interpolator points
        super(MieLens, self).__init__()
        self.lens_angle = lens_angle
        self._check_calculator_kwargs(calculator_kwargs)
        self.calculator_kwargs = calculator_kwargs

    def _check_calculator_kwargs(self, calculator_kwargs):
        msg = ("`calculator_kwargs` must be a dict with keys `'quad_npts'`," +
               "`'interpolator_maxl'`, and/or `'interpolator_npts'`" +
               ", all with integer values.")
        try:
            keys = {k for k in calculator_kwargs.keys()}
        except:
            raise ValueError(msg)
        valid_keys = {'quad_npts', 'interpolator_maxl', 'interpolator_npts'}
        if any([k not in valid_keys for k in keys]):
            raise ValueError(msg)

    def _can_handle(self, scatterer):
        return isinstance(scatterer, Sphere)

    # The only thing I will implement for now
    def _raw_fields(self, positions, scatterer, medium_wavevec, medium_index,
                    illum_polarization):
        """
        Parameters
        ----------
        positions : (3, N) numpy.ndarray
            The (r, theta, phi) coordinates, relative to the sphere,
            of the points to calculate the fields.
        scatterer : ``scatterer.Sphere`` object
        medium_wavevec : float
        medium_index : float
        illum_polarization : 2-element tuple
            The (x, y) field polarizations.
        """
        index_ratio = scatterer.n / medium_index
        size_parameter = medium_wavevec * scatterer.r

        r, theta, phi = positions
        z = r * np.cos(theta)
        rho = r * np.sin(theta)
        phi += np.arctan2(illum_polarization.values[1],
                          illum_polarization.values[0])
        phi %= (2 * np.pi)

        # FIXME mielens assumes that the detector points are at a fixed z!
        # right now I'm picking one z:
        particle_z = np.mean(z)
        if np.ptp(z) / particle_z > 1e-13:
            msg = ("mielens currently assumes the detector is a fixed "+
                  "z from the particle")
            raise ValueError(msg)
        particle_kz = medium_wavevec * particle_z

        field_calculator = MieLensCalculator(
            particle_kz=particle_kz, index_ratio=index_ratio,
            size_parameter=size_parameter, lens_angle=self.lens_angle,
            **self.calculator_kwargs)
        fields_x, fields_y = field_calculator.calculate_scattered_field(
            medium_wavevec * rho, phi)
        field_xyz = np.zeros([3, fields_x.size], dtype='complex')
        field_xyz[0, :] = fields_x
        field_xyz[1, :] = fields_y
        return field_xyz


class MieLensCalculator(object):
    def __init__(self, particle_kz=10.0, index_ratio=1.1, size_parameter=10.0,
                 lens_angle=1.0, quad_npts=100, interpolator_maxl=None,
                 interpolator_npts=None):
        """Calculates the field from a Mie scatterer imaged in a high-NA lens.

        The incindent electric field is E e^{ikz}, with the particle
        position at z. The scattered field takes into account the
        varying phase of the incoming field.

        Parameters
        ----------
        particle_kz : float
            + z is away from the lens
        index_ratio : float
        size_parameter : float
        lens_angle : float
        quad_npts : int, optional
        interpolator_maxl : int or None, optional
        interpolator_npts : int or None, optional

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
        self.interpolator_maxl = interpolator_maxl
        self.interpolator_npts = interpolator_npts

        self._setup_interpolators()
        self._quad_pts, self._quad_wts = gauss_legendre_pts_wts(
            np.cos(self.lens_angle), 1.0, npts=self.quad_npts)

    def _setup_interpolators(self):
        kwargs = {'index_ratio': self.index_ratio,
                  'size_parameter': self.size_parameter,
                  'max_l': self.interpolator_maxl,
                  'npts': self.interpolator_npts}
        self._interpolator_f1 = FarfieldMieInterpolator(i=1, **kwargs)
        self._interpolator_f2 = FarfieldMieInterpolator(i=2, **kwargs)

    def mielens_i_ij(self, krho, fi=None, j=0):
        """Calculates one of several similar integrals over the lens pupil
        which appear in the Mie + lens calculations.

        Parameters
        ----------
        krho : numpy.ndarrays
            The rho values to evaluate the integrals at, in units of 1/k.
        fi : FarfieldMieInterpolator object or None, optional
            Pass a FarfieldMieInterpolator interpolator for speed...
        j, i : int, optional
            The integers which determine which of the Mie fields to evaluate.

        Returns
        -------
        numpy.ndarray
            The value of the integrand evaluated at the krho points.
        """
        if fi is None:  # should be always passed...
            fi = self._interpolator_f1
        if j == 0:
            ji = j0
        elif j == 2:
            ji = j2
        else:
            raise ValueError('j must be one of {0, 2}')
        # We do the integral with the change of variables x = cos(theta),
        # from cos(lens_angle) to 1.0:
        # Placing things in order [quadratrue points, rho-z values]
        pts = self._quad_pts.reshape(-1, 1)
        wts = self._quad_wts.reshape(-1, 1)
        rr = krho.reshape(1, -1)
        theta = np.arccos(pts)  # for F_i(theta)
        sintheta = np.sqrt(1 - pts**2)
        integrand = (np.exp(-1j * self.particle_kz * (pts - 1)) * fi(theta) *
                     ji(rr * sintheta))
        return np.sum(integrand * wts, axis=0)

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

        Other Parameters
        ----------------
        Interpolation parameters to FarfieldMieInterpolator

        Notes
        -----
        This will have problems for large rho, z, because of the quadrature
        points. Could be adaptive if needed....
        """
        # 0. Check inputs:
        shp = krho.shape
        if (shp != phi.shape):
            raise ValueError('krho, phi must all be the same shape')
        # 2. Evaluate the integrals:
        i_10 = np.reshape(self.mielens_i_ij(krho, j=0,
                          fi=self._interpolator_f1), shp)
        i_12 = np.reshape(self.mielens_i_ij(krho, j=2,
                          fi=self._interpolator_f1), shp)
        i_20 = np.reshape(self.mielens_i_ij(krho, j=0,
                          fi=self._interpolator_f2), shp)
        i_22 = np.reshape(self.mielens_i_ij(krho, j=2,
                          fi=self._interpolator_f2), shp)
        # 3. Sum for the field:
        c2p = np.cos(2 * phi)
        s2p = np.sin(2 * phi)
        field_xcomp = -i_10 - i_20 + (i_12 - i_22) * c2p
        field_ycomp = (i_12 - i_22) * s2p
        return field_xcomp.reshape(shp), field_ycomp.reshape(shp)

    def calculate_total_field(self, krho, phi):
        """The total (incident + scattered) field at the detector
        """
        # Uses the incident field as
        #   E_in = E_0 \hat{x} * 4 pi * (f1 / f2) * e^{ik(f1 + f2)} * i
        # which is more-or-less from the brightfield writeups.
        # return 1j - 0.25 * mielens_field(krho, phi, **kwargs)
        fx, fy = self.calculate_scattered_field(krho, phi)
        return 1j - 0.25 * fx, 0.25 * fy

    def calculate_total_intensity(self, krho, phi):
        fx, fy = self.calculate_total_field(krho, phi)
        return np.abs(fx)**2 + np.abs(fy)**2


class FarfieldMieInterpolator(object):
    def __init__(self, i=1, index_ratio=1.1, size_parameter=1.0, max_l=None,
                 npts=None):
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
        """
        # init sets up interpolator
        self.i = i
        self.index_ratio = index_ratio
        self.size_parameter = size_parameter
        self.max_l = self._default_max_l() if max_l is None else max_l
        self.npts = self._default_npts() if npts is None else npts
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
        return self._interp(theta)


def j2(x):
    """A fast J_2(x) defined in terms of other special functions """
    clipped = np.clip(x, 1e-15, np.inf)
    return 2. / clipped * j1(clipped) - j0(clipped)


def spherical_h1n(n, z, derivative=False):
    """Spherical Hankel function H_n(z) or its derivative"""
    return spherical_jn(n, z, derivative) + 1j * spherical_yn(n, z, derivative)


NPTS = 100
def gauss_legendre_pts_wts(a, b, npts=NPTS):
    """Quadrature points for integration on interval [a, b]"""
    pts, wts = np.polynomial.legendre.leggauss(npts)
    pts *= (b - a) * 0.5
    wts *= (b - a) * 0.5
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
        2D arrays with shape (len(theta), max_order) containing the values of the
        angular functions evaluated at theta up to order max_order
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
